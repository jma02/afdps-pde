"""Exact spatial derivatives of independent, per-particle potentials."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from afdps._validation import check_positions, check_tensor, full_precision
from afdps.interfaces import Potential
from afdps.likelihoods import GaussianLikelihood
from afdps.operators import DenseLinearOperator


@dataclass(frozen=True)
class PotentialDerivatives:
    """Detached gradient shaped like positions and optional trace shaped (N,)."""

    gradient: Tensor
    laplacian: Optional[Tensor]


def potential_derivatives(
    potential: Potential, positions: Tensor, *, compute_laplacian: bool = True
) -> PotentialDerivatives:
    """Differentiate only positions, without accumulating parameter gradients.

    Particles must be independent. The generic trace uses one backward pass per
    event coordinate, not a stochastic trace estimator. Detached potentials are
    rejected; express a constant as ``(0 * x).flatten(1).sum(1) + constant`` to
    retain its spatial graph. Results retain neither caller nor derivative graphs.
    """
    check_positions(positions)

    analytic = (
        type(potential) is GaussianLikelihood
        and type(potential.operator) is DenseLinearOperator
    )
    # Clone after disabling inference mode: even inference-tensor inputs must
    # become ordinary tensors, without altering their storage or requires_grad.
    with (
        torch.inference_mode(False),
        torch.set_grad_enabled(not analytic),
        full_precision(positions.device),
    ):
        x = positions.detach().clone().requires_grad_(not analytic)
        values = potential(x)
        check_tensor(values, x, "potential", shape=(x.shape[0],))

        laplacian = None
        if analytic:
            matrix = potential.operator.matrix / potential.noise_std
            observation = potential.observation / potential.noise_std
            gradient = (x @ matrix.T - observation) @ matrix
            if compute_laplacian:
                laplacian = matrix.square().sum().expand(x.shape[0])
        else:
            if not values.requires_grad:
                raise ValueError("potential must retain a spatial autograd graph")
            # Explicit seeds avoid legacy MPS ones_like/implicit-seed fill bugs.
            ones = x.new_tensor(1.0).expand(x.shape[0])
            gradient = torch.autograd.grad(
                values,
                x,
                grad_outputs=ones,
                create_graph=compute_laplacian,
                allow_unused=True,
            )[0]
            if gradient is None:
                raise ValueError("potential must retain a spatial autograd graph")
            if compute_laplacian:
                flat_gradient = gradient.reshape(x.shape[0], -1)
                laplacian = x.new_zeros(x.shape[0])
                for dimension in range(flat_gradient.shape[1]):
                    component = flat_gradient[:, dimension]
                    if not component.requires_grad:
                        continue  # A constant spatial gradient has zero curvature.
                    hessian_row = torch.autograd.grad(
                        component,
                        x,
                        grad_outputs=ones,
                        retain_graph=dimension + 1 < flat_gradient.shape[1],
                        allow_unused=True,
                    )[0]
                    # Parameter-dependent affine gradients can require grad
                    # without depending on x; their spatial Hessian is also zero.
                    if hessian_row is not None:
                        laplacian += hessian_row.reshape(x.shape[0], -1)[:, dimension]

        gradient = gradient.detach()
        if not torch.isfinite(gradient).all():
            raise ValueError("potential gradient must contain only finite values")
        if laplacian is not None:
            laplacian = laplacian.detach()
            if not torch.isfinite(laplacian).all():
                raise ValueError("potential laplacian must contain only finite values")
        return PotentialDerivatives(gradient=gradient, laplacian=laplacian)
