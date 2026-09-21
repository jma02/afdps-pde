"""Exact Stage I sampling for small, dense linear-Gaussian inverse problems."""

from dataclasses import dataclass

import torch
from torch import Tensor

from afdps._validation import (
    check_count,
    check_generator,
    finite_scalar,
    full_precision,
)
from afdps.likelihoods import GaussianLikelihood
from afdps.operators import DenseLinearOperator
from afdps.particles import ParticleEnsemble


@dataclass(frozen=True)
class Gaussian:
    """Prepared Gaussian snapshot; private cached tensors must not be mutated.

    Use condition_gaussian to prepare detached parameters once, then reuse them
    with sample_gaussian. Public diagnostics return independent tensors.
    """

    _mean: Tensor
    _precision_cholesky: Tensor

    @property
    def mean(self) -> Tensor:
        """Return an independent copy of the mean (d,)."""
        return self._mean.clone()

    @property
    @torch.no_grad()
    def covariance(self) -> Tensor:
        """Compute the dense covariance (d, d) on demand for diagnostics."""
        identity = torch.eye(
            self._mean.numel(), dtype=self._mean.dtype, device=self._mean.device
        )
        with full_precision(self._mean.device):
            covariance = torch.cholesky_solve(identity, self._precision_cholesky)
        if not torch.isfinite(covariance).all():
            raise ValueError(
                "Gaussian covariance is nonfinite in this dtype; "
                "rescale the problem or use float64"
            )
        return covariance


@torch.no_grad()
def condition_gaussian(
    likelihood: GaussianLikelihood, *, reference_std: float
) -> Gaussian:
    """Condition N(0, reference_std**2 I) on a dense linear-Gaussian observation.

    Snapshot inputs into detached mean and precision Cholesky; preparation costs
    O(m*d^2 + d^3) time and O(d^2) storage. No jitter or matrix inverse is used.
    """
    if not isinstance(likelihood, GaussianLikelihood):
        raise TypeError("likelihood must be a GaussianLikelihood")
    if not isinstance(likelihood.operator, DenseLinearOperator):
        raise TypeError("likelihood.operator must be a DenseLinearOperator")
    reference_std = finite_scalar(reference_std, "reference_std")
    if reference_std <= 0:
        raise ValueError("reference_std must be positive and finite")
    matrix, observation = likelihood.operator.matrix, likelihood.observation
    if observation.shape != (matrix.shape[0],):
        raise ValueError("observation must have shape (matrix.shape[0],)")
    if observation.dtype != matrix.dtype or observation.device != matrix.device:
        raise ValueError("matrix and observation must share dtype and device")
    if not torch.isfinite(matrix).all() or not torch.isfinite(observation).all():
        raise ValueError("matrix and observation must contain only finite values")

    with full_precision(matrix.device):
        rho = matrix.new_tensor(reference_std)
        sigma = matrix.new_tensor(likelihood.noise_std)
        if not torch.isfinite(sigma) or sigma <= 0:
            raise ValueError("noise_std is not representable in the matrix dtype")
        reference_precision = rho.reciprocal().square()
        if not torch.isfinite(reference_precision) or reference_precision <= 0:
            raise ValueError(
                "reference_std produces unrepresentable precision; "
                "rescale the problem or use float64"
            )
        whitened_matrix = matrix / sigma
        whitened_observation = observation / sigma
        precision = whitened_matrix.T @ whitened_matrix
        precision.diagonal().add_(reference_precision)
        natural_mean = whitened_matrix.T @ whitened_observation
        if (
            not torch.isfinite(precision).all()
            or not torch.isfinite(natural_mean).all()
        ):
            raise ValueError(
                "Gaussian precision or natural mean is nonfinite; "
                "rescale the problem or use float64"
            )
        try:
            factor = torch.linalg.cholesky(precision)
        except torch.linalg.LinAlgError as error:
            raise ValueError(
                "Gaussian precision is not numerically positive definite; "
                "rescale the problem or use float64 (no jitter is added)"
            ) from error
        mean = torch.cholesky_solve(natural_mean.unsqueeze(1), factor).squeeze(1)
        if not torch.isfinite(mean).all():
            raise ValueError("Gaussian mean is nonfinite; rescale the problem")
    return Gaussian(mean, factor)


@torch.no_grad()
def sample_gaussian(
    gaussian: Gaussian, num_particles: int, *, generator: torch.Generator
) -> ParticleEnsemble:
    """Draw independent samples with uniform weights using only the given generator."""
    check_count(num_particles, "num_particles")
    mean = gaussian._mean
    check_generator(generator, mean.device)
    noise = torch.randn(
        num_particles,
        mean.numel(),
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    # Precision is L L.T: solve with L.T, not L, for covariance L^{-T} L^{-1}.
    with full_precision(mean.device):
        offsets = torch.linalg.solve_triangular(
            gaussian._precision_cholesky.T, noise.T, upper=True
        ).T
    return ParticleEnsemble.uniform(mean + offsets)
