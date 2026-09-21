"""Observation potentials independent of diffusion models and samplers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from afdps._validation import finite_scalar
from afdps.interfaces import ForwardOperator


@dataclass(frozen=True)
class GaussianLikelihood:
    """mu_y(x) = ||A(x) - y||^2 / (2 * noise_std^2), Eq. (2.2).

    Additive constants are omitted. noise_std describes iid MEASUREMENT noise,
    not the diffusion noise level. observation is one unbatched measurement;
    the only broadcast allowed is across the leading particle axis. The
    operator may be nonlinear and must preserve autograd for derivatives.
    """

    operator: ForwardOperator
    observation: Tensor
    noise_std: float

    def __post_init__(self) -> None:
        if finite_scalar(self.noise_std, "noise_std") <= 0:
            raise ValueError("noise_std must be positive and finite")
        if self.observation.dtype not in (torch.float32, torch.float64):
            raise TypeError("observation must use torch.float32 or torch.float64")
        if self.observation.numel() == 0 or not torch.isfinite(self.observation).all():
            raise ValueError("observation must be nonempty and finite")

    def __call__(self, positions: Tensor) -> Tensor:
        if positions.ndim < 2 or positions.numel() == 0:
            raise ValueError("positions must have nonempty shape (N, *state_shape)")
        if (
            positions.dtype != self.observation.dtype
            or positions.device != self.observation.device
        ):
            raise ValueError("positions and observation must share dtype and device")
        predicted = self.operator(positions)
        expected_shape = (positions.shape[0], *self.observation.shape)
        if predicted.shape != expected_shape:
            raise ValueError(
                f"operator returned shape {tuple(predicted.shape)}; "
                f"expected {expected_shape} (one fixed observation)"
            )
        if predicted.dtype != positions.dtype or predicted.device != positions.device:
            raise ValueError("operator must preserve dtype and device")
        residual = (predicted - self.observation) / self.noise_std
        return 0.5 * residual.reshape(positions.shape[0], -1).square().sum(dim=1)
