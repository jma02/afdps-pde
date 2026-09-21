"""Weighted ensembles and standalone SMC bookkeeping; no AFDPS dynamics here."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from afdps._validation import check_generator, check_positions


@dataclass(frozen=True)
class ParticleEnsemble:
    """States (N, *state_shape) with unnormalized log weights (N,).

    Zero-mass particles may have log weight -inf; at least one must have finite
    mass. Tensors must share float32/float64 dtype and device; lower precision
    can underflow squared weights and corrupt ESS. Methods return new
    ensembles rather than mutating inputs. Frozen fields do NOT make tensor
    storage immutable: callers must not mutate stored tensors in place.
    """

    positions: Tensor
    log_weights: Tensor

    def __post_init__(self) -> None:
        check_positions(self.positions)
        if self.log_weights.shape != (self.positions.shape[0],):
            raise ValueError("log_weights must have shape (N,)")
        if (
            self.log_weights.dtype != self.positions.dtype
            or self.log_weights.device != self.positions.device
        ):
            raise ValueError("positions and log_weights must share dtype and device")
        if (
            torch.isnan(self.log_weights).any()
            or (self.log_weights == float("inf")).any()
        ):
            raise ValueError("log_weights may not contain NaN or +inf")
        if not torch.isfinite(self.log_weights).any():
            raise ValueError("at least one particle must have finite log weight")

    @classmethod
    def uniform(cls, positions: Tensor) -> ParticleEnsemble:
        """Assign equal unnormalized weights to existing states (no sampling)."""
        if positions.ndim < 2:
            raise ValueError("positions must have shape (N, *state_shape)")
        return cls(positions, positions.new_zeros(positions.shape[0]))

    @property
    def num_particles(self) -> int:
        return self.positions.shape[0]

    @property
    def normalized_log_weights(self) -> Tensor:
        """Normalize stably, including when all finite weights have a huge offset."""
        return torch.log_softmax(self.log_weights, dim=0)

    @property
    def weights(self) -> Tensor:
        return self.normalized_log_weights.exp()

    @property
    def effective_sample_size(self) -> Tensor:
        """ESS = 1 / sum(w_i^2), a scalar tensor in [1, N] up to roundoff."""
        return self.weights.square().sum().reciprocal()

    @property
    def effective_sample_size_fraction(self) -> Tensor:
        """ESS / N, the quantity compared with c in the paper's Algorithm 1."""
        return self.effective_sample_size / self.num_particles

    def mean(self) -> Tensor:
        """Weighted mean with shape state_shape; not a posterior draw."""
        weight_shape = (self.num_particles,) + (1,) * (self.positions.ndim - 1)
        return (self.weights.reshape(weight_shape) * self.positions).sum(dim=0)

    def multinomial_resample(
        self, *, generator: torch.Generator
    ) -> tuple[ParticleEnsemble, Tensor]:
        """Draw N ancestors, reset to equal weights, and return ancestor indices.

        The generator must be on the particles' device. This operation is not a
        differentiable sampling path. The input ensemble remains unchanged.
        """
        check_generator(generator, self.positions.device)
        ancestors = torch.multinomial(
            self.weights,
            self.num_particles,
            replacement=True,
            generator=generator,
        )
        positions = self.positions.index_select(0, ancestors)
        return ParticleEnsemble.uniform(positions), ancestors
