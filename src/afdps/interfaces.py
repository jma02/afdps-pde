"""Structural contracts for one observation and independent particles.

Positions are (N, *state_shape); operators return (N, *observation_shape).
A potential returns (N,), retaining spatial autograd, including for constants.
Scores preserve shape/dtype/device and take INCREASING reverse time: noisy t=0,
clean t=T. Noise/checkpoint adaptation and model.eval() belong to the caller.
Components must not mutate inputs, couple particles, or use hidden randomness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from torch import Generator, Tensor

if TYPE_CHECKING:
    from afdps.particles import ParticleEnsemble

ForwardOperator = Callable[[Tensor], Tensor]
Potential = Callable[[Tensor], Tensor]
ScoreFunction = Callable[[Tensor, float], Tensor]


class DiffusionSchedule(Protocol):
    """Forward F(T-t), G(T-t) on the reverse clock, not reversed drift/noise level."""

    @property
    def terminal_time(self) -> float:
        """Positive finite horizon T."""
        ...

    def forward_drift(self, reverse_time: float) -> float: ...

    def forward_diffusion(self, reverse_time: float) -> float: ...


class Initializer(Protocol):
    """Draw the conditioned t=0 law, with corrected weights for proposal draws.

    Shape/dtype/device and the noisy reference must match the score and schedule.
    An explicit same-device generator is required; exact draws have equal weights.
    """

    def __call__(
        self, num_particles: int, *, generator: Generator
    ) -> ParticleEnsemble: ...
