"""Validated sampler controls, not paper-experiment presets."""

from dataclasses import dataclass
from typing import Literal, Optional

from afdps._validation import check_count, finite_scalar

SamplerVariant = Literal["sde", "ode"]


@dataclass(frozen=True)
class CorrectorConfig:
    """ULA steps per ODE predictor; step_size is Langevin time, not diffusion time.

    Zero steps explicitly selects the weighted ODE without a corrector. Finite
    positive steps introduce ULA discretization bias; no Metropolis step is used.
    """

    num_steps: int = 1
    step_size: float = 1e-3

    def __post_init__(self) -> None:
        check_count(self.num_steps, "corrector num_steps", allow_zero=True)
        if finite_scalar(self.step_size, "corrector step_size") <= 0:
            raise ValueError("corrector step_size must be positive")


@dataclass(frozen=True)
class SamplerConfig:
    """Algorithm controls independent of model, device, and state shape.

    num_steps counts Stage II intervals. A caller-supplied grid must have
    num_steps + 1 entries. Resample when ESS / N < resample_ess_fraction;
    None disables resampling. ODE requires an explicit CorrectorConfig so the
    Langevin step size is a conscious choice; SDE does not use a corrector.
    """

    variant: SamplerVariant = "sde"
    num_particles: int = 16
    num_steps: int = 100
    resample_ess_fraction: Optional[float] = 0.5
    corrector: Optional[CorrectorConfig] = None

    def __post_init__(self) -> None:
        if self.variant not in ("sde", "ode"):
            raise ValueError("variant must be 'sde' or 'ode'")
        check_count(self.num_particles, "num_particles")
        check_count(self.num_steps, "num_steps")
        if self.resample_ess_fraction is not None:
            threshold = finite_scalar(
                self.resample_ess_fraction, "resample_ess_fraction"
            )
            if not 0 < threshold <= 1:
                raise ValueError("resample_ess_fraction must be in (0, 1] or None")
        if self.variant == "ode" and not isinstance(self.corrector, CorrectorConfig):
            raise ValueError("ODE requires an explicit CorrectorConfig")
        if self.variant == "sde" and self.corrector is not None:
            raise ValueError("corrector is only supported for the ODE variant")
