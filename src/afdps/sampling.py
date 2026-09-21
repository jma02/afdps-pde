"""Functional two-stage AFDPS sampling with explicit clocks and randomness."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import torch

from afdps._validation import check_generator, finite_scalar, full_precision
from afdps.config import SamplerConfig
from afdps.derivatives import potential_derivatives
from afdps.dynamics import afdps_dynamics, euler_step, ula_corrector
from afdps.interfaces import DiffusionSchedule, Initializer, Potential, ScoreFunction
from afdps.particles import ParticleEnsemble


@dataclass(frozen=True)
class StepDiagnostics:
    """One interval; ESS fraction is measured BEFORE resampling."""

    step: int
    reverse_time: float
    step_size: float
    ess_fraction: float
    resampled: bool


@dataclass(frozen=True)
class SamplingResult:
    """Final weighted ensemble and scalar-only history, not an unweighted draw."""

    ensemble: ParticleEnsemble
    diagnostics: tuple[StepDiagnostics, ...]


@torch.no_grad()
def sample(
    score: ScoreFunction,
    potential: Potential,
    schedule: DiffusionSchedule,
    initial: Union[ParticleEnsemble, Initializer],
    *,
    generator: torch.Generator,
    config: Optional[SamplerConfig] = None,
    time_grid: Optional[Sequence[float]] = None,
) -> SamplingResult:
    """Run Stage II from t=0, optionally drawing Stage I from a callable initial.

    Pass a conditioned ensemble or an initializer(num_particles, generator=...).
    The grid is validated before initialization; initial weights are retained.
    Inputs are not mutated, outputs are detached, and models must already use
    the reverse clock. No jitter, clipping, or terminal denoising is applied.
    """
    config = SamplerConfig() if config is None else config
    check_generator(generator)
    horizon = finite_scalar(schedule.terminal_time, "terminal_time")
    if horizon <= 0:
        raise ValueError("terminal_time must be positive")
    grid = (
        tuple(horizon * (k / config.num_steps) for k in range(config.num_steps + 1))
        if time_grid is None
        else tuple(finite_scalar(time, "time_grid entry") for time in time_grid)
    )
    if len(grid) != config.num_steps + 1:
        raise ValueError("time_grid must have num_steps + 1 entries")
    if grid[0] != 0.0 or grid[-1] != horizon:
        raise ValueError("time_grid must start at 0 and end at terminal_time")
    if any(end <= start for start, end in zip(grid, grid[1:])):
        raise ValueError("time_grid must be strictly increasing")
    if not isinstance(initial, ParticleEnsemble) and callable(initial):
        initial = initial(config.num_particles, generator=generator)
    if not isinstance(initial, ParticleEnsemble):
        raise TypeError("initial must be a ParticleEnsemble or return one")
    if initial.num_particles != config.num_particles:
        raise ValueError("initial ensemble must contain config.num_particles particles")
    check_generator(generator, initial.positions.device)
    current = ParticleEnsemble(
        initial.positions.detach().clone(), initial.log_weights.detach().clone()
    )
    history = []
    with full_precision(current.positions.device):
        for step, (time, next_time) in enumerate(zip(grid, grid[1:]), start=1):
            dt = next_time - time
            f = schedule.forward_drift(time)
            g = finite_scalar(schedule.forward_diffusion(time), "forward_diffusion")
            score_values = score(current.positions, time)
            derivatives = potential_derivatives(
                potential,
                current.positions,
                compute_laplacian=config.variant == "sde" and g > 0,
            )
            dynamics = afdps_dynamics(
                current.positions,
                score_values,
                derivatives,
                forward_drift=f,
                forward_diffusion=g,
                variant=config.variant,
            )
            # Predictor AND weights use the old state; ULA uses the next time.
            current = euler_step(current, dynamics, step_size=dt, generator=generator)
            if config.corrector is not None:
                corrected = ula_corrector(
                    current.positions,
                    score,
                    potential,
                    reverse_time=next_time,
                    config=config.corrector,
                    generator=generator,
                )
                current = ParticleEnsemble(corrected, current.log_weights)
            ess = current.effective_sample_size_fraction.item()
            threshold = config.resample_ess_fraction
            resampled = threshold is not None and ess < threshold
            if resampled:
                current, _ = current.multinomial_resample(generator=generator)
            history.append(StepDiagnostics(step, next_time, dt, ess, resampled))
    return SamplingResult(current, tuple(history))
