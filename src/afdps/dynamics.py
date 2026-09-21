"""Independently testable AFDPS particle equations and first-order kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from afdps._validation import (
    check_generator,
    check_positions,
    check_tensor,
    finite_scalar,
    full_precision,
)
from afdps.config import CorrectorConfig, SamplerVariant
from afdps.derivatives import PotentialDerivatives, potential_derivatives
from afdps.interfaces import Potential, ScoreFunction
from afdps.particles import ParticleEnsemble


@dataclass(frozen=True)
class ParticleDynamics:
    """Eq. (3.6): spatial drift, uncentered log-weight rate, Brownian amplitude."""

    drift: Tensor
    log_weight_rate: Tensor
    diffusion_std: float


@torch.no_grad()
def afdps_dynamics(
    positions: Tensor,
    score_values: Tensor,
    derivatives: PotentialDerivatives,
    *,
    forward_drift: float,
    forward_diffusion: float,
    variant: SamplerVariant,
) -> ParticleDynamics:
    """Compute canonical eta=1 AFDPS coefficients in increasing reverse time.

    H = -F(T-t)*x + (G(T-t)^2 + V^2)/2 * score. The SDE uses V=G(T-t);
    the ODE predictor uses V=0. No score derivatives or denoising are needed.
    Likelihood derivatives must be evaluated at the same OLD state as score.
    """
    check_positions(positions)
    check_tensor(score_values, positions, "score")
    check_tensor(derivatives.gradient, positions, "potential gradient")
    f = finite_scalar(forward_drift, "forward_drift")
    g = finite_scalar(forward_diffusion, "forward_diffusion")
    if g < 0:
        raise ValueError("forward_diffusion must be nonnegative")
    if variant not in ("sde", "ode"):
        raise ValueError("variant must be 'sde' or 'ode'")
    g_squared = finite_scalar(g * g, "squared forward_diffusion")
    v_squared = g_squared if variant == "sde" else 0.0
    with full_precision(positions.device):
        prior_drift = (
            -f * positions + (0.5 * g_squared + 0.5 * v_squared) * score_values
        )
        gradient = derivatives.gradient
        drift = prior_drift - v_squared * gradient
        rate = -(prior_drift * gradient).flatten(1).sum(dim=1)
        if v_squared > 0:
            laplacian = derivatives.laplacian
            if laplacian is None:
                raise ValueError("SDE dynamics require the potential Laplacian")
            check_tensor(
                laplacian,
                positions,
                "potential Laplacian",
                shape=(positions.shape[0],),
            )
            rate = rate + 0.5 * v_squared * (
                gradient.square().flatten(1).sum(dim=1) - laplacian
            )
    check_tensor(drift, positions, "drift")
    check_tensor(rate, positions, "log-weight rate", shape=(positions.shape[0],))
    return ParticleDynamics(
        drift.detach(), rate.detach(), g if variant == "sde" else 0.0
    )


@torch.no_grad()
def euler_step(
    ensemble: ParticleEnsemble,
    dynamics: ParticleDynamics,
    *,
    step_size: float,
    generator: torch.Generator,
) -> ParticleEnsemble:
    """Euler/Euler-Maruyama with an old-state Euler log-weight increment.

    Subtracting a common rate and normalizing log weights does not change the
    weighted measure. It avoids losing existing relative weights to a huge
    common offset. This does NOT discard particle-dependent reaction terms.
    """
    x = ensemble.positions
    check_positions(x)
    check_generator(generator, x.device)
    dt = finite_scalar(step_size, "step_size")
    diffusion = finite_scalar(dynamics.diffusion_std, "diffusion_std")
    if dt <= 0 or diffusion < 0:
        raise ValueError("step_size must be positive and diffusion_std nonnegative")
    check_tensor(dynamics.drift, x, "drift")
    check_tensor(dynamics.log_weight_rate, x, "log-weight rate", shape=(x.shape[0],))
    with full_precision(x.device):
        positions = x + dt * dynamics.drift
        if diffusion > 0:
            noise = torch.randn(
                x.shape, dtype=x.dtype, device=x.device, generator=generator
            )
            positions = positions + diffusion * math.sqrt(dt) * noise
        rates = dynamics.log_weight_rate
        centered_rates = rates - rates.max()
        check_tensor(centered_rates, x, "centered log-weight rate", shape=(x.shape[0],))
        log_weights = ensemble.normalized_log_weights + dt * centered_rates
        log_weights = torch.log_softmax(log_weights, dim=0)
    return ParticleEnsemble(positions.detach(), log_weights.detach())


@torch.no_grad()
def ula_corrector(
    positions: Tensor,
    score: ScoreFunction,
    potential: Potential,
    *,
    reverse_time: float,
    config: CorrectorConfig,
    generator: torch.Generator,
) -> Tensor:
    """Algorithm 3 at FIXED next diffusion time, with score - grad(mu_y).

    This is unadjusted Langevin: finite step sizes bias the invariant law.
    For an approximate score its target need not equal the model-induced
    posterior PDE. It consumes no RNG or model calls when num_steps is zero.
    """
    check_positions(positions)
    check_generator(generator, positions.device)
    if finite_scalar(reverse_time, "reverse_time") < 0:
        raise ValueError("reverse_time must be nonnegative")
    if not isinstance(config, CorrectorConfig):
        raise TypeError("config must be a CorrectorConfig")
    current = positions.detach().clone()
    with full_precision(current.device):
        for _ in range(config.num_steps):
            score_values = score(current, reverse_time)
            check_tensor(score_values, current, "score")
            gradient = potential_derivatives(
                potential, current, compute_laplacian=False
            ).gradient
            noise = torch.randn(
                current.shape,
                dtype=current.dtype,
                device=current.device,
                generator=generator,
            )
            current = (
                current
                + config.step_size * (score_values - gradient)
                + math.sqrt(2.0 * config.step_size) * noise
            )
            check_positions(current)
    return current.detach()
