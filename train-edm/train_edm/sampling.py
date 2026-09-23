"""EDM Heun sampling and an explicit increasing-reverse-time AFDPS adapter.

Independently implemented from Karras et al., https://arxiv.org/abs/2206.00364;
reference: https://github.com/NVlabs/edm/blob/main/generate.py. No stochastic
churn is used. Computation follows model float32/float64, not reference float64.

The caller still owns likelihood-conditioned Stage I: unconditional EDM draws
are NOT a replacement. A Gaussian noisy initial law is only an approximation
for an arbitrary learned prior. AFDPS's current dense initializer returns flat
states, not NCHW images; callers must adapt shapes and supply a valid ensemble.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from afdps._validation import (
    check_count,
    check_generator,
    check_positions,
    finite_scalar,
    full_precision,
)
from train_edm.model import denoise


def sigma_grid(
    num_steps: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return N positive, strictly descending power-law levels, without zero."""
    check_count(num_steps, "num_steps")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    low = finite_scalar(sigma_min, "sigma_min")
    high = finite_scalar(sigma_max, "sigma_max")
    power = finite_scalar(rho, "rho")
    if not 0 < low < high or power <= 0:
        raise ValueError("require 0 < sigma_min < sigma_max and rho > 0")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("sigma_grid requires torch.float32 or torch.float64")
    endpoints = torch.tensor([high, low], device=device, dtype=dtype)
    roots = endpoints.pow(1.0 / power)
    fraction = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    levels = torch.lerp(roots[0], roots[1], fraction).pow(power)
    levels[0], levels[-1] = endpoints[0], endpoints[1]
    if (
        not torch.isfinite(levels).all()
        or not (levels > 0).all()
        or not (levels[:-1] > levels[1:]).all()
    ):
        raise ValueError(
            "sigma levels must be finite, positive and strictly descending"
        )
    return levels


def _model_spec(model: torch.nn.Module) -> tuple[Tensor, tuple[int, int, int]]:
    if any(module.training for module in model.modules()):
        raise ValueError("model must already be in eval mode")
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model must have parameters to determine dtype/device")
    if parameter.dtype not in (torch.float32, torch.float64):
        raise TypeError("model parameters must use torch.float32 or torch.float64")
    if any(
        p.dtype != parameter.dtype or p.device != parameter.device
        for p in model.parameters()
    ):
        raise ValueError("model parameters must share dtype and device")
    config = model.config
    return parameter, (config.channels, config.image_size, config.image_size)


@torch.no_grad()
def sample_edm(
    model: torch.nn.Module,
    num_images: int,
    *,
    generator: torch.Generator,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> Tensor:
    """Raw detached NCHW Heun samples, with final Euler step to sigma=0.

    Require an eval model and explicit same-device generator. Only the initial
    Gaussian draw consumes RNG (zero churn); never denoise at zero or clip the
    result. The Gaussian start at sigma_max approximates the learned noisy prior.
    """
    check_count(num_images, "num_images")
    parameter, shape = _model_spec(model)
    check_generator(generator, parameter.device)
    with full_precision(parameter.device):
        levels = sigma_grid(
            num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        levels = torch.cat((levels, levels.new_zeros(1)))
        current = levels[0] * torch.randn(
            (num_images, *shape),
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        for index, (sigma, following) in enumerate(zip(levels[:-1], levels[1:])):
            derivative = (current - denoise(model, current, sigma)) / sigma
            step = following - sigma
            proposal = current + step * derivative
            if index < num_steps - 1:
                next_derivative = (
                    proposal - denoise(model, proposal, following)
                ) / following
                proposal = current + step * (0.5 * derivative + 0.5 * next_derivative)
            current = proposal
    check_positions(current)
    return current.detach()


@dataclass(frozen=True)
class EDMSchedule:
    """AFDPS forward coefficients on reverse clock t=0 (noisy) to t=T.

    sigma(t) = sigma_max + (sigma_min - sigma_max)*t/T, F=0,
    G(T-t)^2 = 2*sigma(t)*(sigma_max-sigma_min)/T. G is NOT sigma.
    The terminal prior is the positive-floor p_sigma_min, NOT noiseless data.
    """

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    terminal_time: float = 1.0

    def __post_init__(self) -> None:
        low = finite_scalar(self.sigma_min, "sigma_min")
        high = finite_scalar(self.sigma_max, "sigma_max")
        horizon = finite_scalar(self.terminal_time, "terminal_time")
        if not 0 < low < high or horizon <= 0:
            raise ValueError("require 0 < sigma_min < sigma_max and terminal_time > 0")
        for sigma in (low, high):
            variance = finite_scalar(
                2 * sigma * (high - low) / horizon, "squared forward_diffusion"
            )
            if variance <= 0:
                raise ValueError(
                    "squared forward_diffusion must be representably positive"
                )

    def sigma(self, reverse_time: float) -> float:
        time = finite_scalar(reverse_time, "reverse_time")
        if not 0 <= time <= self.terminal_time:
            raise ValueError("reverse_time must be in [0, terminal_time]")
        fraction = time / self.terminal_time
        return (1 - fraction) * self.sigma_max + fraction * self.sigma_min

    def forward_drift(self, reverse_time: float) -> float:
        self.sigma(reverse_time)
        return 0.0

    def forward_diffusion(self, reverse_time: float) -> float:
        return math.sqrt(
            2
            * self.sigma(reverse_time)
            * (self.sigma_max - self.sigma_min)
            / self.terminal_time
        )

    def time_grid(self, num_steps: int, rho: float = 7.0) -> tuple[float, ...]:
        """Return N+1 increasing reverse times with exact endpoints 0 and T."""
        check_count(num_steps, "num_steps")
        levels = sigma_grid(
            num_steps + 1,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=rho,
            dtype=torch.float64,
        )
        times = (
            (self.sigma_max - levels)
            / (self.sigma_max - self.sigma_min)
            * self.terminal_time
        ).tolist()
        times[0], times[-1] = 0.0, float(self.terminal_time)
        if any(end <= start for start, end in zip(times[:-1], times[1:])):
            raise ValueError("time_grid must be representably strictly increasing")
        return tuple(times)


@torch.no_grad()
def edm_score(
    model: torch.nn.Module,
    positions: Tensor,
    reverse_time: float,
    *,
    schedule: EDMSchedule,
) -> Tensor:
    """Detached (D(x,sigma(t))-x)/sigma(t)^2; bind model/schedule with partial.

    This is a prior score, not a likelihood-conditioned Stage-I initializer.
    The model must already be in eval mode; inputs and its mode are preserved.
    """
    parameter, shape = _model_spec(model)
    check_positions(positions)
    if positions.shape[1:] != shape:
        raise ValueError(f"positions must have NCHW state shape {shape}")
    if positions.dtype != parameter.dtype or positions.device != parameter.device:
        raise ValueError("positions and model must share dtype and device")
    sigma = schedule.sigma(reverse_time)
    with full_precision(positions.device):
        score = (denoise(model, positions, sigma) - positions) / sigma / sigma
    check_positions(score)
    return score.detach()
