"""A simple, explicit forward process for verification and small experiments."""

from __future__ import annotations

from dataclasses import dataclass

from afdps._validation import finite_scalar


@dataclass(frozen=True)
class BrownianSchedule:
    """Forward dx_s = diffusion_std * dW_s, evaluated at s = T - reverse_time.

    F=0 and G=diffusion_std are constant; the perturbation variance accumulated
    by forward time s is G^2*s. This is a variance-exploding process with linear
    variance, NOT a pretrained EDM/VP checkpoint schedule. The initial Gaussian
    reference also includes the clean prior's variance; do not use G as rho.
    Zero diffusion is allowed for stationary-prior verification problems.
    """

    terminal_time: float = 1.0
    diffusion_std: float = 1.0

    def __post_init__(self) -> None:
        horizon = finite_scalar(self.terminal_time, "terminal_time")
        diffusion = finite_scalar(self.diffusion_std, "diffusion_std")
        if horizon <= 0:
            raise ValueError("terminal_time must be positive")
        if diffusion < 0:
            raise ValueError("diffusion_std must be nonnegative")
        finite_scalar(diffusion * diffusion * horizon, "terminal noise variance")

    def _check_time(self, reverse_time: float) -> float:
        time = finite_scalar(reverse_time, "reverse_time")
        if not 0 <= time <= self.terminal_time:
            raise ValueError("reverse_time must be in [0, terminal_time]")
        return time

    def forward_drift(self, reverse_time: float) -> float:
        """F(T-t), the forward linear drift coefficient, not a reversed drift."""
        self._check_time(reverse_time)
        return 0.0

    def forward_diffusion(self, reverse_time: float) -> float:
        """G(T-t), the Brownian amplitude, not the marginal noise level."""
        self._check_time(reverse_time)
        return float(self.diffusion_std)

    def noise_variance(self, reverse_time: float) -> float:
        """Added forward perturbation variance, decreasing to zero at t=T."""
        time = self._check_time(reverse_time)
        return self.diffusion_std**2 * (self.terminal_time - time)
