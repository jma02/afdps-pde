import math

import pytest
import torch

from afdps.schedules import BrownianSchedule


@pytest.mark.parametrize("reverse_time", [0.0, 0.5, 1.25, 2.5])
def test_brownian_coefficients_and_reverse_noise_variance(reverse_time):
    schedule = BrownianSchedule(terminal_time=2.5, diffusion_std=1.7)
    assert schedule.forward_drift(reverse_time) == 0.0
    assert schedule.forward_diffusion(reverse_time) == 1.7
    assert schedule.noise_variance(reverse_time) == pytest.approx(
        1.7**2 * (2.5 - reverse_time)
    )


def test_noise_variance_decreases_but_brownian_amplitude_does_not():
    schedule = BrownianSchedule(terminal_time=3, diffusion_std=2)
    times = [0.0, 1.0, 2.0, 3.0]
    assert [schedule.noise_variance(t) for t in times] == [12.0, 8.0, 4.0, 0.0]
    assert [schedule.forward_diffusion(t) for t in times] == [2.0] * 4
    # G is neither the marginal standard deviation nor the initial prior scale.
    assert schedule.noise_variance(0) != schedule.forward_diffusion(0) ** 2


def test_zero_diffusion_is_a_valid_stationary_verification_schedule():
    schedule = BrownianSchedule(terminal_time=2.0, diffusion_std=0.0)
    for time in (0.0, 0.7, 2.0):
        assert schedule.forward_drift(time) == 0.0
        assert schedule.forward_diffusion(time) == 0.0
        assert schedule.noise_variance(time) == 0.0


@pytest.mark.parametrize(
    "name,value",
    [
        ("terminal_time", 0.0),
        ("terminal_time", -1.0),
        ("terminal_time", float("nan")),
        ("terminal_time", float("inf")),
        ("terminal_time", True),
        ("terminal_time", torch.tensor(1.0)),
        ("diffusion_std", -0.1),
        ("diffusion_std", float("nan")),
        ("diffusion_std", float("inf")),
        ("diffusion_std", False),
        ("diffusion_std", torch.tensor(1.0)),
    ],
)
def test_schedule_rejects_invalid_scalar_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        BrownianSchedule(**{name: value})


@pytest.mark.parametrize("terminal_time,diffusion_std", [(1.0, 1e200), (1e308, 2.0)])
def test_schedule_rejects_overflowing_terminal_variance(terminal_time, diffusion_std):
    with pytest.raises(ValueError, match="terminal noise variance"):
        BrownianSchedule(terminal_time=terminal_time, diffusion_std=diffusion_std)


@pytest.mark.parametrize(
    "method", ["forward_drift", "forward_diffusion", "noise_variance"]
)
@pytest.mark.parametrize(
    "time",
    [
        math.nextafter(0.0, -math.inf),
        math.nextafter(2.0, math.inf),
        -1.0,
        3.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        torch.tensor(1.0),
        None,
    ],
)
def test_all_schedule_queries_validate_reverse_time(method, time):
    schedule = BrownianSchedule(terminal_time=2.0)
    with pytest.raises(ValueError, match="reverse_time"):
        getattr(schedule, method)(time)
