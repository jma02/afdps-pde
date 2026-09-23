import math
from dataclasses import FrozenInstanceError
from functools import partial

import pytest
import torch
from torch.testing import assert_close
from train_edm.model import ModelConfig, UNet
from train_edm.sampling import EDMSchedule, edm_score, sample_edm, sigma_grid

from afdps import CorrectorConfig, ParticleEnsemble, SamplerConfig, sample


class RawModel(torch.nn.Module):
    """Known raw network, not a mocked denoiser; exercises EDM preconditioning."""

    def __init__(self, gain=0.25, noise_scale=0.1):
        super().__init__()
        self.config = ModelConfig(image_size=4, channels=1, width=8, multipliers=(1,))
        self.gain = torch.nn.Parameter(torch.tensor(float(gain)))
        self.noise_scale = noise_scale
        self.calls = []

    def forward(self, x, c_noise):
        self.calls.append(
            (x.detach().clone(), c_noise.detach().clone(), torch.is_grad_enabled())
        )
        return self.gain * x + self.noise_scale * c_noise.reshape(-1, 1, 1, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [2, 6])
def test_seeded_heun_matches_independent_affine_recurrence_and_evaluation_order(
    dtype, steps
):
    model = RawModel(gain=3).to(dtype=dtype).eval()
    model.gain.grad = torch.tensor(17.0, dtype=dtype)
    generator = torch.Generator().manual_seed(23)
    reference_rng = torch.Generator().manual_seed(23)
    global_state = torch.random.get_rng_state().clone()
    low, high, rho = 0.05, 3.0, 3.0
    levels = [
        ((1 - i / (steps - 1)) * high ** (1 / rho) + i / (steps - 1) * low ** (1 / rho))
        ** rho
        for i in range(steps)
    ]
    levels[0], levels[-1] = high, low
    expected = high * torch.randn((2, 1, 4, 4), dtype=dtype, generator=reference_rng)
    # dx/dsigma = a(sigma)*x + b(sigma), derived from the fake RAW network.
    coefficients = []
    for sigma in levels:
        variance = sigma**2 + 0.5**2
        coefficients.append(
            (
                (1 - (0.25 + 3 * sigma * 0.5) / variance) / sigma,
                -0.5 * 0.1 * math.log(sigma) / (4 * math.sqrt(variance)),
            )
        )
    for i, ((a, b), (c, d)) in enumerate(zip(coefficients[:-1], coefficients[1:])):
        h = levels[i + 1] - levels[i]
        expected = (1 + h * (a + c) / 2 + h * h * a * c / 2) * expected + h * (
            b + d + h * c * b
        ) / 2
    a, b = coefficients[-1]
    expected = (1 - low * a) * expected - low * b  # final Euler to zero
    result = sample_edm(
        model,
        2,
        generator=generator,
        num_steps=steps,
        sigma_min=low,
        sigma_max=high,
        rho=rho,
    )
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    assert_close(result, expected, rtol=tolerance, atol=tolerance)
    assert result.shape == (2, 1, 4, 4) and result.dtype == dtype
    assert result.abs().max() > 1  # raw states must NOT be clipped/normalized
    assert not result.requires_grad and not model.training
    assert model.gain.grad.item() == 17
    assert all(not grad for _, _, grad in model.calls)
    calls = [noise.flatten()[0].mul(4).exp().item() for _, noise, _ in model.calls]
    assert calls == pytest.approx(
        [high] + [s for s in levels[1:] for _ in range(2)], rel=tolerance
    )
    assert len(calls) == 2 * steps - 1 and min(calls) > 0
    assert_close(generator.get_state(), reference_rng.get_state(), rtol=0, atol=0)
    assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)
    again = sample_edm(
        model,
        2,
        generator=torch.Generator().manual_seed(23),
        num_steps=steps,
        sigma_min=low,
        sigma_max=high,
        rho=rho,
    )
    assert_close(again, result, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_power_grid_endpoints_spacing_and_precision(dtype):
    grid = sigma_grid(18, dtype=dtype)
    assert grid.shape == (18,) and grid.dtype == dtype and grid.device.type == "cpu"
    assert grid[0] == torch.tensor(80.0, dtype=dtype)
    assert grid[-1] == torch.tensor(0.002, dtype=dtype)
    assert (grid > 0).all() and (grid[:-1] > grid[1:]).all()
    assert_close(
        grid.pow(1 / 7),
        torch.linspace(80 ** (1 / 7), 0.002 ** (1 / 7), 18, dtype=dtype),
    )


@pytest.mark.parametrize(
    "controls",
    [
        {"num_steps": 0},
        {"num_steps": 1},
        {"num_steps": True},
        {"num_steps": 2.5},
        {"sigma_min": 0},
        {"sigma_min": -1},
        {"sigma_min": float("nan")},
        {"sigma_min": 80},
        {"sigma_max": float("inf")},
        {"rho": 0},
        {"rho": True},
        {"sigma_min": 1e-50},
        {"sigma_max": 1e40},
        {"sigma_min": 1.0, "sigma_max": 1.0 + 1e-8},
        {"rho": 1e40},
    ],
)
def test_grid_rejects_invalid_or_unrepresentable_controls(controls):
    with pytest.raises(ValueError):
        sigma_grid(**{"num_steps": 4, **controls})


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
def test_grid_rejects_nonbaseline_dtype(dtype):
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        sigma_grid(4, dtype=dtype)


@pytest.mark.parametrize("steps", [1, 5, 18])
def test_schedule_coefficients_clocks_and_mapped_grid(steps):
    schedule = EDMSchedule(sigma_min=0.05, sigma_max=3.0, terminal_time=2.5)
    assert schedule.sigma(0) == 3.0
    assert schedule.sigma(2.5) == 0.05  # positive-floor law, not noiseless data
    assert schedule.sigma(1.25) == pytest.approx(1.525)
    times = schedule.time_grid(steps)
    assert len(times) == steps + 1 and times[0] == 0 and times[-1] == 2.5
    assert all(a < b for a, b in zip(times[:-1], times[1:]))
    levels = sigma_grid(steps + 1, sigma_min=0.05, sigma_max=3.0, dtype=torch.float64)
    assert [schedule.sigma(t) for t in times] == pytest.approx(levels.tolist())
    for start, end in zip(times[:-1], times[1:]):
        assert schedule.forward_drift(start) == 0
        g0, g1 = schedule.forward_diffusion(start), schedule.forward_diffusion(end)
        assert g0 > 0 and g1 > 0
        assert (g0**2 + g1**2) * (end - start) / 2 == pytest.approx(
            schedule.sigma(start) ** 2 - schedule.sigma(end) ** 2
        )
    with pytest.raises(FrozenInstanceError):
        schedule.sigma_min = 0.1


@pytest.mark.parametrize(
    "controls",
    [
        {"sigma_min": 0},
        {"sigma_min": 80},
        {"sigma_max": -1},
        {"sigma_max": float("inf")},
        {"terminal_time": 0},
        {"terminal_time": True},
        {"terminal_time": float("nan")},
        {"sigma_max": 1e200},
    ],
)
def test_schedule_rejects_invalid_controls(controls):
    with pytest.raises(ValueError):
        EDMSchedule(**controls)


@pytest.mark.parametrize("time", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_schedule_rejects_invalid_clock_for_every_coefficient(time):
    schedule = EDMSchedule()
    for method in (schedule.sigma, schedule.forward_drift, schedule.forward_diffusion):
        with pytest.raises(ValueError, match="reverse_time"):
            method(time)


def test_schedule_rejects_invalid_time_grid_controls():
    for steps in (0, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            EDMSchedule().time_grid(steps)
    with pytest.raises(ValueError, match="rho"):
        EDMSchedule().time_grid(2, rho=-1)
    with pytest.raises(ValueError, match="representably strictly increasing"):
        EDMSchedule(sigma_min=1e-30, sigma_max=1).time_grid(20, rho=20)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("time", [0, 0.4, 2.0])
def test_score_equation_sign_reverse_clock_and_no_autograd(dtype, time):
    model = RawModel(gain=0, noise_scale=0).to(dtype=dtype).eval()
    schedule = EDMSchedule(sigma_min=0.5, sigma_max=2, terminal_time=2)
    positions = (
        torch.linspace(-2, 2, 32, dtype=dtype).reshape(2, 1, 4, 4).requires_grad_()
    )
    original = positions.detach().clone()
    score = partial(edm_score, model, schedule=schedule)(positions, time)
    sigma = 2 + (0.5 - 2) * time / 2
    assert_close(score, -positions / (sigma**2 + 0.5**2))
    assert (score * positions <= 0).all()
    assert model.calls[0][1].flatten()[0].item() == pytest.approx(math.log(sigma) / 4)
    assert (
        not score.requires_grad and positions.grad is None and model.gain.grad is None
    )
    assert not model.calls[0][2] and not model.training
    assert_close(positions, original, rtol=0, atol=0)


def test_score_avoids_overflow_when_squaring_large_float32_sigma():
    model = RawModel(gain=0, noise_scale=0).eval()
    schedule = EDMSchedule(sigma_min=1e19, sigma_max=1e20)
    positions = torch.full((1, 1, 4, 4), 1e20)
    score = edm_score(model, positions, 0, schedule=schedule)
    assert_close(score, torch.full_like(positions, -1e-20), rtol=1e-5, atol=0)


def test_eval_mode_required_without_mutating_model_or_rng():
    model = RawModel()
    generator = torch.Generator().manual_seed(8)
    state = generator.get_state().clone()
    with pytest.raises(ValueError, match="eval"):
        sample_edm(model, 1, generator=generator)
    with pytest.raises(ValueError, match="eval"):
        edm_score(model, torch.zeros(1, 1, 4, 4), 0, schedule=EDMSchedule())
    assert model.training and not model.calls
    assert_close(generator.get_state(), state, rtol=0, atol=0)
    model.eval()
    model.child = torch.nn.Dropout()  # a training child must not escape the guard
    with pytest.raises(ValueError, match="eval"):
        sample_edm(model, 1, generator=generator)
    assert not model.training and model.child.training


@pytest.mark.parametrize(
    "controls",
    [
        {"num_images": 0},
        {"num_images": True},
        {"num_images": 1.5},
        {"num_steps": 1},
        {"sigma_min": 0},
        {"sigma_max": 0.001},
        {"rho": -1},
    ],
)
def test_sampler_validates_before_random_draw(controls):
    generator = torch.Generator().manual_seed(9)
    state = generator.get_state().clone()
    options = {"num_images": 2, **controls}
    with pytest.raises(ValueError):
        sample_edm(RawModel().eval(), generator=generator, **options)
    assert_close(generator.get_state(), state, rtol=0, atol=0)


def test_explicit_generator_and_score_image_dtype_checks():
    model = RawModel().eval()
    with pytest.raises(TypeError, match="generator"):
        sample_edm(model, 1, generator=None)
    score = partial(edm_score, model, schedule=EDMSchedule())
    for positions in (
        torch.zeros(2, 16),
        torch.zeros(2, 3, 4, 4),
        torch.zeros(0, 1, 4, 4),
    ):
        with pytest.raises(ValueError, match="shape"):
            score(positions, 0)
    with pytest.raises(ValueError, match="dtype and device"):
        score(torch.zeros(2, 1, 4, 4, dtype=torch.float64), 0)
    with pytest.raises(ValueError, match="finite"):
        score(torch.full((2, 1, 4, 4), float("nan")), 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_unet_smoke_preserves_dtype_under_autocast(dtype):
    model = (
        UNet(ModelConfig(image_size=4, channels=1, width=8, multipliers=(1,)))
        .to(dtype=dtype)
        .eval()
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        images = sample_edm(
            model, 1, generator=torch.Generator().manual_seed(3), num_steps=2
        )
    assert images.dtype == dtype and images.shape == (1, 1, 4, 4)
    assert torch.isfinite(images).all() and not images.requires_grad
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_generation_score_and_same_device_generator():
    model = RawModel().to("mps").eval()
    with pytest.raises(ValueError, match="device"):
        sample_edm(model, 1, generator=torch.Generator())
    generator = torch.Generator(device="mps").manual_seed(5)
    images = sample_edm(model, 2, generator=generator, num_steps=3)
    assert images.device.type == "mps" and images.dtype == torch.float32
    score = edm_score(model, images, 0.5, schedule=EDMSchedule())
    assert score.device == images.device and score.dtype == images.dtype
    assert torch.isfinite(score).all() and not score.requires_grad
    with pytest.raises(ValueError, match="dtype and device"):
        edm_score(model, images.cpu(), 0.5, schedule=EDMSchedule())


@pytest.mark.parametrize("variant", ["ode", "sde"])
def test_partial_callback_integrates_with_afdps_from_explicit_image_ensemble(variant):
    # Deliberately supplied states/weights: an API integration test, NOT an
    # assertion that unconditional images form the exact initial posterior.
    model = RawModel(gain=0, noise_scale=0).double().eval()
    schedule = EDMSchedule(sigma_min=0.5, sigma_max=0.6, terminal_time=0.05)
    positions = torch.linspace(-0.7, 0.8, 32, dtype=torch.float64).reshape(2, 1, 4, 4)
    original = positions.clone()
    initial = ParticleEnsemble(positions, positions.new_tensor([-0.2, -0.8]))
    config = SamplerConfig(
        variant=variant,
        num_particles=2,
        num_steps=1,
        resample_ess_fraction=None,
        corrector=CorrectorConfig(num_steps=0) if variant == "ode" else None,
    )
    result = sample(
        partial(edm_score, model, schedule=schedule),
        lambda x: 0.1 * x.square().flatten(1).sum(dim=1),
        schedule,
        initial=initial,
        config=config,
        generator=torch.Generator().manual_seed(21),
        time_grid=schedule.time_grid(1),
    )
    g2 = 2 * 0.6 * (0.6 - 0.5) / 0.05
    score = -positions / (0.6**2 + 0.5**2)
    gradient = 0.2 * positions
    prior_drift = (0.5 if variant == "ode" else 1) * g2 * score
    drift = prior_drift - (g2 * gradient if variant == "sde" else 0)
    expected = positions + 0.05 * drift
    rate = -(prior_drift * gradient).flatten(1).sum(dim=1)
    if variant == "sde":
        expected += math.sqrt(0.05 * g2) * torch.randn(
            positions.shape,
            dtype=positions.dtype,
            generator=torch.Generator().manual_seed(21),
        )
        rate += 0.5 * g2 * (gradient.square().flatten(1).sum(dim=1) - 0.2 * 16)
    assert_close(result.ensemble.positions, expected)
    assert_close(
        result.ensemble.log_weights,
        torch.log_softmax(initial.log_weights + 0.05 * rate, 0),
    )
    assert len(model.calls) == 1 and result.diagnostics[-1].reverse_time == 0.05
    assert not result.ensemble.positions.requires_grad and model.gain.grad is None
    assert_close(initial.positions, original, rtol=0, atol=0)
