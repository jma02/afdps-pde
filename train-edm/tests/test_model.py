import math
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from train_edm.model import ModelConfig, UNet, denoise, edm_loss


@pytest.fixture(autouse=True)
def local_torch_state():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(91)
        yield


@pytest.fixture
def model():
    return UNet(ModelConfig(image_size=8, channels=1, width=4, multipliers=(1, 2)))


class Raw(nn.Module):
    """An analytic raw network, independent of U-Net and preconditioning."""

    def __init__(self):
        super().__init__()
        self.config = ModelConfig(image_size=4, channels=1, sigma_data=0.7)
        self.slope = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

    def forward(self, x, noise_labels):
        self.last_x, self.last_labels = x, noise_labels
        return self.slope * x + noise_labels[:, None, None, None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unet_shape_noise_conditioning_and_all_parameters_trainable(model, dtype):
    model = model.to(dtype=dtype)
    x = torch.randn(2, 1, 8, 8, dtype=dtype, requires_grad=True)
    labels = torch.tensor([-0.6, 0.2], dtype=dtype, requires_grad=True)
    output = model(x, labels)
    assert output.shape == x.shape and output.dtype == dtype
    assert model.config == ModelConfig(8, 1, 4, (1, 2))
    assert not torch.equal(output, model(x, labels + 0.1))
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(labels.grad).all() and labels.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("multipliers", [(1,), (2, 3), (1, 2, 2)])
def test_resolution_skips_and_singleton_bottleneck(multipliers):
    size = 2 ** (len(multipliers) - 1)
    model = UNet(ModelConfig(size, channels=2, width=3, multipliers=multipliers))
    x = torch.randn(1, 2, size, size)
    assert model(x, torch.zeros(1)).shape == x.shape


@pytest.mark.parametrize("training", [True, False])
def test_particles_are_independent_in_forward_and_autograd(model, training):
    model = model.double().train(training)
    x = torch.randn(3, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    sigma = x.new_tensor([0.1, 0.5, 1.2])
    together = denoise(model, x, sigma)
    separate = torch.cat(
        [denoise(model, x[i : i + 1], sigma[i]) for i in range(x.shape[0])]
    )
    assert_close(together, separate, rtol=1e-10, atol=1e-10)
    gradient = torch.autograd.grad(together[0].square().sum(), x, create_graph=True)[0]
    assert_close(gradient[1:], torch.zeros_like(gradient[1:]), rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(gradient[0].sum(), x)[0]).all()
    assert model.training is training


@pytest.mark.parametrize("shape", ["scalar", "vector", "broadcast"])
def test_preconditioning_matches_independent_equations(shape):
    model = Raw()
    x = torch.linspace(-2, 1, 48, dtype=torch.float64).reshape(3, 1, 4, 4)
    values = [0.4] * 3 if shape == "scalar" else [0.03, 0.7, 4.0]
    sigma = 0.4 if shape == "scalar" else x.new_tensor(values)
    if shape == "broadcast":
        sigma = sigma[:, None, None, None]
    expected = []
    for image, level in zip(x, values):
        denominator = level**2 + model.config.sigma_data**2
        raw = 0.3 * image / denominator**0.5 + math.log(level) / 4
        expected.append(
            model.config.sigma_data**2 / denominator * image
            + level * model.config.sigma_data / denominator**0.5 * raw
        )
    actual = denoise(model, x, sigma)
    assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)
    levels = x.new_tensor(values)
    assert_close(model.last_x, x / (levels[:, None, None, None] ** 2 + 0.7**2).sqrt())
    assert_close(model.last_labels, levels.log() / 4)


@pytest.mark.parametrize("settings", [{}, {"p_mean": 0.2, "p_std": 0.0}])
def test_seeded_lognormal_weighted_loss_matches_formula_and_isolates_rng(settings):
    model = Raw()
    clean = torch.linspace(-0.8, 0.9, 48, dtype=torch.float64).reshape(3, 1, 4, 4)
    original = clean.clone()
    generator = torch.Generator().manual_seed(18)
    reference = torch.Generator().manual_seed(18)
    state = torch.random.get_rng_state().clone()
    z = torch.randn((3, 1, 1, 1), dtype=clean.dtype, generator=reference)
    sigma = (z * settings.get("p_std", 1.2) + settings.get("p_mean", -1.2)).exp()
    noise = torch.randn(clean.shape, dtype=clean.dtype, generator=reference)
    noisy = clean + sigma * noise
    denominator = sigma.square() + 0.7**2
    raw = 0.3 * noisy / denominator.sqrt() + sigma.log() / 4
    prediction = 0.7**2 / denominator * noisy + sigma * 0.7 / denominator.sqrt() * raw
    expected = (
        denominator / (sigma * 0.7).square() * (prediction - clean).square()
    ).mean()
    actual = edm_loss(model, clean, generator=generator, **settings)
    assert actual.shape == ()
    assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert_close(generator.get_state(), reference.get_state())
    assert_close(torch.random.get_rng_state(), state)
    assert_close(clean, original, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(model.slope.grad) and model.slope.grad.abs() > 0


def test_float32_coefficients_avoid_overflow_for_large_finite_sigma():
    x = torch.ones(2, 1, 4, 4)
    sigma = torch.tensor([1e-30, 1e30])
    levels = sigma.double()[:, None, None, None]
    denominator = levels.square() + 0.7**2
    raw = 0.3 / denominator.sqrt() + levels.log() / 4
    expected = 0.7**2 / denominator + levels * 0.7 / denominator.sqrt() * raw
    assert_close(denoise(Raw().float(), x, sigma), expected.float().expand_as(x))


def test_default_unet_shape():
    model = UNet(ModelConfig())
    x = torch.zeros(1, 3, 32, 32)
    assert model(x, torch.zeros(1)).shape == x.shape


def test_cpu_training_loss_descends_on_fixed_noise(model):
    clean = torch.zeros(4, 1, 8, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = edm_loss(
            model,
            clean,
            generator=torch.Generator().manual_seed(31),
            p_mean=-0.5,
            p_std=0.3,
        )
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < 0.35 * losses[0]


def test_ambient_autocast_cannot_lower_precision(model):
    x = torch.randn(2, 1, 8, 8)
    expected = denoise(model, x, 0.4)
    generator = torch.Generator().manual_seed(3)
    reference = edm_loss(model, x, generator=generator)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert_close(denoise(model, x, 0.4), expected, rtol=0, atol=0)
        actual = edm_loss(model, x, generator=torch.Generator().manual_seed(3))
    assert actual.dtype == torch.float32
    assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 0},
        {"image_size": True},
        {"channels": 0},
        {"channels": 1.5},
        {"width": 0},
        {"width": 1},
        {"width": 2.5},
        {"multipliers": ()},
        {"multipliers": [1, 2]},
        {"multipliers": (1, 0)},
        {"multipliers": (True,)},
        {"multipliers": (1, 1.5)},
        {"image_size": 6},
        {"sigma_data": 0},
        {"sigma_data": -1},
        {"sigma_data": float("inf")},
        {"sigma_data": float("nan")},
        {"sigma_data": True},
    ],
)
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_frozen_config_and_constructor_type():
    with pytest.raises(FrozenInstanceError):
        ModelConfig().width = 8
    with pytest.raises(TypeError, match="ModelConfig"):
        UNet({})


@pytest.mark.parametrize(
    "x,error",
    [
        (torch.zeros(2, 1, 8), "NCHW"),
        (torch.zeros(2, 2, 8, 8), "NCHW"),
        (torch.zeros(2, 1, 8, 4), "NCHW"),
        (torch.zeros(0, 1, 8, 8), "nonempty"),
        (torch.zeros(2, 1, 8, 8, dtype=torch.int64), "float32 or torch.float64"),
        (torch.zeros(2, 1, 8, 8, dtype=torch.float16), "float32 or torch.float64"),
        (torch.full((2, 1, 8, 8), float("nan")), "finite"),
        (torch.zeros(2, 1, 8, 8, dtype=torch.float64), "dtype and device"),
    ],
)
def test_invalid_images_rejected_by_each_entry_point(model, x, error):
    with pytest.raises((ValueError, TypeError), match=error):
        model(x, torch.zeros(2))
    with pytest.raises((ValueError, TypeError), match=error):
        denoise(model, x, 0.4)
    with pytest.raises((ValueError, TypeError), match=error):
        edm_loss(model, x, generator=torch.Generator())


@pytest.mark.parametrize(
    "labels,error",
    [
        (torch.tensor(0.5), "shape"),
        (torch.zeros(2, 1), "shape"),
        (torch.zeros(2, dtype=torch.float64), "dtype and device"),
        (torch.zeros(2, device="meta"), "dtype and device"),
        (torch.full((2,), float("inf")), "finite"),
        ([0.1, 0.2], "tensor"),
    ],
)
def test_invalid_noise_labels(model, labels, error):
    with pytest.raises((ValueError, TypeError), match=error):
        model(torch.zeros(2, 1, 8, 8), labels)


@pytest.mark.parametrize(
    "sigma,error",
    [
        (0, "positive"),
        (-1, "positive"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (True, "scalar"),
        ([0.5], "scalar"),
        (torch.zeros(2), "positive"),
        (torch.ones(3), "sigma must be"),
        (torch.ones(2, 1), "sigma must be"),
        (torch.ones(2, 1, 8, 8), "sigma must be"),
        (torch.ones(2, dtype=torch.float64), "dtype and device"),
        (torch.ones(2, dtype=torch.int64), "dtype and device"),
        (torch.ones(2, device="meta"), "dtype and device"),
    ],
)
def test_invalid_sigma(model, sigma, error):
    with pytest.raises((ValueError, TypeError), match=error):
        denoise(model, torch.zeros(2, 1, 8, 8), sigma)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"generator": None}, "explicit torch.Generator"),
        ({"p_mean": float("nan")}, "p_mean"),
        ({"p_mean": True}, "p_mean"),
        ({"p_std": -1}, "p_std"),
        ({"p_std": float("inf")}, "p_std"),
    ],
)
def test_invalid_loss_arguments_preserve_generator(model, kwargs, error):
    generator = torch.Generator().manual_seed(3)
    state = generator.get_state().clone()
    options = {"generator": generator, **kwargs}
    with pytest.raises((ValueError, TypeError), match=error):
        edm_loss(model, torch.zeros(2, 1, 8, 8), **options)
    assert_close(generator.get_state(), state)


def test_model_device_mismatch_is_explicit(model):
    with pytest.raises(ValueError, match="dtype and device"):
        denoise(model.to("meta"), torch.zeros(2, 1, 8, 8), 0.4)


@pytest.mark.parametrize(
    "bad_output",
    [
        torch.zeros(3, 1, 4, 3, dtype=torch.float64),
        torch.zeros(3, 1, 4, 4),
        torch.full((3, 1, 4, 4), float("nan"), dtype=torch.float64),
    ],
)
def test_raw_model_output_contract(bad_output, monkeypatch):
    model = Raw()
    monkeypatch.setattr(model, "forward", lambda x, labels: bad_output)
    with pytest.raises(ValueError, match="raw model output"):
        denoise(model, torch.zeros(3, 1, 4, 4, dtype=torch.float64), 0.5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_inference_training_and_generator_device(model):
    model = model.to("mps")
    clean = torch.zeros(2, 1, 8, 8, device="mps")
    assert denoise(model, clean, 0.4).device.type == "mps"
    with pytest.raises(ValueError, match="share device"):
        edm_loss(model, clean, generator=torch.Generator())
    loss = edm_loss(
        model, clean, generator=torch.Generator(device="mps").manual_seed(3)
    )
    loss.backward(
        loss.new_tensor(1.0)
    )  # torch 2.2 MPS implicit scalar seed workaround.
    assert torch.isfinite(loss)
