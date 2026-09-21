import math
from unittest.mock import Mock

import pytest
import torch
from torch.testing import assert_close

from afdps.config import CorrectorConfig
from afdps.derivatives import PotentialDerivatives
from afdps.dynamics import ParticleDynamics, afdps_dynamics, euler_step, ula_corrector
from afdps.particles import ParticleEnsemble


def quadratic_potential(positions):
    return 0.5 * positions.square().flatten(1).sum(1)


def negative_score(positions, reverse_time):
    return -positions


@pytest.fixture
def ensemble():
    positions = torch.arange(12, dtype=torch.float64).reshape(3, 1, 2, 2) / 8
    return ParticleEnsemble(positions, positions.new_tensor([-1.0, -torch.inf, 0.0]))


@pytest.mark.parametrize("variant", ["sde", "ode"])
@pytest.mark.parametrize("state_shape", [(4,), (1, 2, 2)])
def test_paper_coefficients_with_nonzero_forward_drift_and_event_axis_sums(
    variant, state_shape
):
    x = torch.linspace(-1.2, 1.5, 12, dtype=torch.float64).reshape(3, *state_shape)
    score = 0.3 - 0.7 * x
    # mu(x) = sum(x**4)/4, so curvature differs between particles.
    gradient = x**3
    laplacian = 3 * x.square().reshape(3, -1).sum(1)
    f, g = 0.35, 1.4
    result = afdps_dynamics(
        x,
        score,
        PotentialDerivatives(gradient, laplacian if variant == "sde" else None),
        forward_drift=f,
        forward_diffusion=g,
        variant=variant,
    )
    # Expanded equations (3.4)/(3.6), independently reduced per particle.
    score_factor = g**2 if variant == "sde" else g**2 / 2
    expected_rate = []
    for position, phi, grad, lap in zip(x, score, gradient, laplacian):
        rate = f * (position * grad).sum() - score_factor * (phi * grad).sum()
        if variant == "sde":
            rate += g**2 / 2 * (grad.square().sum() - lap)
        expected_rate.append(rate)
    expected_drift = -f * x + score_factor * score
    if variant == "sde":
        expected_drift -= g**2 * gradient
    assert_close(result.drift, expected_drift)
    assert_close(result.log_weight_rate, torch.stack(expected_rate))
    assert result.log_weight_rate.shape == (3,)
    assert result.diffusion_std == (g if variant == "sde" else 0.0)
    if variant == "sde":
        wrong_rate = -(result.drift * gradient).flatten(1).sum(1) + g**2 / 2 * (
            gradient.square().flatten(1).sum(1) - laplacian
        )
        assert not torch.allclose(result.log_weight_rate, wrong_rate)


def test_ode_has_no_likelihood_drift_or_laplacian_requirement(ensemble):
    x = ensemble.positions
    results = [
        afdps_dynamics(
            x,
            -x,
            PotentialDerivatives(scale * torch.ones_like(x), None),
            forward_drift=-0.3,
            forward_diffusion=1.6,
            variant="ode",
        )
        for scale in (1.0, 2.0)
    ]
    assert_close(results[0].drift, 0.3 * x - 1.6**2 / 2 * x)
    assert_close(results[0].drift, results[1].drift)
    assert_close(results[1].log_weight_rate, 2 * results[0].log_weight_rate)
    assert results[0].diffusion_std == 0.0


@pytest.mark.parametrize("variant", ["sde", "ode"])
def test_zero_forward_diffusion_requires_no_laplacian(variant, ensemble):
    x = ensemble.positions
    result = afdps_dynamics(
        x,
        torch.ones_like(x),
        PotentialDerivatives(x, None),
        forward_drift=0.4,
        forward_diffusion=0.0,
        variant=variant,
    )
    assert_close(result.drift, -0.4 * x)
    assert_close(result.log_weight_rate, 0.4 * x.square().flatten(1).sum(1))
    assert result.diffusion_std == 0.0


@pytest.mark.parametrize("variant", ["sde", "ode"])
def test_constant_potential_recovers_prior_dynamics_and_zero_rates(variant, ensemble):
    x = ensemble.positions.requires_grad_()
    parameter = torch.tensor(0.8, dtype=x.dtype, requires_grad=True)
    result = afdps_dynamics(
        x,
        -parameter * x,
        PotentialDerivatives(torch.zeros_like(x), x.new_zeros(x.shape[0])),
        forward_drift=0.2,
        forward_diffusion=1.3,
        variant=variant,
    )
    factor = 1.3**2 if variant == "sde" else 1.3**2 / 2
    assert_close(result.drift, -0.2 * x - factor * parameter * x)
    assert_close(result.log_weight_rate, x.new_zeros(x.shape[0]))
    assert not result.drift.requires_grad
    assert not result.log_weight_rate.requires_grad
    assert x.grad is None
    assert parameter.grad is None


@pytest.mark.parametrize("step_size", [0.04, 0.25])
@pytest.mark.parametrize("diffusion_std", [0.0, 1.7])
def test_euler_matches_explicit_noise_and_log_euler_without_mutating_inputs(
    ensemble, step_size, diffusion_std
):
    x = ensemble.positions
    original_positions = x.clone()
    original_log_weights = ensemble.log_weights.clone()
    dynamics = ParticleDynamics(0.2 - x, x.new_tensor([0.7, -0.4, 0.2]), diffusion_std)
    generator = torch.Generator().manual_seed(71)
    reference_generator = torch.Generator().manual_seed(71)
    initial_rng = generator.get_state().clone()
    global_rng = torch.random.get_rng_state().clone()
    expected = x + step_size * dynamics.drift
    if diffusion_std:
        noise = torch.randn(x.shape, dtype=x.dtype, generator=reference_generator)
        expected += diffusion_std * math.sqrt(step_size) * noise
    result = euler_step(ensemble, dynamics, step_size=step_size, generator=generator)
    assert_close(result.positions, expected, rtol=0, atol=0)
    expected_log_weights = torch.log_softmax(
        ensemble.log_weights + step_size * dynamics.log_weight_rate, dim=0
    )
    assert_close(result.log_weights, expected_log_weights)
    assert torch.isneginf(result.log_weights[1])
    assert_close(generator.get_state(), reference_generator.get_state())
    if not diffusion_std:
        assert_close(generator.get_state(), initial_rng)
    assert_close(torch.random.get_rng_state(), global_rng)
    assert_close(ensemble.positions, original_positions)
    assert_close(ensemble.log_weights, original_log_weights)
    assert not result.positions.requires_grad
    assert not result.log_weights.requires_grad


def test_euler_old_weight_rate_is_independent_of_this_steps_noise(ensemble):
    x = ensemble.positions
    dynamics = afdps_dynamics(
        x,
        -x,
        PotentialDerivatives(x**3, 3 * x.square().flatten(1).sum(1)),
        forward_drift=0.2,
        forward_diffusion=1.1,
        variant="sde",
    )
    first, second = [
        euler_step(
            ensemble,
            dynamics,
            step_size=0.1,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in (11, 29)
    ]
    assert not torch.allclose(first.positions, second.positions)
    assert_close(first.log_weights, second.log_weights, rtol=0, atol=0)
    assert_close(
        first.weights,
        torch.softmax(ensemble.log_weights + 0.1 * dynamics.log_weight_rate, dim=0),
    )


def test_log_weight_and_rate_shifts_cancel_and_zero_mass_stays_zero(ensemble):
    x = ensemble.positions
    rate = x.new_tensor([0.3, -0.6, 0.9])
    results = []
    for log_shift, rate_shift in ((0.0, 0.0), (1e6, 0.0), (0.0, 1e5)):
        results.append(
            euler_step(
                ParticleEnsemble(x, ensemble.log_weights + log_shift),
                ParticleDynamics(torch.zeros_like(x), rate + rate_shift, 0.0),
                step_size=0.1,
                generator=torch.Generator().manual_seed(0),
            )
        )
    for result in results:
        assert_close(result.weights, results[0].weights)
        assert result.weights[1] == 0
    # Avoid letting a huge common reaction rate erase pre-existing relative mass.
    result = euler_step(
        ensemble,
        ParticleDynamics(torch.zeros_like(x), x.new_full((3,), 1e20), 0.0),
        step_size=1.0,
        generator=torch.Generator(),
    )
    assert_close(result.weights, ensemble.weights)


def test_ula_exact_noise_recomputes_score_and_gradient_at_fixed_next_time():
    x = torch.tensor([[0.2, -0.4], [1.1, 0.7]], dtype=torch.float64, requires_grad=True)
    original = x.detach().clone()
    layer = torch.nn.Linear(2, 2, bias=False, dtype=x.dtype)
    with torch.no_grad():
        layer.weight.copy_(x.new_tensor([[-0.8, 0.2], [0.1, -0.5]]))
    potential_scale = torch.tensor(0.6, dtype=x.dtype, requires_grad=True)
    score_calls, potential_calls = [], []

    def score(positions, reverse_time):
        score_calls.append((positions.clone(), reverse_time, torch.is_grad_enabled()))
        return layer(positions) + reverse_time

    def potential(positions):
        potential_calls.append(positions.detach().clone())
        return 0.5 * potential_scale * positions.square().sum(1)

    config = CorrectorConfig(num_steps=3, step_size=0.07)
    generator = torch.Generator().manual_seed(23)
    reference_generator = torch.Generator().manual_seed(23)
    global_rng = torch.random.get_rng_state().clone()
    expected = original.clone()
    expected_inputs = []
    for _ in range(config.num_steps):
        expected_inputs.append(expected.clone())
        phi = expected @ layer.weight.detach().T + 0.8
        gradient = 0.6 * expected
        noise = torch.randn(
            expected.shape, dtype=x.dtype, generator=reference_generator
        )
        expected = expected + config.step_size * (phi - gradient)
        expected += math.sqrt(2 * config.step_size) * noise
    result = ula_corrector(
        x, score, potential, reverse_time=0.8, config=config, generator=generator
    )
    assert_close(result, expected)
    assert len(score_calls) == len(potential_calls) == config.num_steps
    for (state, time, grad_enabled), potential_state, expected_state in zip(
        score_calls, potential_calls, expected_inputs
    ):
        assert time == 0.8
        assert not grad_enabled
        assert_close(state, expected_state)
        assert_close(potential_state, expected_state)
    assert not result.requires_grad
    assert x.grad is None
    assert layer.weight.grad is None
    assert potential_scale.grad is None
    assert_close(x, original)
    assert_close(generator.get_state(), reference_generator.get_state())
    assert_close(torch.random.get_rng_state(), global_rng)


def test_zero_corrector_steps_make_no_model_calls_or_random_draws(ensemble):
    score, potential = Mock(), Mock()
    generator = torch.Generator().manual_seed(19)
    state = generator.get_state().clone()
    result = ula_corrector(
        ensemble.positions,
        score,
        potential,
        reverse_time=1.0,
        config=CorrectorConfig(num_steps=0, step_size=0.1),
        generator=generator,
    )
    assert_close(result, ensemble.positions)
    score.assert_not_called()
    potential.assert_not_called()
    assert_close(generator.get_state(), state)


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            torch.float64,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "mps",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_kernels_preserve_supported_dtype_and_device(device, dtype):
    x = torch.tensor([[0.2, -0.1], [0.3, 0.4]], dtype=dtype, device=device)
    dynamics = afdps_dynamics(
        x,
        -x,
        PotentialDerivatives(x, x.new_full((2,), 2.0)),
        forward_drift=0.1,
        forward_diffusion=0.7,
        variant="sde",
    )
    predicted = euler_step(
        ParticleEnsemble.uniform(x),
        dynamics,
        step_size=0.01,
        generator=torch.Generator(device=x.device).manual_seed(3),
    )
    corrected = ula_corrector(
        predicted.positions,
        negative_score,
        quadratic_potential,
        reverse_time=0.9,
        config=CorrectorConfig(num_steps=2, step_size=0.01),
        generator=torch.Generator(device=x.device).manual_seed(5),
    )
    for value in (
        dynamics.drift,
        dynamics.log_weight_rate,
        predicted.positions,
        predicted.log_weights,
        corrected,
    ):
        assert value.dtype == dtype
        assert value.device == x.device
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kernels_disable_outer_cpu_autocast_including_score_and_potential(dtype):
    x = torch.tensor([[0.4, -0.2], [0.7, 0.3]], dtype=dtype)
    precision = x.new_tensor([[0.8, 0.1], [0.1, 0.6]])
    operator = x.new_tensor([[1.0, 0.3], [-0.2, 0.7]])

    def score(positions, reverse_time):
        return -(positions @ precision)

    def potential(positions):
        return 0.5 * (positions @ operator).square().sum(1)

    def run():
        dynamics = afdps_dynamics(
            x,
            -x,
            PotentialDerivatives(x, x.new_full((2,), 2.0)),
            forward_drift=0.1,
            forward_diffusion=0.8,
            variant="sde",
        )
        predicted = euler_step(
            ParticleEnsemble.uniform(x),
            dynamics,
            step_size=0.02,
            generator=torch.Generator().manual_seed(6),
        )
        corrected = ula_corrector(
            x,
            score,
            potential,
            reverse_time=0.5,
            config=CorrectorConfig(num_steps=2, step_size=0.03),
            generator=torch.Generator().manual_seed(9),
        )
        return dynamics.drift, dynamics.log_weight_rate, predicted.positions, corrected

    expected = run()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = run()
    for result, reference in zip(actual, expected):
        assert result.dtype == dtype
        assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_drift": float("nan")},
        {"forward_drift": float("inf")},
        {"forward_drift": True},
        {"forward_drift": torch.tensor(0.0)},
        {"forward_diffusion": -0.1},
        {"forward_diffusion": float("nan")},
        {"forward_diffusion": float("inf")},
        {"forward_diffusion": False},
        {"forward_diffusion": torch.tensor(1.0)},
        {"forward_diffusion": 1e200},
        {"variant": "unknown"},
    ],
)
def test_dynamics_reject_invalid_coefficients(kwargs, ensemble):
    x = ensemble.positions
    controls = {"forward_drift": 0.0, "forward_diffusion": 1.0, "variant": "sde"}
    controls.update(kwargs)
    with pytest.raises(ValueError):
        afdps_dynamics(
            x, -x, PotentialDerivatives(x, x.new_zeros(x.shape[0])), **controls
        )


@pytest.mark.parametrize("field", ["score", "gradient", "laplacian"])
@pytest.mark.parametrize(
    "problem", ["shape", "dtype", "device", "nonfinite", "missing"]
)
def test_dynamics_validate_score_and_potential_tensors(field, problem, ensemble):
    x = ensemble.positions
    values = {"score": -x, "gradient": x, "laplacian": x.new_zeros(x.shape[0])}
    reference = values[field]
    bad_values = {
        "shape": reference.unsqueeze(-1),
        "dtype": reference.float(),
        "device": torch.empty(reference.shape, dtype=x.dtype, device="meta"),
        "nonfinite": torch.full_like(reference, float("nan")),
        "missing": None,
    }
    values[field] = bad_values[problem]
    with pytest.raises((TypeError, ValueError)):
        afdps_dynamics(
            x,
            values["score"],
            PotentialDerivatives(values["gradient"], values["laplacian"]),
            forward_drift=0.0,
            forward_diffusion=1.0,
            variant="sde",
        )


@pytest.mark.parametrize(
    "positions",
    [
        torch.zeros(2),
        torch.empty(0, 2),
        torch.empty(2, 0),
        torch.ones(2, 1, dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.float16),
        torch.tensor([[float("nan")]]),
        None,
    ],
)
def test_dynamics_reject_invalid_positions(positions):
    with pytest.raises((TypeError, ValueError)):
        afdps_dynamics(
            positions,
            torch.zeros(2, 1),
            PotentialDerivatives(torch.zeros(2, 1), None),
            forward_drift=0.0,
            forward_diffusion=0.0,
            variant="ode",
        )


@pytest.mark.parametrize(
    "step_size", [0, -0.1, float("nan"), float("inf"), True, torch.tensor(0.1)]
)
def test_euler_rejects_invalid_step_size(step_size, ensemble):
    x = ensemble.positions
    with pytest.raises(ValueError, match="step_size"):
        euler_step(
            ensemble,
            ParticleDynamics(torch.zeros_like(x), x.new_zeros(3), 0.0),
            step_size=step_size,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize(
    "diffusion", [-0.1, float("nan"), float("inf"), False, torch.tensor(1.0)]
)
def test_euler_rejects_invalid_diffusion(diffusion, ensemble):
    x = ensemble.positions
    with pytest.raises(ValueError, match="diffusion_std"):
        euler_step(
            ensemble,
            ParticleDynamics(torch.zeros_like(x), x.new_zeros(3), diffusion),
            step_size=0.1,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("field", ["drift", "log_weight_rate"])
@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite"])
def test_euler_rejects_invalid_dynamics_tensors(field, problem, ensemble):
    x = ensemble.positions
    fields = {"drift": torch.zeros_like(x), "log_weight_rate": x.new_zeros(3)}
    reference = fields[field]
    fields[field] = {
        "shape": reference.unsqueeze(-1),
        "dtype": reference.float(),
        "nonfinite": torch.full_like(reference, float("inf")),
    }[problem]
    with pytest.raises(ValueError):
        euler_step(
            ensemble,
            ParticleDynamics(**fields, diffusion_std=0.0),
            step_size=0.1,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("kernel", ["euler", "ula"])
@pytest.mark.parametrize("generator", [None, 17])
def test_kernels_require_an_explicit_generator(kernel, generator, ensemble):
    with pytest.raises(TypeError, match="Generator"):
        if kernel == "euler":
            euler_step(
                ensemble,
                ParticleDynamics(
                    torch.zeros_like(ensemble.positions),
                    ensemble.positions.new_zeros(3),
                    0.0,
                ),
                step_size=0.1,
                generator=generator,
            )
        else:
            ula_corrector(
                ensemble.positions,
                negative_score,
                quadratic_potential,
                reverse_time=0.0,
                config=CorrectorConfig(num_steps=0),
                generator=generator,
            )


@pytest.mark.parametrize("kernel", ["euler", "ula"])
def test_kernels_reject_a_generator_on_a_different_device(kernel, ensemble):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        pytest.skip("No second generator device available")
    generator = torch.Generator(device=device)
    with pytest.raises(ValueError, match="share device"):
        if kernel == "euler":
            euler_step(
                ensemble,
                ParticleDynamics(
                    torch.zeros_like(ensemble.positions),
                    ensemble.positions.new_zeros(3),
                    0.0,
                ),
                step_size=0.1,
                generator=generator,
            )
        else:
            ula_corrector(
                ensemble.positions,
                negative_score,
                quadratic_potential,
                reverse_time=0.0,
                config=CorrectorConfig(),
                generator=generator,
            )


@pytest.mark.parametrize(
    "reverse_time", [-0.1, float("nan"), float("inf"), True, torch.tensor(1.0)]
)
def test_ula_rejects_invalid_reverse_time(reverse_time, ensemble):
    with pytest.raises(ValueError, match="reverse_time"):
        ula_corrector(
            ensemble.positions,
            negative_score,
            quadratic_potential,
            reverse_time=reverse_time,
            config=CorrectorConfig(),
            generator=torch.Generator(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_steps": -1},
        {"num_steps": 1.5},
        {"num_steps": True},
        {"step_size": 0},
        {"step_size": -0.1},
        {"step_size": float("nan")},
        {"step_size": float("inf")},
        {"step_size": True},
        {"step_size": torch.tensor(0.1)},
    ],
)
def test_corrector_config_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        CorrectorConfig(**kwargs)


def test_ula_requires_a_corrector_config(ensemble):
    with pytest.raises(TypeError, match="CorrectorConfig"):
        ula_corrector(
            ensemble.positions,
            negative_score,
            quadratic_potential,
            reverse_time=0.0,
            config=None,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite", "missing"])
def test_ula_rejects_invalid_score_outputs_before_using_potential(problem, ensemble):
    x = ensemble.positions
    value = {
        "shape": x[:, 0],
        "dtype": x.float(),
        "nonfinite": torch.full_like(x, float("nan")),
        "missing": None,
    }[problem]
    potential = Mock()
    with pytest.raises((TypeError, ValueError)):
        ula_corrector(
            x,
            Mock(return_value=value),
            potential,
            reverse_time=0.1,
            config=CorrectorConfig(),
            generator=torch.Generator(),
        )
    potential.assert_not_called()


def test_euler_rejects_overflow_in_centered_weight_rates(ensemble):
    x = ensemble.positions
    maximum = torch.finfo(x.dtype).max
    with pytest.raises(ValueError, match="finite"):
        euler_step(
            ensemble,
            ParticleDynamics(
                torch.zeros_like(x), x.new_tensor([maximum, -maximum, 0]), 0.0
            ),
            step_size=0.1,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("kernel", ["coefficients", "euler", "ula"])
def test_kernels_reject_nonfinite_results_instead_of_returning_invalid_particles(
    kernel,
):
    x = torch.ones(2, 1, dtype=torch.float64)
    large = torch.full_like(x, torch.finfo(x.dtype).max)
    with pytest.raises(ValueError, match="finite"):
        if kernel == "coefficients":
            afdps_dynamics(
                x,
                large,
                PotentialDerivatives(torch.zeros_like(x), x.new_zeros(2)),
                forward_drift=0.0,
                forward_diffusion=2.0,
                variant="sde",
            )
        elif kernel == "euler":
            euler_step(
                ParticleEnsemble.uniform(x),
                ParticleDynamics(large, x.new_zeros(2), 0.0),
                step_size=2.0,
                generator=torch.Generator(),
            )
        else:
            ula_corrector(
                x,
                Mock(return_value=large),
                quadratic_potential,
                reverse_time=0.0,
                config=CorrectorConfig(step_size=2.0),
                generator=torch.Generator(),
            )
