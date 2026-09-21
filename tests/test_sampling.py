from dataclasses import FrozenInstanceError
from functools import partial
from unittest.mock import Mock

import pytest
import torch
from torch.testing import assert_close

from afdps import (
    BrownianSchedule,
    CorrectorConfig,
    DenseLinearOperator,
    GaussianLikelihood,
    ParticleEnsemble,
    SamplerConfig,
    condition_gaussian,
    sample,
    sample_gaussian,
)


def zero_potential(x):
    return x.flatten(1).sum(dim=1) * 0


def test_sample_retains_initial_weights_without_double_likelihood_or_mutation():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    log_weights = torch.tensor([-1.0, 0.0], requires_grad=True)
    initial = ParticleEnsemble(x, log_weights)
    sampler = partial(
        sample,
        score=lambda x, t: -x,
        potential=lambda x: x.square().flatten(1).sum(dim=1),
        schedule=BrownianSchedule(diffusion_std=0),
        config=SamplerConfig(num_particles=2, num_steps=3, resample_ess_fraction=None),
    )
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state().clone()
    result = sampler(initial=initial, generator=generator, time_grid=[0, 0.1, 0.7, 1])
    assert_close(result.ensemble.positions, x)
    assert_close(result.ensemble.weights, initial.weights)
    assert_close(generator.get_state(), state)
    assert not result.ensemble.positions.requires_grad
    assert not result.ensemble.log_weights.requires_grad
    assert len(result.diagnostics) == 3
    assert [d.reverse_time for d in result.diagnostics] == [0.1, 0.7, 1.0]
    assert [d.step for d in result.diagnostics] == [1, 2, 3]
    assert not any(d.resampled for d in result.diagnostics)
    assert [d.step_size for d in result.diagnostics] == pytest.approx([0.1, 0.6, 0.3])
    result.ensemble.positions.zero_()
    result.ensemble.log_weights.zero_()
    assert_close(x, torch.tensor([[1.0], [2.0]]))
    assert_close(log_weights, torch.tensor([-1.0, 0.0]))


@pytest.mark.parametrize("threshold", [None, 1.0])
def test_ode_old_state_weights_new_time_corrector_and_resampling_order(threshold):
    class Schedule:
        terminal_time = 1.0

        def forward_drift(self, t):
            return 0.2 + 0.1 * t

        def forward_diffusion(self, t):
            return 0.7 + 0.2 * t

    calls = []

    def score(x, t):
        calls.append((t, x.clone()))
        return -(1 + t) * x

    x = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
    initial = ParticleEnsemble(x, x.new_tensor([-0.3, -0.7, 0.0]))
    sampler = partial(
        sample,
        score,
        lambda x: 0.5 * x.square().sum(dim=1),
        Schedule(),
        config=SamplerConfig(
            variant="ode",
            num_particles=3,
            num_steps=1,
            corrector=CorrectorConfig(num_steps=1, step_size=0.05),
            resample_ess_fraction=threshold,
        ),
    )
    reference_rng = torch.Generator().manual_seed(19)
    predicted = 0.555 * x  # H = -0.2*x - 0.5*(0.7**2)*x at t=0.
    noise = torch.randn(x.shape, dtype=x.dtype, generator=reference_rng)
    corrected = 0.85 * predicted + (0.1**0.5) * noise
    expected_log_weights = torch.log_softmax(
        initial.log_weights + 0.445 * x[:, 0].square(), dim=0
    )
    expected_ess = expected_log_weights.exp().square().sum().reciprocal().item() / 3
    if threshold is not None:
        ancestors = torch.multinomial(
            expected_log_weights.exp(), 3, replacement=True, generator=reference_rng
        )
        corrected = corrected[ancestors]
        expected_log_weights = torch.zeros_like(expected_log_weights)
    result = sampler(initial=initial, generator=torch.Generator().manual_seed(19))
    assert [time for time, _ in calls] == [0.0, 1.0]
    assert_close(calls[0][1], x)
    assert_close(calls[1][1], predicted)
    assert_close(result.ensemble.positions, corrected)
    assert_close(result.ensemble.log_weights, expected_log_weights)
    assert result.diagnostics[0].ess_fraction == pytest.approx(expected_ess)
    assert result.diagnostics[0].resampled == (threshold is not None)


def test_relative_ess_threshold_is_strict_and_reset_weights_are_uniform():
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    initial = ParticleEnsemble(x, x.new_tensor([-torch.inf, 0.0, -torch.inf]))
    for threshold, should_resample in ((1 / 3, False), (0.4, True)):
        sampler = partial(
            sample,
            lambda x, t: -x,
            zero_potential,
            BrownianSchedule(diffusion_std=0),
            config=SamplerConfig(
                num_particles=3, num_steps=1, resample_ess_fraction=threshold
            ),
        )
        result = sampler(initial=initial, generator=torch.Generator().manual_seed(4))
        assert result.diagnostics[0].resampled == should_resample
        if should_resample:
            assert_close(result.ensemble.positions, x.new_full((3, 1), 2.0))
            assert_close(result.ensemble.log_weights, x.new_zeros(3))
        else:
            assert_close(result.ensemble.positions, x)
            assert_close(result.ensemble.weights, initial.weights)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sde_reproducibility_global_rng_isolation_and_detached_score(dtype):
    score_parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=dtype))
    initial = ParticleEnsemble.uniform(torch.ones(4, 1, dtype=dtype))
    sampler = partial(
        sample,
        lambda x, t: -score_parameter * x,
        zero_potential,
        BrownianSchedule(),
        initial=lambda n, *, generator: initial,
        config=SamplerConfig(num_particles=4, num_steps=5, resample_ess_fraction=None),
    )
    global_state = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = sampler(generator=torch.Generator().manual_seed(6))
    with torch.inference_mode():
        second = sampler(generator=torch.Generator().manual_seed(6))
    assert_close(first.ensemble.positions, second.ensemble.positions, rtol=0, atol=0)
    assert first.diagnostics == second.diagnostics
    assert_close(first.ensemble.weights, torch.full((4,), 0.25, dtype=dtype))
    assert first.ensemble.positions.dtype == dtype
    assert not first.ensemble.positions.requires_grad
    assert score_parameter.grad is None
    assert_close(torch.random.get_rng_state(), global_state)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("variant", ["sde", "ode"])
def test_stage_two_preserves_mps_device_and_resampling(variant):
    x = torch.tensor([[0.2], [1.0], [-0.5]], device="mps")
    initial = ParticleEnsemble(x, x.new_tensor([-float("inf"), 0.0, -float("inf")]))
    sampler = partial(
        sample,
        lambda x, t: -x,
        zero_potential,
        BrownianSchedule(diffusion_std=0.2),
        config=SamplerConfig(
            variant=variant,
            num_particles=3,
            num_steps=2,
            resample_ess_fraction=0.5,
            corrector=CorrectorConfig() if variant == "ode" else None,
        ),
    )
    result = sampler(
        initial=initial, generator=torch.Generator(device=x.device).manual_seed(8)
    )
    assert result.ensemble.positions.device == x.device
    assert result.ensemble.log_weights.device == x.device
    assert result.diagnostics[0].resampled
    assert torch.isfinite(result.ensemble.positions).all()


def test_resampling_rejects_implicit_global_generator():
    ensemble = ParticleEnsemble.uniform(torch.zeros(2, 1))
    state = torch.random.get_rng_state().clone()
    with pytest.raises(TypeError, match="explicit torch.Generator"):
        ensemble.multinomial_resample(generator=None)
    assert_close(torch.random.get_rng_state(), state)


def test_sampler_preserves_float32_under_autocast():
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.tensor([[1.1, 0.7]])), torch.tensor([0.4]), 0.8
    )
    initial = ParticleEnsemble.uniform(torch.tensor([[0.2, 1.3], [1.0, -0.3]]))
    sampler = partial(
        sample,
        lambda x, t: x @ torch.tensor([[-0.5, 0.1], [0.1, -0.5]]),
        likelihood,
        BrownianSchedule(),
        config=SamplerConfig(num_particles=2, num_steps=3, resample_ess_fraction=None),
    )
    expected = sampler(initial=initial, generator=torch.Generator().manual_seed(6))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = sampler(initial=initial, generator=torch.Generator().manual_seed(6))
    assert_close(actual.ensemble.positions, expected.ensemble.positions, rtol=0, atol=0)
    assert_close(actual.ensemble.weights, expected.ensemble.weights, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["sde", "ode"])
@pytest.mark.parametrize("seed", [11, 72])
def test_two_stage_sampler_matches_analytic_gaussian_posterior(variant, seed):
    schedule = BrownianSchedule(terminal_time=0.5, diffusion_std=1.0)
    prior_variance = 0.5
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.tensor([[1.0]], dtype=torch.float64)),
        torch.tensor([0.8], dtype=torch.float64),
        1.0,
    )
    initial_std = (prior_variance + schedule.noise_variance(0)) ** 0.5
    gaussian = condition_gaussian(likelihood, reference_std=initial_std)
    target = condition_gaussian(likelihood, reference_std=prior_variance**0.5)
    sampler = partial(
        sample,
        score=lambda x, t: -x / (prior_variance + schedule.noise_variance(t)),
        potential=likelihood,
        schedule=schedule,
        initial=partial(sample_gaussian, gaussian),
        config=SamplerConfig(
            variant=variant,
            num_particles=6000,
            num_steps=120,
            resample_ess_fraction=None,
            corrector=CorrectorConfig(num_steps=1, step_size=0.001)
            if variant == "ode"
            else None,
        ),
    )
    result = sampler(generator=torch.Generator().manual_seed(seed))
    mean = result.ensemble.mean()
    centered = result.ensemble.positions - mean
    variance = (result.ensemble.weights[:, None] * centered.square()).sum(dim=0)
    # Finite particles, Euler steps, and (for ODE) ULA have numerical error.
    assert_close(mean, target.mean, rtol=0, atol=0.04)
    assert_close(variance, target.covariance.diagonal(), rtol=0, atol=0.04)
    assert len(result.diagnostics) == 120
    assert result.diagnostics[-1].reverse_time == schedule.terminal_time
    assert not any(d.resampled for d in result.diagnostics)


def test_weighted_ode_refines_to_exact_gaussian_flow_for_fixed_particles():
    schedule = BrownianSchedule(terminal_time=0.5, diffusion_std=1.0)
    initial = ParticleEnsemble(
        torch.tensor([[-2.0], [-0.5], [0.7], [1.5]], dtype=torch.float64),
        torch.tensor([-0.3, 0.0, -1.0, -0.7], dtype=torch.float64),
    )

    def potential(x):
        return 0.5 * (x[:, 0] - 0.8).square()

    target_positions = initial.positions * (0.5 / 1.0) ** 0.5
    target_weights = torch.softmax(
        initial.log_weights
        + potential(initial.positions)
        - potential(target_positions),
        dim=0,
    )
    errors = []
    for steps in (4, 80):
        sampler = partial(
            sample,
            lambda x, t: -x / (0.5 + schedule.noise_variance(t)),
            potential,
            schedule,
            config=SamplerConfig(
                variant="ode",
                num_particles=4,
                num_steps=steps,
                corrector=CorrectorConfig(num_steps=0),
                resample_ess_fraction=None,
            ),
        )
        rng = torch.Generator().manual_seed(0)
        state = rng.get_state().clone()
        result = sampler(initial=initial, generator=rng)
        assert_close(rng.get_state(), state)
        errors.append(
            (result.ensemble.positions - target_positions).abs().max().item()
            + (result.ensemble.weights - target_weights).abs().max().item()
        )
    assert errors[1] < errors[0] / 10
    assert errors[1] < 0.01


@pytest.mark.parametrize(
    "grid",
    [
        [0, 1],
        [0, 0.2, 0.7],
        [0.1, 0.5, 1],
        [0, 0, 1],
        [0, 1.5, 1],
        [0, float("nan"), 1],
    ],
)
def test_invalid_grid_fails_before_initializer_or_rng(grid):
    initializer = Mock()
    sampler = partial(
        sample,
        Mock(),
        Mock(),
        BrownianSchedule(),
        initializer,
        config=SamplerConfig(num_steps=2),
    )
    rng = torch.Generator().manual_seed(7)
    state = rng.get_state().clone()
    with pytest.raises(ValueError, match="time_grid"):
        sampler(generator=rng, time_grid=grid)
    initializer.assert_not_called()
    assert_close(rng.get_state(), state)


def test_missing_or_invalid_initial_state_fails_explicitly():
    sampler = partial(
        sample,
        lambda x, t: -x,
        zero_potential,
        BrownianSchedule(),
        config=SamplerConfig(num_particles=2, num_steps=1),
    )
    with pytest.raises(TypeError, match="initial"):
        sampler(generator=torch.Generator())
    with pytest.raises(TypeError, match="explicit torch.Generator"):
        sampler(initial=None, generator=None)
    with pytest.raises(TypeError, match="ParticleEnsemble"):
        sampler(initial=lambda n, *, generator: None, generator=torch.Generator())
    with pytest.raises(TypeError, match="ParticleEnsemble"):
        sampler(initial=torch.zeros(2, 1), generator=torch.Generator())
    with pytest.raises(ValueError, match="num_particles"):
        sampler(
            initial=ParticleEnsemble.uniform(torch.zeros(1, 1)),
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("variant", ["sde", "ode"])
def test_callable_initialization_matches_explicit_function_composition(variant):
    likelihood = GaussianLikelihood(DenseLinearOperator(torch.eye(2)), torch.ones(2), 1)
    gaussian = condition_gaussian(likelihood, reference_std=1)
    initializer = partial(sample_gaussian, gaussian)
    config = SamplerConfig(
        variant=variant,
        num_particles=4,
        num_steps=3,
        corrector=CorrectorConfig() if variant == "ode" else None,
    )
    run = partial(
        sample, lambda x, t: -x, likelihood, BrownianSchedule(), config=config
    )
    first_rng = torch.Generator().manual_seed(18)
    second_rng = torch.Generator().manual_seed(18)
    first = run(initial=initializer, generator=first_rng)
    initial = initializer(config.num_particles, generator=second_rng)
    second = run(initial=initial, generator=second_rng)
    assert_close(first.ensemble.positions, second.ensemble.positions, rtol=0, atol=0)
    assert_close(
        first.ensemble.log_weights, second.ensemble.log_weights, rtol=0, atol=0
    )
    assert first.diagnostics == second.diagnostics
    assert_close(first_rng.get_state(), second_rng.get_state())


def test_sample_uses_default_config_without_requiring_a_sampler_object():
    class CallableEnsemble(ParticleEnsemble):
        def __call__(self, *args, **kwargs):
            pytest.fail("an existing ensemble must not be invoked as an initializer")

    initial = CallableEnsemble.uniform(torch.ones(16, 1))
    result = sample(
        lambda x, t: -x,
        zero_potential,
        BrownianSchedule(diffusion_std=0),
        initial,
        generator=torch.Generator(),
    )
    assert len(result.diagnostics) == 100
    assert_close(result.ensemble.positions, initial.positions)
    assert_close(result.ensemble.weights, initial.weights)


@pytest.mark.parametrize("name", ["num_particles", "num_steps"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_config_rejects_invalid_counts(name, value):
    with pytest.raises(ValueError, match=name):
        SamplerConfig(**{name: value})


@pytest.mark.parametrize(
    "threshold", [0.0, -0.1, 1.1, float("nan"), float("inf"), True]
)
def test_config_rejects_invalid_ess_threshold(threshold):
    with pytest.raises(ValueError, match="resample_ess_fraction"):
        SamplerConfig(resample_ess_fraction=threshold)


def test_config_requires_explicit_corrector_only_for_ode_and_is_frozen():
    with pytest.raises(ValueError, match="variant"):
        SamplerConfig(variant="dps")
    with pytest.raises(ValueError, match="explicit CorrectorConfig"):
        SamplerConfig(variant="ode")
    with pytest.raises(ValueError, match="only supported"):
        SamplerConfig(corrector=CorrectorConfig())
    config = SamplerConfig(num_particles=1, num_steps=1, resample_ess_fraction=None)
    with pytest.raises(FrozenInstanceError):
        config.num_particles = 2


@pytest.mark.parametrize("count", [-1, 0.5, True])
def test_corrector_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="num_steps"):
        CorrectorConfig(num_steps=count)


@pytest.mark.parametrize("step", [0, -1, True, float("nan"), float("inf")])
def test_corrector_rejects_invalid_step_size(step):
    with pytest.raises(ValueError, match="step_size"):
        CorrectorConfig(step_size=step)
