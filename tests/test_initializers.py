from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
import torch
from torch.testing import assert_close

from afdps import DenseLinearOperator, GaussianLikelihood
from afdps.initializers import Gaussian, condition_gaussian, sample_gaussian


def make_correlated_gaussian(dtype=torch.float64, device="cpu"):
    matrix = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=dtype, device=device)
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([1.0, 0.0]), 1.0
    )
    return condition_gaussian(likelihood, reference_std=1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_correlated_moments_and_precision_factor_orientation(dtype):
    gaussian = make_correlated_gaussian(dtype)
    mean = torch.tensor([0.4, 0.2], dtype=dtype)
    covariance = torch.tensor([[0.6, -0.2], [-0.2, 0.4]], dtype=dtype)
    precision = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=dtype)
    assert_close(gaussian.mean, mean)
    assert_close(gaussian.covariance, covariance)

    ensemble = sample_gaussian(
        gaussian, 11, generator=torch.Generator().manual_seed(27)
    )
    noise = torch.randn(11, 2, dtype=dtype, generator=torch.Generator().manual_seed(27))
    # An independent algebraic identity catches using L instead of L.T in
    # sampling, without relying on a statistical covariance estimate.
    whitened = (ensemble.positions - mean) @ torch.linalg.cholesky(precision)
    assert_close(whitened, noise)
    assert ensemble.positions.shape == (11, 2)
    assert ensemble.positions.dtype == dtype
    assert ensemble.positions.device == mean.device
    assert_close(ensemble.log_weights, torch.zeros(11, dtype=dtype))
    assert_close(ensemble.weights, torch.full((11,), 1 / 11, dtype=dtype))
    assert_close(ensemble.effective_sample_size, mean.new_tensor(11.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonunit_reference_and_measurement_scales(dtype):
    matrix = torch.diag(torch.tensor([1.0, 2.0], dtype=dtype))
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([1.0, 2.0]), 0.5
    )
    gaussian = condition_gaussian(likelihood, reference_std=2.0)
    assert_close(gaussian.mean, matrix.new_tensor([16 / 17, 64 / 65]))
    assert_close(gaussian.covariance, torch.diag(matrix.new_tensor([4 / 17, 4 / 65])))


def test_preparation_preserves_float32_inside_cpu_autocast():
    matrix = torch.tensor([[1.1, 1.2], [0.3, 1.4]], dtype=torch.float32)
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([0.7, -0.2]), 0.4
    )
    expected = condition_gaussian(likelihood, reference_std=1.3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = condition_gaussian(likelihood, reference_std=1.3)
        draws = sample_gaussian(actual, 4, generator=torch.Generator().manual_seed(6))
        covariance = actual.covariance
    assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    assert_close(covariance, expected.covariance, rtol=0, atol=0)
    expected_draws = sample_gaussian(
        expected, 4, generator=torch.Generator().manual_seed(6)
    )
    assert_close(draws.positions, expected_draws.positions, rtol=0, atol=0)


def test_cpu_autocast_does_not_round_away_the_reference_precision():
    matrix = torch.tensor([[16.0, 16.0]], dtype=torch.float32)
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([1.0]), 1.0
    )
    expected = condition_gaussian(likelihood, reference_std=1.0)
    # In bfloat16 the Gram entries are 256 and adding a unit ridge rounds back
    # to 256, losing positive definiteness before Cholesky can upcast.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = condition_gaussian(likelihood, reference_std=1.0)
    assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    assert_close(actual.covariance, expected.covariance, rtol=0, atol=0)


def test_rank_deficient_rectangular_operator_retains_nullspace_variance():
    matrix = torch.tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float64)
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([3.0, 6.0]), 1.0
    )
    gaussian = condition_gaussian(likelihood, reference_std=1.0)
    expected_covariance = matrix.new_tensor(
        [[6 / 11, -5 / 11, 0.0], [-5 / 11, 6 / 11, 0.0], [0.0, 0.0, 1.0]]
    )
    assert_close(gaussian.mean, matrix.new_tensor([15 / 11, 15 / 11, 0.0]))
    assert_close(gaussian.covariance, expected_covariance)
    null_vector = matrix.new_tensor([1.0, -1.0, 0.0]) / (2**0.5)
    assert_close(
        null_vector @ gaussian.covariance @ null_vector, matrix.new_tensor(1.0)
    )
    ensemble = sample_gaussian(gaussian, 5, generator=torch.Generator().manual_seed(3))
    assert ensemble.positions.shape == (5, 3)


def test_zero_operator_returns_the_reference_even_with_nonzero_observation():
    matrix = torch.zeros(2, 3, dtype=torch.float64)
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([4.0, -3.0]), 0.7
    )
    gaussian = condition_gaussian(likelihood, reference_std=2.0)
    assert_close(gaussian.mean, matrix.new_zeros(3))
    assert_close(gaussian.covariance, 4 * torch.eye(3, dtype=matrix.dtype))
    ensemble = sample_gaussian(gaussian, 5, generator=torch.Generator().manual_seed(23))
    expected = 2 * torch.randn(
        5, 3, dtype=matrix.dtype, generator=torch.Generator().manual_seed(23)
    )
    assert_close(ensemble.positions, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_dimensional_state_and_single_particle_keep_their_axes(dtype):
    matrix = torch.tensor([[2.0]], dtype=dtype)
    likelihood = GaussianLikelihood(
        DenseLinearOperator(matrix), matrix.new_tensor([4.0]), 1.0
    )
    gaussian = condition_gaussian(likelihood, reference_std=3.0)
    assert_close(gaussian.mean, matrix.new_tensor([72 / 37]))
    assert_close(gaussian.covariance, matrix.new_tensor([[9 / 37]]))
    ensemble = sample_gaussian(gaussian, 1, generator=torch.Generator().manual_seed(4))
    assert ensemble.positions.shape == (1, 1)
    assert_close(ensemble.log_weights, matrix.new_zeros(1))


@pytest.mark.parametrize("seed", [7, 123, 2025])
def test_empirical_moments_match_analytic_distribution(seed):
    gaussian = make_correlated_gaussian()
    count = 20000
    positions = sample_gaussian(
        gaussian, count, generator=torch.Generator().manual_seed(seed)
    ).positions
    expected_mean = positions.new_tensor([0.4, 0.2])
    expected_covariance = positions.new_tensor([[0.6, -0.2], [-0.2, 0.4]])
    sample_mean = positions.mean(dim=0)
    centered = positions - sample_mean
    sample_covariance = centered.T @ centered / (count - 1)

    # Six analytic Monte Carlo standard errors; fixed seeds, generous bounds.
    variances = expected_covariance.diagonal()
    mean_error = (variances / count).sqrt()
    covariance_error = (
        (expected_covariance.square() + variances[:, None] * variances[None, :])
        / (count - 1)
    ).sqrt()
    assert ((sample_mean - expected_mean).abs() < 6 * mean_error).all()
    assert (
        (sample_covariance - expected_covariance).abs() < 6 * covariance_error
    ).all()


def test_only_explicit_generator_advances_and_seeded_draws_repeat():
    global_state = torch.random.get_rng_state().clone()
    gaussian = make_correlated_gaussian()
    generator = torch.Generator().manual_seed(19)
    local_state = generator.get_state().clone()
    first = sample_gaussian(gaussian, 8, generator=generator)
    assert not torch.equal(generator.get_state(), local_state)
    second = sample_gaussian(gaussian, 8, generator=generator)
    assert not torch.equal(first.positions, second.positions)
    repeat = sample_gaussian(gaussian, 8, generator=torch.Generator().manual_seed(19))
    assert_close(first.positions, repeat.positions, rtol=0, atol=0)
    assert_close(torch.random.get_rng_state(), global_state)


def test_conditioning_is_cached_in_a_frozen_data_record():
    with patch("torch.linalg.cholesky", wraps=torch.linalg.cholesky) as cholesky:
        gaussian = make_correlated_gaussian()
        assert isinstance(gaussian, Gaussian)
        generator = torch.Generator().manual_seed(9)
        sample_gaussian(gaussian, 3, generator=generator)
        sample_gaussian(gaussian, 3, generator=generator)
        _ = gaussian.covariance
        cholesky.assert_called_once()
    with pytest.raises(FrozenInstanceError):
        gaussian._mean = gaussian.mean


def test_prepared_distribution_is_detached_and_owns_its_cached_parameters():
    matrix = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    observation = torch.tensor([1.0, 0.0], dtype=matrix.dtype, requires_grad=True)
    likelihood = GaussianLikelihood(DenseLinearOperator(matrix), observation, 1.0)
    gaussian = condition_gaussian(likelihood, reference_std=1.0)
    mean = gaussian.mean
    covariance = gaussian.covariance
    before = sample_gaussian(gaussian, 8, generator=torch.Generator().manual_seed(9))
    assert matrix.requires_grad and observation.requires_grad
    assert matrix.grad is None and observation.grad is None
    assert not mean.requires_grad and not covariance.requires_grad
    assert not before.positions.requires_grad

    with torch.no_grad():
        matrix.fill_(99.0)
        observation.fill_(-99.0)
    gaussian.mean.zero_()
    gaussian.covariance.zero_()
    assert_close(gaussian.mean, mean)
    assert_close(gaussian.covariance, covariance)
    after = sample_gaussian(gaussian, 8, generator=torch.Generator().manual_seed(9))
    assert_close(after.positions, before.positions, rtol=0, atol=0)
    after.positions.zero_()
    assert_close(gaussian.mean, mean)


@pytest.mark.parametrize("reference_std", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_reference_scales_are_rejected(reference_std):
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.eye(2)), torch.zeros(2), 1.0
    )
    with pytest.raises(ValueError, match="reference_std"):
        condition_gaussian(likelihood, reference_std=reference_std)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_counts_fail_before_rng_consumption(count):
    gaussian = make_correlated_gaussian()
    generator = torch.Generator().manual_seed(5)
    state = generator.get_state().clone()
    with pytest.raises(ValueError, match="num_particles"):
        sample_gaussian(gaussian, count, generator=generator)
    assert_close(generator.get_state(), state)


@pytest.mark.parametrize("generator", [None, 3])
def test_implicit_or_invalid_generators_are_rejected(generator):
    gaussian = make_correlated_gaussian()
    state = torch.random.get_rng_state().clone()
    with pytest.raises(TypeError, match="explicit torch.Generator"):
        sample_gaussian(gaussian, 3, generator=generator)
    assert_close(torch.random.get_rng_state(), state)


def test_only_linear_gaussian_likelihoods_are_supported():
    with pytest.raises(TypeError, match="GaussianLikelihood"):
        condition_gaussian(lambda x: x, reference_std=1.0)
    likelihood = GaussianLikelihood(lambda x: x.square(), torch.zeros(2), 1.0)
    with pytest.raises(TypeError, match="DenseLinearOperator"):
        condition_gaussian(likelihood, reference_std=1.0)


@pytest.mark.parametrize(
    "observation", [torch.zeros(1, 2), torch.zeros(1), torch.tensor(1.0)]
)
def test_observation_must_be_an_unbatched_vector_of_the_right_length(observation):
    likelihood = GaussianLikelihood(DenseLinearOperator(torch.eye(2)), observation, 1.0)
    with pytest.raises(ValueError, match="observation must have shape"):
        condition_gaussian(likelihood, reference_std=1.0)


def test_operator_and_observation_must_share_dtype():
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.eye(2)), torch.zeros(2, dtype=torch.float64), 1.0
    )
    with pytest.raises(ValueError, match="dtype and device"):
        condition_gaussian(likelihood, reference_std=1.0)


@pytest.mark.parametrize("reference_std", [1e-30, 1e30])
def test_unrepresentable_reference_precision_is_rejected(reference_std):
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.eye(2)), torch.zeros(2), 1.0
    )
    with pytest.raises(ValueError, match="unrepresentable precision"):
        condition_gaussian(likelihood, reference_std=reference_std)


@pytest.mark.parametrize("noise_std", [1e-100, 1e100])
def test_unrepresentable_measurement_noise_is_rejected(noise_std):
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.eye(2)), torch.zeros(2), noise_std
    )
    with pytest.raises(ValueError, match="noise_std is not representable"):
        condition_gaussian(likelihood, reference_std=1.0)


@pytest.mark.parametrize("matrix_value, observation_value", [(1e30, 1.0), (1e10, 1e30)])
def test_nonfinite_precision_or_natural_mean_fails_clearly(
    matrix_value, observation_value
):
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.tensor([[matrix_value]])),
        torch.tensor([observation_value]),
        1.0,
    )
    with pytest.raises(ValueError, match="nonfinite"):
        condition_gaussian(likelihood, reference_std=1.0)


def test_unrepresentable_covariance_diagnostic_fails_clearly():
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.zeros(1, 1, dtype=torch.float64)),
        torch.zeros(1, dtype=torch.float64),
        1.0,
    )
    # Precision ~1e-320 is representable, but covariance ~1e320 is not.
    gaussian = condition_gaussian(likelihood, reference_std=1e160)
    with pytest.raises(ValueError, match="covariance is nonfinite"):
        _ = gaussian.covariance


def test_loss_of_positive_definiteness_is_reported_without_jitter():
    # In float32, adding the unit ridge to 2**24 is rounded away exactly.
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.tensor([[4096.0, 4096.0]], dtype=torch.float32)),
        torch.zeros(1),
        1.0,
    )
    with pytest.raises(ValueError, match="positive definite.*no jitter"):
        condition_gaussian(likelihood, reference_std=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_sampling_preserves_device_dtype_and_generator_contract(dtype):
    gaussian = make_correlated_gaussian(dtype, device="cuda")
    device = gaussian.mean.device
    first = sample_gaussian(
        gaussian, 8, generator=torch.Generator(device=device).manual_seed(3)
    )
    repeat = sample_gaussian(
        gaussian, 8, generator=torch.Generator(device=device).manual_seed(3)
    )
    assert first.positions.device == first.log_weights.device == device
    assert first.positions.dtype == first.log_weights.dtype == dtype
    assert_close(first.positions, repeat.positions, rtol=0, atol=0)
    assert_close(gaussian.mean.cpu(), torch.tensor([0.4, 0.2], dtype=dtype))
    with pytest.raises(ValueError, match="share device"):
        sample_gaussian(gaussian, 8, generator=torch.Generator())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_operator_and_observation_must_share_device():
    likelihood = GaussianLikelihood(
        DenseLinearOperator(torch.eye(2)), torch.zeros(2, device="cuda"), 1.0
    )
    with pytest.raises(ValueError, match="dtype and device"):
        condition_gaussian(likelihood, reference_std=1.0)
