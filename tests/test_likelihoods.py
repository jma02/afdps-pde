import pytest
import torch
from torch.testing import assert_close

from afdps import DenseLinearOperator, GaussianLikelihood


def test_linear_gaussian_values_gradient_and_laplacian_match_analytic_formula():
    matrix = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
    observation = matrix.new_tensor([0.5, -0.25])
    positions = matrix.new_tensor([[0.1, 0.2], [0.3, -0.4]]).requires_grad_()
    sigma = 0.4
    potential = GaussianLikelihood(DenseLinearOperator(matrix), observation, sigma)
    values = potential(positions)
    residual = positions @ matrix.T - observation
    assert values.shape == (2,)
    assert values.dtype == positions.dtype
    assert values.device == positions.device
    assert_close(values, residual.square().sum(dim=1) / (2 * sigma**2))

    gradient = torch.autograd.grad(values.sum(), positions, create_graph=True)[0]
    assert_close(gradient, residual @ matrix / sigma**2)
    laplacian = torch.zeros(2, dtype=positions.dtype)
    for dimension in range(positions.shape[1]):
        hessian_row = torch.autograd.grad(
            gradient[:, dimension].sum(), positions, retain_graph=True
        )[0]
        laplacian += hessian_row[:, dimension]
    assert_close(laplacian, (matrix.square().sum() / sigma**2).expand(2))


def test_rectangular_underdetermined_operator_and_likelihood():
    matrix = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    positions = matrix.new_tensor(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [2.0, -1.0, 1.0]]
    ).requires_grad_()
    operator = DenseLinearOperator(matrix)
    expected = matrix.new_tensor([[7.0, 2.0], [2.0, 0.0], [-1.0, 1.0], [4.0, -1.0]])
    assert_close(operator(positions), expected)
    potential = GaussianLikelihood(operator, matrix.new_tensor([1.0, -1.0]), 2.0)
    values = potential(positions)
    assert_close(values, matrix.new_tensor([5.625, 0.25, 1.0, 1.125]))
    gradient = torch.autograd.grad(values.sum(), positions)[0]
    expected_gradient = matrix.new_tensor(
        [[1.5, 0.75, 3.0], [0.25, 0.25, 0.5], [-0.5, 0.5, -1.0], [0.75, 0.0, 1.5]]
    )
    assert_close(gradient, expected_gradient)


def test_nonlinear_operator_retains_second_derivatives():
    positions = torch.tensor([[1.0], [2.0]], dtype=torch.float64, requires_grad=True)
    potential = GaussianLikelihood(
        lambda x: x.square(), positions.new_tensor([0.0]), 1.0
    )
    gradient = torch.autograd.grad(
        potential(positions).sum(), positions, create_graph=True
    )[0]
    curvature = torch.autograd.grad(gradient.sum(), positions)[0]
    assert_close(gradient, 2 * positions.pow(3))
    assert_close(curvature, 6 * positions.square())


def test_event_dimensions_are_summed_not_averaged():
    positions = torch.ones(3, 1, 2, 2)
    potential = GaussianLikelihood(lambda x: x, torch.zeros(1, 2, 2), 2.0)
    assert_close(potential(positions), torch.full((3,), 0.5))


def test_scalar_observation_is_supported_without_cross_particle_broadcast():
    potential = GaussianLikelihood(lambda x: x.sum(dim=1), torch.tensor(1.0), 1.0)
    assert_close(
        potential(torch.tensor([[1.0, 1.0], [2.0, 2.0]])), torch.tensor([0.5, 4.5])
    )


@pytest.mark.parametrize("sigma", [0.0, -0.1, float("nan"), float("inf"), True])
def test_invalid_measurement_noise_is_rejected(sigma):
    with pytest.raises(ValueError, match="noise_std"):
        GaussianLikelihood(lambda x: x, torch.zeros(1), sigma)


def test_observation_shape_mismatch_is_not_silently_broadcast():
    potential = GaussianLikelihood(lambda x: x, torch.zeros(1, 2), 1.0)
    with pytest.raises(ValueError, match="one fixed observation"):
        potential(torch.ones(3, 2))


def test_likelihood_requires_consistent_dtype_and_operator_shape():
    potential = GaussianLikelihood(lambda x: x, torch.zeros(2), 1.0)
    with pytest.raises(ValueError, match="dtype and device"):
        potential(torch.ones(3, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="positions"):
        potential(torch.ones(2))
    wrong_dtype = GaussianLikelihood(lambda x: x.double(), torch.zeros(2), 1.0)
    with pytest.raises(ValueError, match="operator must preserve"):
        wrong_dtype(torch.ones(3, 2))


@pytest.mark.parametrize("observation", [torch.empty(0), torch.tensor([float("nan")])])
def test_invalid_observations_are_rejected(observation):
    with pytest.raises(ValueError, match="observation"):
        GaussianLikelihood(lambda x: x, observation, 1.0)


def test_integer_observation_is_rejected():
    with pytest.raises(TypeError, match="observation"):
        GaussianLikelihood(lambda x: x, torch.ones(1, dtype=torch.long), 1.0)


@pytest.mark.parametrize(
    "matrix", [torch.zeros(2), torch.empty(0, 2), torch.tensor([[float("inf")]])]
)
def test_invalid_operator_matrix_is_rejected(matrix):
    with pytest.raises(ValueError, match="matrix"):
        DenseLinearOperator(matrix)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_operator_and_likelihood_are_rejected(dtype):
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        DenseLinearOperator(torch.ones(2, 2, dtype=dtype))
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        GaussianLikelihood(lambda x: x, torch.zeros(2, dtype=dtype), 1.0)


def test_operator_shape_and_dtype_checks():
    with pytest.raises(TypeError):
        DenseLinearOperator(torch.ones(2, 2, dtype=torch.long))
    operator = DenseLinearOperator(torch.eye(2))
    with pytest.raises(ValueError, match="shape"):
        operator(torch.zeros(3, 1))
    with pytest.raises(ValueError, match="dtype and device"):
        operator(torch.zeros(3, 2, dtype=torch.float64))
