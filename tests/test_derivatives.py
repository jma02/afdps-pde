from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.testing import assert_close

from afdps.derivatives import PotentialDerivatives, potential_derivatives
from afdps.likelihoods import GaussianLikelihood
from afdps.operators import DenseLinearOperator


def _check_result(result, positions, gradient, laplacian):
    assert isinstance(result, PotentialDerivatives)
    assert result.gradient.shape == positions.shape
    outputs = [result.gradient]
    assert_close(result.gradient, gradient)
    if laplacian is None:
        assert result.laplacian is None
    else:
        assert result.laplacian.shape == (positions.shape[0],)
        assert_close(result.laplacian, laplacian)
        outputs.append(result.laplacian)
    for output in outputs:
        assert output.dtype == positions.dtype
        assert output.device == positions.device
        assert not output.requires_grad
        assert output.grad_fn is None
        assert not output.is_inference()
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute_laplacian", [False, True])
def test_dense_gaussian_uses_analytic_derivatives(
    dtype, compute_laplacian, monkeypatch
):
    matrix = torch.tensor([[1.0, 2.0, -0.5], [-1.0, 0.5, 3.0]], dtype=dtype)
    observation = matrix.new_tensor([0.25, -0.75])
    positions = matrix.new_tensor([[0.1, 0.2, -0.3], [0.3, -0.4, 0.5]])
    sigma = 0.4
    potential = GaussianLikelihood(DenseLinearOperator(matrix), observation, sigma)

    def forbid_autograd(*args, **kwargs):
        pytest.fail("the exact dense Gaussian must not invoke autograd")

    monkeypatch.setattr(torch.autograd, "grad", forbid_autograd)
    result = potential_derivatives(
        potential, positions, compute_laplacian=compute_laplacian
    )
    gradient = (positions @ matrix.T - observation) @ matrix / sigma**2
    laplacian = (matrix.square().sum() / sigma**2).expand(2)
    _check_result(result, positions, gradient, laplacian if compute_laplacian else None)


@pytest.mark.parametrize("sigma", [1e-200, 1e200])
def test_analytic_path_scales_before_squaring(sigma):
    matrix = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    observation = matrix.new_tensor([0.1])
    positions = matrix.new_tensor([[0.2, 0.4], [-0.3, 0.2]])
    potential = GaussianLikelihood(
        DenseLinearOperator(matrix * sigma), observation * sigma, sigma
    )
    result = potential_derivatives(potential, positions)
    _check_result(
        result,
        positions,
        (positions @ matrix.T - observation) @ matrix,
        matrix.square().sum().expand(2),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute_laplacian", [False, True])
def test_nonlinear_image_likelihood(dtype, compute_laplacian):
    positions = torch.linspace(-0.8, 1.2, 24, dtype=dtype).reshape(4, 1, 2, 3)
    positions = positions.transpose(-1, -2)
    assert not positions.is_contiguous()
    observation = torch.linspace(-0.1, 0.4, 6, dtype=dtype).reshape(1, 3, 2)
    sigma = 0.7
    potential = GaussianLikelihood(lambda x: x.square(), observation, sigma)
    result = potential_derivatives(
        potential, positions, compute_laplacian=compute_laplacian
    )
    gradient = 2 * positions * (positions.square() - observation) / sigma**2
    laplacian = (
        ((6 * positions.square() - 2 * observation) / sigma**2).flatten(1).sum(1)
    )
    _check_result(result, positions, gradient, laplacian if compute_laplacian else None)


@pytest.mark.parametrize("compute_laplacian", [False, True])
def test_generic_trace_has_one_backward_per_event_dimension(
    compute_laplacian, monkeypatch
):
    positions = torch.tensor(
        [[0.3, -0.7], [1.2, 0.4], [-0.8, 2.0]], dtype=torch.float64
    )
    calls = []
    autograd_grad = torch.autograd.grad

    def record_grad(*args, **kwargs):
        calls.append(kwargs)
        return autograd_grad(*args, **kwargs)

    def potential(x):
        return x[:, 0].square() * x[:, 1].square() + x[:, 0].sin()

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    rng_state = torch.random.get_rng_state()
    result = potential_derivatives(
        potential, positions, compute_laplacian=compute_laplacian
    )
    a, b = positions.unbind(1)
    gradient = torch.stack((2 * a * b.square() + a.cos(), 2 * b * a.square()), dim=1)
    laplacian = 2 * (a.square() + b.square()) - a.sin()
    _check_result(result, positions, gradient, laplacian if compute_laplacian else None)
    assert len(calls) == 1 + (positions.shape[1] if compute_laplacian else 0)
    assert calls[0]["create_graph"] is compute_laplacian
    assert all(not call.get("create_graph", False) for call in calls[1:])
    assert_close(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize("subclass", ["likelihood", "operator"])
def test_subclasses_use_their_overridden_semantics(subclass):
    class CubicLikelihood(GaussianLikelihood):
        def __call__(self, positions):
            return super().__call__(positions) + positions.pow(3).sum(1)

    class SquaredOperator(DenseLinearOperator):
        def __call__(self, positions):
            return super().__call__(positions).square()

    positions = torch.tensor([[0.2, -0.4], [0.7, 1.2]], dtype=torch.float64)
    matrix = torch.eye(2, dtype=positions.dtype)
    observation = positions.new_zeros(2)
    if subclass == "likelihood":
        potential = CubicLikelihood(DenseLinearOperator(matrix), observation, 1.0)
        gradient = positions + 3 * positions.square()
        laplacian = (1 + 6 * positions).sum(1)
    else:
        potential = GaussianLikelihood(SquaredOperator(matrix), observation, 1.0)
        gradient = 2 * positions.pow(3)
        laplacian = 6 * positions.square().sum(1)
    _check_result(
        potential_derivatives(potential, positions), positions, gradient, laplacian
    )


@pytest.mark.parametrize(
    "kind", ["affine", "constant", "parameter_affine", "parameter_constant"]
)
@pytest.mark.parametrize("compute_laplacian", [False, True])
def test_affine_and_graph_connected_constant_potentials(kind, compute_laplacian):
    positions = torch.tensor([[0.2, -0.4], [1.0, 2.0]], dtype=torch.float64)
    parameter = torch.nn.Parameter(positions.new_tensor([1.5, -0.5]))
    coefficient = parameter if kind.startswith("parameter") else parameter.detach()
    if kind.endswith("constant"):
        coefficient = 0 * coefficient

    def potential(x):
        return (x * coefficient).sum(1) + 3.0

    result = potential_derivatives(
        potential, positions, compute_laplacian=compute_laplacian
    )
    _check_result(
        result,
        positions,
        coefficient.expand_as(positions),
        positions.new_zeros(2) if compute_laplacian else None,
    )
    assert parameter.grad is None


@pytest.mark.parametrize("compute_laplacian", [False, True])
@pytest.mark.parametrize(
    "kind", ["constant", "detached", "unrelated_leaf", "parameter_only"]
)
def test_disconnected_potentials_are_rejected(kind, compute_laplacian):
    positions = torch.ones(2, 3, dtype=torch.float64)
    parameter = torch.nn.Parameter(positions.new_tensor(2.0))

    def potential(x):
        if kind == "constant":
            return x.new_full((x.shape[0],), 2.0)
        if kind == "parameter_only":
            return parameter.expand(x.shape[0])
        values = x.square().sum(1).detach()
        return values.requires_grad_(kind == "unrelated_leaf")

    with pytest.raises(ValueError, match="spatial autograd graph"):
        potential_derivatives(potential, positions, compute_laplacian=compute_laplacian)
    assert parameter.grad is None
    assert not positions.requires_grad


@pytest.mark.parametrize("shape", [(), (3,), (0, 2), (2, 0), (2, 1, 0)])
def test_invalid_position_shape(shape):
    with pytest.raises(ValueError, match="positions.*shape"):
        potential_derivatives(lambda x: x.square().sum(1), torch.zeros(shape))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
def test_invalid_position_dtype(dtype):
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        potential_derivatives(
            lambda x: x.square().sum(1), torch.ones(2, 3, dtype=dtype)
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_positions_are_rejected(value):
    with pytest.raises(ValueError, match="positions.*finite"):
        potential_derivatives(lambda x: x.square().sum(1), torch.tensor([[value]]))


@pytest.mark.parametrize(
    ("potential", "error", "message"),
    [
        (lambda x: [1.0, 2.0], TypeError, "tensor"),
        (lambda x: x.sum(), ValueError, "shape"),
        (lambda x: x.sum(1, keepdim=True), ValueError, "shape"),
        (lambda x: x.sum(1)[:1], ValueError, "shape"),
        (lambda x: x.sum(1).double(), ValueError, "dtype and device"),
        (lambda x: x.sum(1).long(), ValueError, "dtype and device"),
        (
            lambda x: torch.empty(x.shape[0], device="meta"),
            ValueError,
            "dtype and device",
        ),
        (lambda x: x.sum(1) * float("inf"), ValueError, "potential.*finite"),
        (lambda x: x.sum(1) * float("nan"), ValueError, "potential.*finite"),
    ],
)
def test_invalid_potential_results(potential, error, message):
    with pytest.raises(error, match=message):
        potential_derivatives(potential, torch.ones(2, 3, dtype=torch.float32))


@pytest.mark.parametrize(
    "mismatch",
    ["observation_shape", "position_shape", "matrix_dtype", "observation_dtype"],
)
def test_analytic_path_keeps_likelihood_validation(mismatch):
    matrix = torch.eye(2, dtype=torch.float32)
    observation = torch.zeros(2, dtype=torch.float32)
    positions = torch.ones(3, 2, dtype=torch.float32)
    if mismatch == "observation_shape":
        observation = observation.unsqueeze(0)
    elif mismatch == "position_shape":
        positions = positions.unsqueeze(1)
    elif mismatch == "matrix_dtype":
        matrix = matrix.double()
    else:
        observation = observation.double()
    potential = GaussianLikelihood(DenseLinearOperator(matrix), observation, 1.0)
    with pytest.raises(ValueError, match="shape|dtype and device"):
        potential_derivatives(potential, positions)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("context", [nullcontext, torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("compute_laplacian", [False, True])
def test_ambient_modes_and_autocast_are_restored(
    dtype, context, analytic, compute_laplacian
):
    matrix = torch.tensor([[0.23, -0.47], [1.31, 0.79]], dtype=dtype)
    observation = matrix.new_tensor([0.13, -0.29])
    likelihood = GaussianLikelihood(DenseLinearOperator(matrix), observation, 0.7)
    inputs = []

    def generic(x):
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        assert not torch.is_autocast_cpu_enabled()
        assert not x.is_inference()
        inputs.append(x)
        return likelihood(x)

    with context(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        positions = matrix.new_tensor([[0.19, -0.37], [-0.53, 0.71]])
        saved = positions.clone()
        grad_enabled = torch.is_grad_enabled()
        inference_enabled = torch.is_inference_mode_enabled()
        result = potential_derivatives(
            likelihood if analytic else generic,
            positions,
            compute_laplacian=compute_laplacian,
        )
        assert torch.is_grad_enabled() is grad_enabled
        assert torch.is_inference_mode_enabled() is inference_enabled
        assert torch.is_autocast_cpu_enabled()
        assert_close(positions, saved)
        assert not positions.requires_grad
    if not analytic:
        assert len(inputs) == 1
        assert inputs[0].data_ptr() != positions.data_ptr()
    gradient = (positions @ matrix.T - observation) @ matrix / 0.7**2
    laplacian = (matrix.square().sum() / 0.7**2).expand(2)
    _check_result(result, positions, gradient, laplacian if compute_laplacian else None)


@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("existing_grad", [False, True])
@pytest.mark.parametrize("leaf_input", [False, True])
def test_inputs_parameters_and_existing_graphs_are_untouched(
    analytic, existing_grad, leaf_input
):
    leaf = torch.tensor(
        [[0.2, -0.4], [0.7, 1.2]], dtype=torch.float64, requires_grad=True
    )
    positions = leaf if leaf_input else 2 * leaf
    if not leaf_input:
        positions.retain_grad()
    matrix = torch.nn.Parameter(leaf.new_tensor([[1.0, 0.3], [-0.5, 2.0]]))
    observation = torch.nn.Parameter(leaf.new_tensor([0.1, -0.2]))
    likelihood = GaussianLikelihood(DenseLinearOperator(matrix), observation, 0.8)
    tensors = [leaf, matrix, observation] + ([] if leaf_input else [positions])
    for tensor in tensors:
        if existing_grad:
            tensor.grad = torch.full_like(tensor, 7.0)
    snapshots = [
        (tensor.detach().clone(), tensor.grad, tensor._version, tensor.grad_fn)
        for tensor in tensors
    ]
    original_values = likelihood(positions)
    potential = likelihood if analytic else lambda x: likelihood(x)
    result = potential_derivatives(potential, positions)
    for tensor, (value, grad, version, grad_fn) in zip(tensors, snapshots):
        assert_close(tensor, value)
        assert tensor.grad is grad
        if grad is not None:
            assert_close(grad, torch.full_like(grad, 7.0))
        assert tensor._version == version
        assert tensor.grad_fn is grad_fn
        assert tensor.requires_grad
    _check_result(
        result,
        positions,
        (positions @ matrix.T - observation) @ matrix / 0.8**2,
        (matrix.square().sum() / 0.8**2).expand(2),
    )
    # The caller's pre-existing graph has not been traversed or freed.
    torch.autograd.grad(original_values.sum(), (leaf, matrix, observation))


@pytest.mark.parametrize("compute_laplacian", [False, True])
def test_finite_values_with_nonfinite_gradient_are_rejected(compute_laplacian):
    with pytest.raises(ValueError, match="gradient.*finite"):
        potential_derivatives(
            lambda x: x.sqrt().sum(1),
            torch.zeros(2, 1, dtype=torch.float64),
            compute_laplacian=compute_laplacian,
        )


def test_nonfinite_laplacian_is_only_evaluated_when_requested():
    positions = torch.zeros(2, 1, dtype=torch.float64)

    def potential(x):
        return x.pow(1.5).sum(1)

    result = potential_derivatives(potential, positions, compute_laplacian=False)
    _check_result(result, positions, torch.zeros_like(positions), None)
    with pytest.raises(ValueError, match="laplacian.*finite"):
        potential_derivatives(potential, positions)


@pytest.mark.parametrize(
    ("matrix", "observation", "position", "message"),
    [
        (0.0, 1e30, 0.0, "potential.*finite"),
        (1e30, 0.0, 1e-11, "gradient.*finite"),
        (1e30, 0.0, 0.0, "laplacian.*finite"),
    ],
)
def test_analytic_nonfinite_values_and_derivatives(
    matrix, observation, position, message
):
    positions = torch.tensor([[position]], dtype=torch.float32)
    potential = GaussianLikelihood(
        DenseLinearOperator(positions.new_tensor([[matrix]])),
        positions.new_tensor([observation]),
        1.0,
    )
    with pytest.raises(ValueError, match=message):
        potential_derivatives(potential, positions)
    if message.startswith("laplacian"):
        result = potential_derivatives(potential, positions, compute_laplacian=False)
        _check_result(result, positions, torch.zeros_like(positions), None)


@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_device_is_preserved(device, analytic):
    positions = torch.tensor(
        [[0.2, -0.4], [0.7, 1.2]], dtype=torch.float32, device=device
    )
    potential = GaussianLikelihood(
        DenseLinearOperator(
            torch.eye(2, dtype=positions.dtype, device=positions.device)
        ),
        positions.new_zeros(2),
        1.0,
    )
    if not analytic:
        result = potential_derivatives(lambda x: x.square().sum(1) / 2, positions)
    else:
        result = potential_derivatives(potential, positions)
    _check_result(result, positions, positions, positions.new_full((2,), 2.0))


def test_result_is_a_frozen_dataclass():
    result = potential_derivatives(lambda x: x.square().sum(1), torch.ones(2, 1))
    with pytest.raises(FrozenInstanceError):
        result.gradient = torch.zeros(2, 1)
    with pytest.raises(FrozenInstanceError):
        result.laplacian = None
