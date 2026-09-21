import pytest
import torch
from torch.testing import assert_close

from afdps import ParticleEnsemble


def test_uniform_weights_ess_and_image_shaped_mean():
    positions = torch.arange(24, dtype=torch.float64).reshape(3, 1, 2, 4)
    ensemble = ParticleEnsemble.uniform(positions)
    assert ensemble.num_particles == 3
    assert_close(ensemble.weights, torch.full((3,), 1 / 3, dtype=positions.dtype))
    assert_close(
        ensemble.effective_sample_size, torch.tensor(3.0, dtype=positions.dtype)
    )
    assert_close(ensemble.effective_sample_size_fraction, positions.new_tensor(1.0))
    assert_close(ensemble.mean(), positions.mean(dim=0))


def test_nonuniform_weights_and_mean():
    positions = torch.tensor([[0.0], [4.0]], dtype=torch.float64)
    weights = positions.new_tensor([0.25, 0.75])
    ensemble = ParticleEnsemble(positions, weights.log())
    assert_close(ensemble.weights, weights)
    assert_close(ensemble.mean(), positions.new_tensor([3.0]))
    assert_close(ensemble.effective_sample_size, positions.new_tensor(1.6))
    assert_close(ensemble.effective_sample_size_fraction, positions.new_tensor(0.8))


def test_normalization_is_stable_for_extreme_offsets_and_zero_mass():
    positions = torch.zeros(3, 1, dtype=torch.float64)
    ensemble = ParticleEnsemble(
        positions, positions.new_tensor([1e20, 1e20, -torch.inf])
    )
    assert_close(ensemble.weights, positions.new_tensor([0.5, 0.5, 0.0]))
    assert_close(ensemble.weights.sum(), positions.new_tensor(1.0))
    assert torch.isneginf(ensemble.normalized_log_weights[-1])


def test_resampling_is_seeded_retains_ancestry_and_does_not_mutate_input():
    positions = torch.arange(24, dtype=torch.float64).reshape(12, 2)
    log_weights = torch.linspace(-3, 0, 12, dtype=positions.dtype)
    ensemble = ParticleEnsemble(positions, log_weights)
    global_state = torch.random.get_rng_state().clone()
    first, first_indices = ensemble.multinomial_resample(
        generator=torch.Generator().manual_seed(42)
    )
    second, second_indices = ensemble.multinomial_resample(
        generator=torch.Generator().manual_seed(42)
    )
    assert_close(first_indices, second_indices)
    assert_close(first.positions, second.positions)
    assert_close(first.positions, positions[first_indices])
    assert_close(first.log_weights, torch.zeros_like(log_weights))
    assert first.positions.dtype == positions.dtype
    assert first.positions.device == positions.device
    assert_close(ensemble.log_weights, log_weights)
    assert_close(torch.random.get_rng_state(), global_state)
    first.positions.zero_()
    assert_close(ensemble.positions, positions)
    assert ensemble.positions.sum() > 0


def test_resampling_replaces_and_never_selects_zero_mass():
    positions = torch.tensor([[0.0], [1.0], [2.0]])
    ensemble = ParticleEnsemble(positions, torch.tensor([-torch.inf, 0.0, -torch.inf]))
    result, ancestors = ensemble.multinomial_resample(
        generator=torch.Generator().manual_seed(0)
    )
    assert_close(ancestors, torch.ones(3, dtype=torch.long))
    assert_close(result.positions, torch.ones(3, 1))
    assert_close(result.effective_sample_size, torch.tensor(3.0))


@pytest.mark.parametrize(
    "log_weights",
    [
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [-float("inf"), -float("inf")],
    ],
)
def test_invalid_weights_are_rejected(log_weights):
    with pytest.raises(ValueError):
        ParticleEnsemble(torch.zeros(2, 1), torch.tensor(log_weights))


@pytest.mark.parametrize(
    "positions",
    [torch.tensor(1.0), torch.zeros(2), torch.empty(0, 2), torch.empty(2, 0)],
)
def test_invalid_particle_shapes_are_rejected(positions):
    with pytest.raises(ValueError):
        ParticleEnsemble.uniform(positions)


def test_incompatible_weights_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        ParticleEnsemble(torch.zeros(2, 1), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="dtype and device"):
        ParticleEnsemble(torch.zeros(2, 1), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype and device"):
        ParticleEnsemble(torch.zeros(2, 1), torch.zeros(2, device="meta"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_ensembles_are_rejected_before_ess_can_underflow(dtype):
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        ParticleEnsemble.uniform(torch.zeros(8192, 1, dtype=dtype))


def test_nonfloating_and_nonfinite_positions_are_rejected():
    with pytest.raises(TypeError):
        ParticleEnsemble.uniform(torch.ones(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        ParticleEnsemble.uniform(torch.tensor([[float("nan")]]))
