"""Sample the exact Stage I Gaussian; this does not run AFDPS Stage II.

Run after `python -m pip install -e .`:
    python examples/linear_gaussian_stage1.py
"""

from __future__ import annotations

import torch

from afdps import DenseLinearOperator, GaussianLikelihood
from afdps.initializers import condition_gaussian, sample_gaussian


def main() -> None:
    matrix = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    likelihood = GaussianLikelihood(
        operator=DenseLinearOperator(matrix),
        observation=torch.tensor([1.0, 0.0], dtype=matrix.dtype),
        noise_std=1.0,
    )
    gaussian = condition_gaussian(likelihood, reference_std=1.0)
    ensemble = sample_gaussian(
        gaussian, 4096, generator=torch.Generator().manual_seed(7)
    )
    centered = ensemble.positions - ensemble.mean()
    sample_covariance = centered.T @ centered / (ensemble.num_particles - 1)

    print("Exact linear-Gaussian Stage I only; no Stage II dynamics have run.")
    print(f"analytic mean: {gaussian.mean.tolist()}")
    print(f"sample mean: {ensemble.mean().tolist()}")
    print(f"analytic covariance: {gaussian.covariance.tolist()}")
    print(f"sample covariance: {sample_covariance.tolist()}")
    print(f"uniform-weight ESS: {ensemble.effective_sample_size.item():.1f}")


if __name__ == "__main__":
    main()
