"""Run both AFDPS stages against a known Gaussian posterior, without checkpoints.

python examples/linear_gaussian_posterior.py --variant sde
python examples/linear_gaussian_posterior.py --variant ode
"""

from __future__ import annotations

import argparse
from functools import partial

import torch

from afdps import (
    BrownianSchedule,
    CorrectorConfig,
    DenseLinearOperator,
    GaussianLikelihood,
    SamplerConfig,
    condition_gaussian,
    sample,
    sample_gaussian,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sde", "ode"), default="sde")
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dtype = torch.float64
    schedule = BrownianSchedule(terminal_time=0.5, diffusion_std=1.0)
    prior_variance = 0.5
    likelihood = GaussianLikelihood(
        operator=DenseLinearOperator(
            torch.tensor([[1.0, 0.4], [-0.2, 0.7]], dtype=dtype)
        ),
        observation=torch.tensor([0.8, -0.4], dtype=dtype),
        noise_std=1.0,
    )
    # At t=0 the noisy Gaussian includes BOTH clean-prior variance and the
    # Brownian perturbation variance. Stage I conditions it on y exactly.
    gaussian = condition_gaussian(
        likelihood, reference_std=(prior_variance + schedule.noise_variance(0)) ** 0.5
    )
    target = condition_gaussian(likelihood, reference_std=prior_variance**0.5)

    def score(positions, reverse_time):
        return -positions / (prior_variance + schedule.noise_variance(reverse_time))

    result = sample(
        score=score,
        potential=likelihood,
        schedule=schedule,
        initial=partial(sample_gaussian, gaussian),
        generator=torch.Generator().manual_seed(args.seed),
        config=SamplerConfig(
            variant=args.variant,
            num_particles=args.particles,
            num_steps=args.steps,
            resample_ess_fraction=0.5,
            corrector=(
                CorrectorConfig(num_steps=1, step_size=0.001)
                if args.variant == "ode"
                else None
            ),
        ),
    )
    ensemble = result.ensemble
    mean = ensemble.mean()
    centered = ensemble.positions - mean
    covariance = centered.T @ (ensemble.weights[:, None] * centered)

    print(
        f"AFDPS-{args.variant.upper()}: finite-particle, first-order numerical sampling"
    )
    print(f"analytic mean: {target.mean.tolist()}")
    print(f"weighted mean: {mean.tolist()}")
    print(f"analytic covariance: {target.covariance.tolist()}")
    print(f"weighted covariance: {covariance.tolist()}")
    print(
        f"final ESS: {ensemble.effective_sample_size.item():.1f}/{ensemble.num_particles}"
    )
    print(f"resampling steps: {sum(step.resampled for step in result.diagnostics)}")
    print("This analytic benchmark is not a pretrained-image experiment reproduction.")


if __name__ == "__main__":
    main()
