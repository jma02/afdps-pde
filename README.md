# afdps-pde

A typed PyTorch research implementation of **Approximation-Free Diffusion
Posterior Sampling (AFDPS)** from Chen et al.,
[arXiv:2506.03979v2](https://arxiv.org/html/2506.03979v2).

Implemented: exact dense linear-Gaussian Stage I, SDE Euler–Maruyama and ODE+ULA
Stage II, exact likelihood derivatives, stable log weights, ESS resampling,
custom grids, and scalar diagnostics. Analytic Gaussian examples exercise both
stages. This is **not** a reproduction of the pretrained-image experiments:
finite particles, Euler integration, and ULA introduce numerical error.

## Setup

Python 3.9+ is supported; Python 3.11/3.12 is convenient for a fresh environment.
Install an appropriate PyTorch build first if you need CUDA.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python examples/linear_gaussian_posterior.py --variant sde
python examples/linear_gaussian_posterior.py --variant ode
python -m ruff check src/afdps tests examples
python -m ruff format --check src/afdps tests examples
```

The existing local `venv` combines PyTorch 2.2.2 with NumPy 2.0.2 and emits an ABI
warning. The package does not use NumPy; use `numpy<2` with that PyTorch build or
upgrade PyTorch in a compatible environment. Dependencies are not changed
automatically. CUDA tests skip when unavailable; Stage II also tests MPS float32.
Stage I requires a supported Cholesky backend. No implicit CPU fallback occurs.

## Functional API

```python
from functools import partial
import torch
from afdps import (
    BrownianSchedule, DenseLinearOperator, GaussianLikelihood, SamplerConfig,
    condition_gaussian, sample_gaussian, sample,
)

schedule = BrownianSchedule(terminal_time=0.5)
prior_variance = 0.5
A = torch.tensor([[1.0]], dtype=torch.float64)
likelihood = GaussianLikelihood(DenseLinearOperator(A), A.new_tensor([0.8]), 1.0)
gaussian = condition_gaussian(
    likelihood,
    reference_std=(prior_variance + schedule.noise_variance(0)) ** 0.5,
)

# Exact verification score, not a pretrained checkpoint adapter.
def score(x, t):
    return -x / (prior_variance + schedule.noise_variance(t))

result = sample(
    score, likelihood, schedule,
    initial=partial(sample_gaussian, gaussian),
    config=SamplerConfig(num_particles=2048, num_steps=200),
    generator=torch.Generator().manual_seed(7),
)
print(result.ensemble.mean())  # Weighted estimate, not a posterior draw
print(result.diagnostics[-1])  # ESS before any resampling
```

`condition_gaussian` snapshots the likelihood into a cached `Gaussian` record.
Its `.mean` and `.covariance` diagnostics return independent tensors. Reuse it
across `sample_gaussian(gaussian, count, generator=...)` calls without repeating
factorization. Recondition when the inverse problem changes.

`sample` accepts either an initializer callable or an existing `ParticleEnsemble`
as `initial`. The latter runs Stage II only, preserving its initial weights and
storage. Both start at t=0; this is not a mid-trajectory resume API. Optional
`time_grid` entries must be finite, strictly increasing, start at 0, end at the
schedule's horizon, and number `config.num_steps + 1`. Defaults are uniform.

For ODE+ULA, pass an explicit corrector configuration:

```python
from afdps import CorrectorConfig
config = SamplerConfig(
    variant="ode", num_particles=2048, num_steps=200,
    corrector=CorrectorConfig(num_steps=1, step_size=0.001),
)
```

These controls are not paper-prescribed hyperparameters. `num_steps=0` in the
corrector explicitly disables ULA; `resample_ess_fraction=None` disables
resampling but retains weights. SDE rejects a corrector configuration.

Use `functools.partial` to bind repeated arguments. Operational code is
functional; small configuration, problem, and result records retain validation
and useful value properties.
Implementation helpers need at least two distinct production callers; public
numerical operations remain independently testable functions.

## Contracts and limits

- One observation; positions `(N, *state_shape)`, potential values `(N,)`.
  Scores/operators/potentials must not couple particles, mutate inputs, or use
  hidden randomness. Set neural models to evaluation mode explicitly.
- Use float32/float64 and an explicit same-device `torch.Generator`. Sampling
  returns detached tensors; likelihood differentiation enables autograd locally.
- Reverse time increases from noisy 0 to clean T. Schedules return `F(T-t)` and
  `G(T-t)`; model noise conditioning is a separate adapter responsibility.
- The fixed potential `-log p(y|x)` is evaluated at current particles, not denoised
  predictions. Constants must retain autograd: `(0*x).flatten(1).sum(1) + c`.
- Stage I already conditions on the likelihood; do not weight it again.
  The initial law, score, schedule, and likelihood must agree.
- Generic exact Laplacians need one backward pass per state coordinate; this is
  a correctness baseline, not a scalable large-image implementation.
- ULA has finite-step bias; approximate learned scores need not preserve the
  model-induced posterior. Endpoint scores must be finite. No clipping, jitter,
  terminal denoising, forced resampling, or evidence estimation is hidden here.

Pretrained adapters, general/nonlinear Stage I, training, scalable derivative
estimators, physical PDE solvers, and experiment presets remain outside scope.
See [architecture and equations](docs/architecture.md) and
`examples/linear_gaussian_stage1.py` for an isolated Stage I check.
Tests live in `tests/`; all source code is in `src/afdps/`.
Downloads remain opt-in; no datasets or pretrained weights are bundled.
