# AFDPS implementation map

This document targets [arXiv:2506.03979v2](https://arxiv.org/html/2506.03979v2).
It maps the implemented linear-Gaussian Stage I initializer and both Stage II
variants to the paper, and distinguishes numerical algorithms from exact
continuous-time theory. No pretrained-image experiment reproduction is claimed.

## Mathematical boundary

For one observation `y`, the forward problem is `y = A(x) + measurement_noise`.
The potential is `mu_y(x) = -log p(y | x)`, up to a constant. AFDPS evolves the
family `q_hat_y(x,t) proportional to p_hat_reverse(x,t) * exp(-mu_y(x))`.
This is not the usual noisy conditional distribution `p_t(x_t | y)`.
The likelihood is evaluated on the **current state**, not a denoised estimate.

The algorithm has two separate stages ([Section 3.1](https://arxiv.org/html/2506.03979v2#S3.SS1)):

1. **Initialization:** sample the noisy Gaussian reference multiplied by the
   likelihood, `q_hat_y(x,0) proportional to N(x; 0, rho^2 I) exp(-mu_y(x))`.
   Unconditional Gaussian noise is generally not a valid initializer.
2. **Evolution:** transport, diffuse, and reweight an ensemble according to
   the posterior PDE, then resample when appropriate. A guided diffusion
   trajectory without its weight dynamics is not AFDPS.

## Module responsibilities

| Module | Responsibility | Current state |
| --- | --- | --- |
| `interfaces.py` | Score, operator, potential, schedule, Stage I contracts | Callable types and structural protocols |
| `config.py` | Particle count, step count, variant, fractional ESS threshold | Validated; defaults are not paper presets |
| `operators.py` | Forward mapping independent of the prior | Small dense linear example |
| `likelihoods.py` | Fixed-observation negative log-likelihood | Isotropic Gaussian potential; retains autograd |
| `initializers.py` | Stage I Gaussian reference conditioned on data | `condition_gaussian` prepares a record; `sample_gaussian` draws |
| `derivatives.py` | Potential gradient and Laplacian | Analytic dense Gaussian path and exact autograd fallback |
| `schedules.py` | Forward coefficients on the reverse clock | Brownian/linear-variance verification process |
| `particles.py` | Ensemble state and generic SMC bookkeeping | Stable weights, ESS, and multinomial resampling |
| `dynamics.py` | Eq. (3.6) coefficients and individual kernels | Canonical SDE/ODE Euler and ULA |
| `sampling.py` | Compose Stage I and Stage II components | Functional `sample`, weighted result and diagnostics |

Training and checkpoint loading are outside the inference core. Models will
be adapted to `ScoreFunction`; inverse problems supply `ForwardOperator` and
`Potential`. No sampler should need to import a particular U-Net, dataset,
physical PDE solver, or experiment runner.

## Array, derivative, and time conventions

- Positions have shape `(N, *state_shape)`, including `(N, d)` or `(N, C, H, W)`.
  There is no separate observation-batch axis in this first API.
- Operator output is `(N, *observation_shape)`; `y` is unbatched. Shape mismatch
  is an error, not an invitation to broadcast across observation axes.
- `mu_y` returns one scalar per particle. For isotropic Gaussian measurement
  noise, it is `sum((A(x) - y)^2) / (2 * sigma_obs^2)`; sum over all observation
  axes, never average over them or over particles.
- Operators, potentials, and scores act independently on each particle. The
  generic potential must preserve spatial derivatives through autograd. SDE
  weights require both `grad(mu_y)` and `laplacian(mu_y)`; the generic trace
  evaluator uses one backward pass per event coordinate, not Hutchinson probes.
  Detached/disconnected potentials are rejected rather than assigned a zero
  gradient; connected affine potentials have zero Laplacian. Express a constant
  potential as `(0*x).flatten(1).sum(1) + constant`. For a linear Gaussian likelihood,
  the Laplacian is the particle-independent `||A||_F^2 / sigma_obs^2` and can
  cancel from **relative** weights, but this does not generalize to nonlinear A.
- `reverse_time` increases from `0` (noisy initialization) to `T` (clean end).
  `DiffusionSchedule` returns `F(T-t)` and `G(T-t)` from the forward SDE in
  [Eq. (2.3)](https://arxiv.org/html/2506.03979v2#S2.SS2). It does not introduce
  the reverse-drift sign itself.
- A model adapter must separately convert the reverse clock to its checkpoint's
  conditioning parameter. Scores, denoised images, and epsilon predictions
  are not interchangeable. EDM noise level, Brownian diffusion amplitude,
  measurement standard deviation, and Stage I reference scale are distinct.
- The current numerical primitives require float32 or float64 and consistent
  dtype/device across states, observations, operators, and weights. Half
  precision is explicitly rejected: squared probabilities can underflow,
  producing infinite ESS even for a uniform ensemble. Future mixed-precision
  model adapters must preserve full-precision likelihood/weight arithmetic.
- Random operations take an explicit same-device `torch.Generator`. No global
  seed, dtype, or device is changed on import. Tensor dataclasses freeze field
  bindings, not the underlying tensor storage; callers own that storage.

## Implemented Stage I: linear-Gaussian conditioning

`condition_gaussian(likelihood, reference_std=rho)` prepares a `Gaussian` from a
`GaussianLikelihood` with `DenseLinearOperator`, one observation vector `(m,)`,
and matching float32/float64 dtype and device. For `A` shaped `(m, d)`,

```text
B = A / sigma_obs
c = y / sigma_obs
Lambda = B.T @ B + I / rho^2
L = cholesky(Lambda)             # Lambda = L @ L.T, a PRECISION factor
mean = cholesky_solve(B.T @ c, L)
z_i ~ N(0, I)
x_i = mean + solve(L.T, z_i)     # NOT solve(L, z_i)
```

The covariance is `Lambda^{-1} = L^{-T} L^{-1}`. Neither preparation nor
sampling constructs a matrix inverse. The diagnostic `covariance` property
solves against the identity on demand; `mean` returns a copy. The proper prior
ridge means underdetermined and rank-deficient matrices are valid. Extreme
conditioning can still erase that ridge in finite precision; the implementation
reports nonfinite parameters or failed factorizations without adding jitter.
Rescale the problem or use float64 rather than silently changing the target.
Preparation and solves disable ambient autocasting to retain the input floating
dtype. Covariance diagnostics also report overflow instead of returning infinity.

Preparation costs `O(m*d^2 + d^3)` time and retains `O(d^2)` factors.
`sample_gaussian(gaussian, N, generator=...)` costs `O(N*d^2)` and returns
`(N, d)` states with **zero log weights** because
they already follow the conditioned Stage I law, not an importance proposal.
No additional likelihood reweighting is appropriate. The prepared record caches
detached parameters: input mutations cannot change it, and returned diagnostic
tensors do not expose its cache. Repeated draws do not refactorize. All sampling uses
an explicit same-device generator; `None` is rejected rather than falling back
to global RNG state. This is an inference API, not a differentiable sampler.

Tests check closed-form means/covariances, the deterministic whitening identity
`(positions - mean) @ L = noise`, rank-deficient/nullspace cases, nonunit scales,
RNG isolation, and empirical moments over three fixed seeds. CUDA tests run
only when a CUDA device is available. Nonlinear/matrix-free operators, correlated
measurement noise, and nonzero/anisotropic Gaussian reference priors remain
outside this first initializer's scope.

## Implemented Stage II dynamics

Let `f = F(T-t)`, `g = G(T-t)`, `v = V(t)`, `s = phi_theta(x,t)`, and let the
prior reverse drift be `H = -f*x + (g^2 + v^2)/2 * s`. The particle formulation
in [Eq. (3.6), Section 3.2](https://arxiv.org/html/2506.03979v2#S3.SS2) requires

```text
position drift = H - v^2 * grad(mu_y)
Brownian amplitude = v
log-weight rate = v^2/2 * (||grad(mu_y)||^2 - laplacian(mu_y))
                  - dot(H, grad(mu_y))
```

A common centering term cancels when weights are normalized. The
particle-dependent log-weight rate does not cancel.

### SDE: Algorithm 2

For the canonical variant, `v = g`. Use Euler-Maruyama transport with noise
amplitude `v * sqrt(dt)`, update log weights, then perform ESS-triggered
resampling. Keep drift and weight-rate calculations independently testable.

### ODE plus corrector: Algorithms 3 and 4

The predictor sets `v = 0`, so its score multiplier is `g^2/2` and it contains
**no likelihood-gradient drift**. Its weights still evolve by `-H dot grad(mu_y)`.
An unadjusted Langevin corrector uses `score - grad(mu_y)` and Brownian noise
`sqrt(2 * corrector_step_size)` at the next time. The order is predictor,
corrector, then resampling; weight increments use the old state/time.
Finite-step ULA is not an exact invariant Markov kernel.

`CorrectorConfig(num_steps, step_size)` is mandatory for the ODE variant;
`step_size` is Langevin time and is not multiplied by the predictor's `dt`.
`num_steps=0` explicitly disables correction for weighted-ODE verification. The
SDE variant rejects corrector settings. These controls are not inferred from
Appendix D.2, which does not supply a ULA step size.

The ULA field uses the supplied `phi_theta - grad(mu_y)`. An approximate learned
score need not be the score of its own generated prior marginal; the corrector
therefore need not preserve the model-induced posterior, even apart from finite
step-size bias. No weight adjustment in this implementation removes those errors.
The canonical guidance coefficient is fixed at one; Appendix B's optional eta
family and time-dependent/tempered potentials are not implemented.

### Scheduling and the integration loop

`BrownianSchedule(T, G)` supplies the forward process `dx_s = G dW_s` with
`F=0`, constant `G`, and perturbation variance `G^2*(T-t)` on the reverse clock.
It is an explicit verification process, not a pretrained EDM/VP noise schedule.
For a clean Gaussian prior of variance `tau^2`, the exact reverse-clock score is
`-x / (tau^2 + G^2*(T-t))`, and Stage I must use
`rho^2 = tau^2 + G^2*T`, not just the perturbation variance. This closes the loop
in `examples/linear_gaussian_posterior.py` without mismatched initial marginals.

`sample(score, potential, schedule, initial, generator=..., config=...)` accepts
an initializer callable or a conditioned ensemble. Bind `sample_gaussian` to a
prepared record with `functools.partial` for both stages; pass an ensemble for
Stage II alone. There is no sampler object or mutable hidden state. Both paths
start at t=0 and return `SamplingResult(ensemble, diagnostics)`, preserving
initial weights instead of reapplying the likelihood. Components must be
compatible and deterministic; checkpoints, evaluation mode, and time-to-noise
adapters remain the caller's responsibility. Grid checks precede initialization
and RNG consumption. Configuration and results remain small frozen data records.

Default integration uses `num_steps` uniform intervals. A custom `time_grid`
must have `num_steps+1` finite real entries, start at zero, end at the schedule's
positive finite horizon, and be strictly increasing. Each step evaluates the
coefficients, score, and potential derivatives at the old state/time; ODE
corrector evaluations use the next time, including the final endpoint. No score
or noise-floor clamp, terminal denoising, adaptive stepping, or continuation from
an arbitrary intermediate distribution is performed.

Score evaluations and state evolution do not build autograd graphs; likelihood
derivatives enable autograd locally on cloned states, even under an outer
`no_grad`/`inference_mode` context. Kernels preserve dtype/device and disable
ambient autocast where supported. Nonfinite dynamics fail explicitly, not by
clipping or changing the target. Every controlled random operation requires a
same-device `torch.Generator`; `None` is rejected, including for resampling.

## Weight conventions and results

`ParticleEnsemble` stores unnormalized log weights, permits `-inf` for
zero-mass particles, rejects NaN/+inf/all-zero mass, and normalizes stably.

- `weights` sum to one.
- `effective_sample_size = 1 / sum(weights^2)` is a particle count.
- `effective_sample_size_fraction = ESS / N` is the quantity compared to `c`
  in Algorithm 1. Resampling uses strict `< resample_ess_fraction`; setting the
  threshold to `None` disables resampling without discarding weights.
- Multinomial resampling is with replacement. It returns ancestor indices and
  resets log weights to zero (the paper's mean-one beta convention).
- Without resampling, results remain weighted. A weighted mean or highest-weight
  particle is a reconstruction statistic, not an unweighted posterior draw.

[Appendix D.2](https://arxiv.org/html/2506.03979v2#A4.SS2) describes an experimental
shortcut that skips resampling and selects the highest-weight image. A future
reproduction runner should expose this explicitly rather than changing the
meaning of the core sampler. `StepDiagnostics` records the one-based step,
next reverse time, predictor step size, relative ESS **before resampling**, and
whether resampling occurred. Final weights stay nonuniform unless resampling
actually runs; there is no forced final resampling. This scalar history costs
O(number of steps), not storage of every particle trajectory.

Euler weight updates use `logw += dt * rate`, subtracting a common rate offset
for stability and normalizing with log-softmax. This is equivalent to the
centered reaction term for normalized weights; it is not additive Euler in
ordinary weights. Common offsets/log normalizers are discarded, so these
outputs must not be used as a marginal-likelihood/evidence estimator.

## Incremental implementation and validation

1. **Done: exact Stage I for a dense linear Gaussian problem.** Implemented
   with Cholesky solves and analytic/moment tests; see the section above and
   `examples/linear_gaussian_stage1.py`.
2. **Done: individual equations and derivatives.** Tests check nonzero forward
   drift signs, Brownian scaling, old-state weight rates, the prior (not guided)
   drift in the reaction term, and exact nonlinear/image-shaped potential traces.
3. **Done: SDE loop.** Tests cover log-shift stability, preserved initial weights,
   explicit grids, fractional-ESS boundaries/resets, diagnostics, and Gaussian
   posterior moments over fixed seeds without resampling.
4. **Done: ODE plus ULA.** Tests verify predictor/corrector/resampling order,
   fixed next-time ULA evaluations, correct noise amplitude, and convergence of
   the uncorrected ODE to an analytic Gaussian flow as the grid is refined.
   Finite-step ULA posterior checks allow numerical/statistical error.
5. **Done: a compact unconditional EDM baseline.** `train-edm/` provides a custom
   U-Net, EDM loss/preconditioning, optimizer/EMA/RNG checkpoints, Heun generation,
   and a score/schedule adapter for its own checkpoints. See its README for the
   positive-noise endpoint and learned-prior Stage-I limitations.
6. **Next: larger priors and applications.** Add third-party EDM/VP checkpoint
   adapters, scalable likelihood derivatives, matrix-free imaging/PDE operators,
   and experiment configs. Generic exact Hessian traces remain a correctness
   baseline, not a large-image implementation. Downloads remain opt-in.

## Interpretation and reproduction cautions

The paper overloads forward and reverse time notation. Schedule formulas must
be reconciled with the forward SDE and independently checked; do not blindly
put a decreasing noise schedule's derivative under a square root.

The displayed linear-Gaussian initialization density appears to omit a factor
of `1/2` that is required by its stated precision. Algorithms 2/4 appear to
reference next-time weights on the right side of their update. Resolve such
apparent typos against the continuous equations and analytic tests rather than
copying the pseudocode literally.

“Approximation-free” refers to avoiding heuristic conditional-score
approximations, not eliminating score-model, finite-particle, initialization,
or numerical discretization errors. [Section 4](https://arxiv.org/html/2506.03979v2#S4)
explicitly excludes discretization error from its analysis. The present tests
validate primitives, the exact linear-Gaussian initial law, the discretized
Stage II equations, and final Gaussian posterior moments. They do not establish
accuracy for arbitrary learned priors, ill-conditioned nonlinear problems, or
the paper's high-dimensional imaging experiments.
