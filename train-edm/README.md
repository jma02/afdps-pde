# Unconditional EDM

A compact, independently written training baseline: residual U-Net, EDM
preconditioning/loss, Adam, EMA, resumable checkpoints, Heun generation, and an
AFDPS score adapter. No trainer/controller classes, downloads, or vendored code.
Neural layers use `nn.Module`; orchestration is functional. Shared implementation
helpers have at least two distinct production callers.

## Install and data

From the repository root:

```bash
python -m pip install -e '.[train,dev]'
train-edm --help  # Also: python -m train_edm --help
```

`--data` accepts either:

- An image folder, searched recursively for PNG/JPEG/BMP/WebP. Images are
  EXIF-oriented, converted to grayscale or RGB, center-cropped/resized to
  `--size`, and scaled from `[0,255]` to `[-1,1]`. Class-folder names are ignored.
  Only the current minibatch is decoded; Pillow is the sole extra dependency.
- A `.pt` file containing one **float32 NCHW tensor** already normalized to
  `[-1,1]`, with nonzero N and matching size/channels. No resizing or silent
  normalization is applied. This option loads the tensor into CPU memory;
  it also supports channel counts other than 1 or 3, e.g. scientific fields.

Only load trusted tensor/checkpoint files. We use `weights_only=True`, not
pickled model objects, but that is not a security guarantee for old PyTorch.
No NumPy/torchvision interop is required. The existing local Torch/NumPy ABI
warning is unrelated to these tensor operations.

## Train, resume, evaluate, generate

```bash
train-edm train --data data/train --output runs/edm --steps 100000 \
  --size 32 --channels 3 --width 32 --multipliers 1 2 2 --device cuda

# Steps is the TOTAL target, not additional steps. All hyperparameters are restored.
train-edm resume --data data/train --output runs/edm --steps 200000 --device cuda

# Use a separately held-out folder/tensor. This estimates EDM loss, not FID.
train-edm evaluate --checkpoint runs/edm/checkpoint.pt --data data/validation \
  --batches 16 --batch-size 64 --device cuda

train-edm sample --checkpoint runs/edm/checkpoint.pt --output runs/edm/samples \
  --count 16 --steps 18 --seed 7 --device cuda
```

CPU is supported; CUDA is selected by default when available, otherwise CPU.
MPS can be requested explicitly with `--device mps`. Model width/multipliers,
`--sigma-data`, batch size, learning rate, noise distribution, and EMA half-life
are configurable (`train-edm train --help`). Size must be divisible by
`2**(number_of_multipliers-1)`. Bottleneck spatial attention is quadratic in its
pixel count: downsample enough for large images. Training is float32, single
process/device, without AMP, DDP, augmentation, or a data-worker pool.

Training writes `checkpoint.pt` and scalar `metrics.jsonl` at step 1, every
`--save-every` steps (default 1000), and at completion. The checkpoint includes
model/EMA/Adam state, model/training config, step, both RNG streams, and dataset
metadata. Writes use a temporary file plus atomic replacement. Starting a new
run over an existing checkpoint is rejected; use `resume` explicitly.

Minibatches are sampled uniformly **with replacement**, not by epochs. Separate
CPU data-index and same-device noise generators make split/resumed CPU runs
identical to uninterrupted runs in the tested environment. Resume rejects
changed model/training config, dataset paths/sizes/mtimes, or device type.
Dataset metadata is not a content checksum; don't edit data in place. Exact
bitwise equivalence across hardware or PyTorch versions is not promised.

Generation loads EMA by default. `samples.pt` contains raw, unclipped float
samples. PNG previews are separately clipped/scaled to `[0,255]` for 1/3 channels.
Existing sample output directories are rejected. Validation preserves model mode
and global RNG; its standard log-normal noise distribution can be overridden
with `--p-mean/--p-std` when comparing non-default training objectives.

### Quick CPU smoke run

This tests mechanics, **not** useful image quality:

```bash
mkdir -p data
python - <<'PY'
import torch
rng = torch.Generator().manual_seed(7)
torch.save(2 * torch.rand(16, 1, 8, 8, generator=rng) - 1, 'data/edm-smoke.pt')
PY
train-edm train --data data/edm-smoke.pt --output runs/edm-smoke --steps 2 \
  --size 8 --channels 1 --width 4 --multipliers 1 2 --batch-size 2 --device cpu
train-edm resume --data data/edm-smoke.pt --output runs/edm-smoke --steps 3 --device cpu
train-edm evaluate --data data/edm-smoke.pt --checkpoint runs/edm-smoke/checkpoint.pt \
  --batches 1 --batch-size 2 --device cpu
train-edm sample --checkpoint runs/edm-smoke/checkpoint.pt \
  --output runs/edm-smoke/samples --count 2 --steps 3 --sigma-max 1 --device cpu
```

## Equations and scope

For raw network `F`, data scale `s`, and noise level `sigma > 0`:

```text
D(x,sigma) = c_skip*x + c_out*F(c_in*x, log(sigma)/4)
c_skip = s²/(sigma²+s²)
c_out  = sigma*s/sqrt(sigma²+s²)
c_in   = 1/sqrt(sigma²+s²)
log(sigma) ~ Normal(P_mean=-1.2, P_std=1.2)
L = mean[(1/sigma² + 1/s²) * (D(y+sigma*noise,sigma)-y)²]
```

EMA uses `beta = 0.5**(batch_size / half_life_images)`, with half-life ramped
up as `min(configured_half_life, images_seen*ema_rampup)`. Gradients are checked
for finiteness, not silently repaired or clipped. The constant learning rate,
small U-Net, and defaults are baseline choices, not NVIDIA benchmark presets.

Heun generation uses the rho=7 power-law grid, defaults `[80, 0.002]`, and a
final Euler step to zero; the network is never evaluated at zero. Stochastic
churn is disabled. Computation uses the model dtype (normally float32), unlike
the reference sampler's float64 state. A Gaussian start at finite sigma_max and
finite solver steps are approximations. No FID, pretrained weights, or
architecture compatibility with NVIDIA EDM checkpoints is claimed.

## AFDPS adapter

```python
from functools import partial
from train_edm.sampling import EDMSchedule, edm_score
from train_edm.training import load_model

model = load_model("runs/edm/checkpoint.pt", device="cpu")  # Frozen, eval, EMA.
schedule = EDMSchedule(sigma_min=0.002, sigma_max=80.0)
score = partial(edm_score, model, schedule=schedule)
time_grid = schedule.time_grid(num_steps=200)
# Pass score, schedule, time_grid, and YOUR conditioned ensemble to afdps.sample.
```

The adapter computes `(D(x,sigma(t))-x)/sigma(t)²` on increasing reverse time:
`sigma(t) = (1-t/T)*sigma_max + (t/T)*sigma_min`, forward drift `F=0`, and
`G(T-t)² = 2*sigma(t)*(sigma_max-sigma_min)/T`. G is not sigma. Its terminal
prior is **p_sigma_min**, not the noiseless data law; it does not silently clip
time, evaluate a singular zero-noise score, or denoise the final AFDPS ensemble.

This is not a general learned-prior Stage I. Unconditional samples are **not**
likelihood-conditioned initial particles. You must provide a compatible Stage-I
law, NCHW state shape, normalization, dtype/device, and fixed likelihood. The
existing dense Gaussian initializer returns flat states, and its Gaussian noisy
reference is only an approximation for arbitrary learned priors. Generic exact
likelihood Laplacians also remain unsuitable for large images. Learned score and
initialization error do not disappear by using the AFDPS adapter.

## References and tests

Equations were checked against [Karras et al., EDM](https://arxiv.org/abs/2206.00364)
and NVIDIA's [preconditioning](https://github.com/NVlabs/edm/blob/main/training/networks.py),
[loss](https://github.com/NVlabs/edm/blob/main/training/loss.py), and
[sampler](https://github.com/NVlabs/edm/blob/main/generate.py). The implementation
is independently authored rather than copied/vendored. Reference repositories
retain their own licenses; no repository-wide license is inferred here.

```bash
python -m pytest -q train-edm/tests
python -m ruff check src tests examples train-edm
python -m ruff format --check src tests examples train-edm
```

Tests independently check coefficients/loss, gradients and particle independence,
Heun ordering, clock/score conversion, trainability, EMA, exact CPU resume,
image loading, global RNG isolation, checkpoint validation, and the four CLI
commands. A smoke-trained checkpoint is not evidence of generative quality.
