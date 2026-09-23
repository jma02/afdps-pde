"""Single-device, full-precision EDM training; explicit RNG and resumable state."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from afdps._validation import check_count, finite_scalar, full_precision
from train_edm.model import ModelConfig, UNet, edm_loss


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    lr: float = 2e-4
    seed: int = 0
    p_mean: float = -1.2
    p_std: float = 1.2
    ema_half_life: float = 500_000  # Images, not optimizer steps.
    ema_rampup: float = 0.05

    def __post_init__(self):
        check_count(self.batch_size, "batch_size")
        check_count(self.seed, "seed", allow_zero=True)
        if self.seed >= 2**64 - 1:
            raise ValueError("seed must be less than 2**64 - 1")
        for name in ("lr", "p_mean", "p_std", "ema_half_life", "ema_rampup"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), name))
        for name in ("lr", "ema_half_life", "ema_rampup"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.p_std < 0:
            raise ValueError("p_std must be nonnegative")


def _dataset(path, config):
    path = Path(path).resolve(strict=True)
    if path.is_file() and path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=True)
        shape = (config.channels, config.image_size, config.image_size)
        if not isinstance(data, torch.Tensor) or data.ndim != 4 or not len(data):
            raise ValueError("data must be a nonempty NCHW tensor")
        if data.shape[1:] != shape or data.dtype != torch.float32:
            raise ValueError(f"data must be float32 with event shape {shape}")
        if not torch.isfinite(data).all() or data.min() < -1 or data.max() > 1:
            raise ValueError("training data must be finite and normalized to [-1, 1]")
        data = data.detach()
        files = (path,)
    elif path.is_dir():
        if config.channels not in (1, 3):
            raise ValueError("image folders support 1 or 3 channels; use .pt otherwise")
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = tuple(
            sorted(
                p
                for p in path.rglob("*")
                if p.suffix.lower() in extensions and p.is_file()
            )
        )
        if not files:
            raise ValueError("image folder is empty")
        data = files
    else:
        raise ValueError("data must be an image folder or a .pt tensor")
    # Detect changed dataset ordering/content metadata before restoring RNG states.
    metadata = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    signature = hashlib.sha256(json.dumps(metadata).encode()).hexdigest()
    return data, signature


def _batch(data, config, count, generator):
    indices = torch.randint(len(data), (count,), device="cpu", generator=generator)
    if isinstance(data, torch.Tensor):
        return data[indices]
    from PIL import Image, ImageOps

    images = []
    for index in indices.tolist():
        with Image.open(data[index]) as image:
            image = ImageOps.exif_transpose(image).convert(
                "L" if config.channels == 1 else "RGB"
            )
            image = ImageOps.fit(
                image, (config.image_size,) * 2, method=Image.Resampling.LANCZOS
            )
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            images.append(
                pixels.reshape(
                    config.image_size, config.image_size, config.channels
                ).permute(2, 0, 1)
            )
    return torch.stack(images).float().div_(127.5).sub_(1)


def _model(config, device, seed):
    # Isolate initialization from the caller's RNG streams and default dtype.
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(
                torch.Generator(device="cpu").manual_seed(seed).get_state()
            )
            return UNet(config).to(device)
    finally:
        torch.set_default_dtype(dtype)


def _checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "model_config",
        "train_config",
        "model",
        "ema",
        "optimizer",
        "step",
        "data_rng",
        "noise_rng",
        "data_signature",
        "device_type",
    }
    if (
        not isinstance(state, dict)
        or state.get("format") != 1
        or not required <= state.keys()
    ):
        raise ValueError("unsupported or incomplete EDM checkpoint")
    check_count(state["step"], "checkpoint step", allow_zero=True)
    return state


def load_model(checkpoint, *, device="cpu", ema: bool = True) -> UNet:
    """Load this project's state-dict format, not NVIDIA pickle checkpoints."""
    state = _checkpoint(checkpoint)
    model = _model(ModelConfig(**state["model_config"]), device, seed=0)
    model.load_state_dict(state["ema" if ema else "model"], strict=True)
    return model.eval().requires_grad_(False)


def train(
    data,
    output,
    *,
    steps: int,
    model_config: Optional[ModelConfig] = None,
    config: Optional[TrainConfig] = None,
    device="cpu",
    resume: bool = False,
    save_every: int = 1000,
) -> Path:
    """Train to a TOTAL step count; resume inherits config and requires unchanged data.

    Minibatches are sampled with replacement, with no worker prefetch state to lose.
    Checkpoints are atomically replaced; only explicitly resumed runs may overwrite.
    """
    check_count(steps, "steps")
    check_count(save_every, "save_every")
    output = Path(output)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError("checkpoint already exists; use resume or a new output")
    state = _checkpoint(checkpoint) if resume else None
    if state is not None:
        if model_config is not None and asdict(model_config) != state["model_config"]:
            raise ValueError("resume must use the saved model configuration")
        if config is not None and asdict(config) != state["train_config"]:
            raise ValueError("resume must use the saved training configuration")
        model_config = ModelConfig(**state["model_config"])
        config = TrainConfig(**state["train_config"])
    model_config = ModelConfig() if model_config is None else model_config
    config = TrainConfig() if config is None else config
    dataset, signature = _dataset(data, model_config)
    start = 0 if state is None else state["step"]
    if steps <= start:
        raise ValueError("steps must exceed the saved step count")
    device = torch.device(device)
    if state is not None:
        if state["data_signature"] != signature:
            raise ValueError(
                "resume dataset changed (paths, sizes, or modification times)"
            )
        if state["device_type"] != device.type:
            raise ValueError(
                "resume requires the same device type; load_model allows transfers"
            )
    model = _model(model_config, device, config.seed)
    device = next(model.parameters()).device
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    data_rng = torch.Generator(device="cpu").manual_seed(config.seed)
    noise_rng = torch.Generator(device=device).manual_seed(config.seed + 1)
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        data_rng.set_state(state["data_rng"])
        noise_rng.set_state(state["noise_rng"])
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    with full_precision(device):
        for step in range(start + 1, steps + 1):
            clean = _batch(dataset, model_config, config.batch_size, data_rng).to(
                device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = edm_loss(
                model,
                clean,
                generator=noise_rng,
                p_mean=config.p_mean,
                p_std=config.p_std,
            )
            if not torch.isfinite(loss):
                raise ValueError(
                    "nonfinite training loss; check data and training controls"
                )
            loss.backward(gradient=loss.new_tensor(1.0))
            # Check gradients without silently clipping or replacing invalid values.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            half_life = min(
                config.ema_half_life, step * config.batch_size * config.ema_rampup
            )
            beta = 0.5 ** (config.batch_size / half_life)
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, 1 - beta)
                for average, current in zip(ema.buffers(), model.buffers()):
                    average.copy_(current)
            if step == 1 or step % save_every == 0 or step == steps:
                record = {
                    "step": step,
                    "images_seen": step * config.batch_size,
                    "loss": loss.item(),
                }
                state = {
                    "format": 1,
                    "model_config": asdict(model_config),
                    "train_config": asdict(config),
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "data_rng": data_rng.get_state(),
                    "noise_rng": noise_rng.get_state(),
                    "data_signature": signature,
                    "device_type": device.type,
                    "torch_version": str(torch.__version__),
                }
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(checkpoint)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
    return checkpoint


@torch.no_grad()
def evaluate(
    model, data, *, batches=16, batch_size=32, seed=0, p_mean=-1.2, p_std=1.2
) -> float:
    """Seeded held-out EDM loss estimate, NOT FID; preserve model mode and global RNG."""
    check_count(batches, "batches")
    check_count(batch_size, "batch_size")
    check_count(seed, "seed", allow_zero=True)
    if seed >= 2**64 - 1:
        raise ValueError("seed must be less than 2**64 - 1")
    dataset, _ = _dataset(data, model.config)
    device = next(model.parameters()).device
    data_rng = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = torch.Generator(device=device).manual_seed(seed + 1)
    training = model.training
    model.eval()
    total = 0.0
    try:
        for _ in range(batches):
            images = _batch(dataset, model.config, batch_size, data_rng).to(device)
            loss = edm_loss(
                model, images, generator=noise_rng, p_mean=p_mean, p_std=p_std
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite validation loss")
            total += loss.item()
    finally:
        model.train(training)
    return total / batches
