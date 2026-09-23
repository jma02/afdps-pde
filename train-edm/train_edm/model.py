"""Small, independently authored unconditional residual U-Net (not NVIDIA code).

EDM equations: https://arxiv.org/abs/2206.00364 (Table 1), cross-checked with
https://raw.githubusercontent.com/NVlabs/edm/main/training/loss.py (EDMLoss) and
https://raw.githubusercontent.com/NVlabs/edm/main/training/networks.py (EDMPrecond).
No architecture/checkpoint compatibility is claimed. This full-precision baseline
has no dropout, class labels, augmentation, AMP, or distributed-training machinery.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from afdps._validation import (
    check_count,
    check_generator,
    check_positions,
    check_tensor,
    finite_scalar,
    full_precision,
)


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    channels: int = 3
    width: int = 32
    multipliers: tuple[int, ...] = (1, 2, 2)
    sigma_data: float = 0.5

    def __post_init__(self) -> None:
        for name in ("image_size", "channels", "width"):
            check_count(getattr(self, name), name)
        if self.width < 2:
            raise ValueError("width must be at least 2")
        if not isinstance(self.multipliers, tuple) or not self.multipliers:
            raise ValueError("multipliers must be a nonempty tuple")
        for multiplier in self.multipliers:
            check_count(multiplier, "multiplier")
        if self.image_size % 2 ** (len(self.multipliers) - 1):
            raise ValueError("image_size must be divisible by 2**(len(multipliers)-1)")
        scale = finite_scalar(self.sigma_data, "sigma_data")
        if scale <= 0:
            raise ValueError("sigma_data must be positive")
        object.__setattr__(self, "sigma_data", scale)


def _check_images(model: nn.Module, x: Tensor) -> None:
    check_positions(x)
    config = model.config
    shape = (config.channels, config.image_size, config.image_size)
    if x.ndim != 4 or x.shape[1:] != shape:
        raise ValueError("images must be NCHW matching config.channels and image_size")
    parameter = next(model.parameters(), None)
    if parameter is not None and (
        x.dtype != parameter.dtype or x.device != parameter.device
    ):
        raise ValueError("images and model must share dtype and device")


class _ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, embedding: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, inputs)
        self.norm2 = nn.GroupNorm(1, outputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.noise = nn.Linear(embedding, outputs)
        self.skip = (
            nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)
        )

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h) + self.noise(embedding)[:, :, None, None]
        return (self.skip(x) + self.conv2(F.silu(h))) / math.sqrt(2)


class _Attention(nn.Module):
    """Single-head spatial attention; neither normalization nor attention mixes N."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv(self.norm(x)).flatten(2).chunk(3, dim=1)
        weights = (q.transpose(1, 2) @ k / math.sqrt(x.shape[1])).softmax(dim=-1)
        h = (v @ weights.transpose(1, 2)).reshape_as(x)
        return (x + self.output(h)) / math.sqrt(2)


class UNet(nn.Module):
    """Raw F(x, noise_labels), with labels shaped (N,) in the input dtype/device.

    One encoder/decoder block per resolution, concatenated skips, and bottleneck
    attention. Width >= 2 keeps GroupNorm valid even for a single 1x1 image.
    Construction uses PyTorch's RNG; the caller owns deterministic initialization.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        widths = [config.width * multiplier for multiplier in config.multipliers]
        embedding = 4 * config.width
        self.register_buffer("frequencies", torch.logspace(0, 3, config.width))
        self.embedding = nn.Sequential(
            nn.Linear(2 * config.width, embedding),
            nn.SiLU(),
            nn.Linear(embedding, embedding),
            nn.SiLU(),
        )
        self.input = nn.Conv2d(config.channels, widths[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        previous = widths[0]
        for width in widths:
            self.encoder.append(_ResidualBlock(previous, width, embedding))
            previous = width
        self.middle1 = _ResidualBlock(previous, previous, embedding)
        self.attention = _Attention(previous)
        self.middle2 = _ResidualBlock(previous, previous, embedding)
        self.decoder = nn.ModuleList()
        for width in reversed(widths):
            self.decoder.append(_ResidualBlock(previous + width, width, embedding))
            previous = width
        self.output = nn.Sequential(
            nn.GroupNorm(1, previous),
            nn.SiLU(),
            nn.Conv2d(previous, config.channels, 3, padding=1),
        )

    def forward(self, x: Tensor, noise_labels: Tensor) -> Tensor:
        _check_images(self, x)
        check_tensor(noise_labels, x, "noise_labels", shape=(x.shape[0],))
        with full_precision(x.device):
            phases = noise_labels[:, None] * self.frequencies[None, :]
            embedding = self.embedding(torch.cat((phases.sin(), phases.cos()), dim=1))
            h, skips = self.input(x), []
            for index, block in enumerate(self.encoder):
                if index:
                    h = F.avg_pool2d(h, 2)
                h = block(h, embedding)
                skips.append(h)
            h = self.middle2(self.attention(self.middle1(h, embedding)), embedding)
            for block in self.decoder:
                skip = skips.pop()
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = block(torch.cat((h, skip), dim=1), embedding)
            return self.output(h)


def denoise(model: nn.Module, x: Tensor, sigma: Union[float, Tensor]) -> Tensor:
    """EDM D(x,sigma); sigma is scalar, (N,), or (N,1,1,1), including N=1.

    Tensor sigma must match x's float32/float64 dtype and device. No input casts,
    moves, gradient detaches, or model-mode changes are performed.
    """
    _check_images(model, x)
    if not isinstance(sigma, Tensor):
        sigma = x.new_tensor(finite_scalar(sigma, "sigma"))
    n = x.shape[0]
    if sigma.shape not in ((), (1,), (n,), (1, 1, 1, 1), (n, 1, 1, 1)):
        raise ValueError("sigma must be scalar, (N,), or (N,1,1,1)")
    check_tensor(sigma, x, "sigma", shape=tuple(sigma.shape))
    if (sigma <= 0).any():
        raise ValueError("sigma must be positive and finite")
    with full_precision(x.device):
        sigma = sigma.reshape(-1, 1, 1, 1)
        sd = x.new_tensor(model.config.sigma_data)
        if not torch.isfinite(sd) or sd <= 0:
            raise ValueError("sigma_data must be positive/finite in the input dtype")
        # hypot avoids overflow from squaring a large, otherwise finite sigma.
        scale = torch.hypot(sigma, sd)
        raw = model(x / scale, (sigma.log() / 4).flatten().expand(n))
        check_tensor(raw, x, "raw model output")
        return (sd / scale).square() * x + (sigma / scale) * sd * raw


def edm_loss(
    model: nn.Module,
    clean: Tensor,
    *,
    generator: torch.Generator,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> Tensor:
    """Mean lognormal-noise EDM loss; an explicit same-device RNG is mandatory.

    p_std may be zero. Baseline training uses float32, without autocast; float64
    is also preserved. Nonfinite/unrepresentable sampled noise levels are rejected.
    """
    _check_images(model, clean)
    check_generator(generator, clean.device)
    p_mean, p_std = finite_scalar(p_mean, "p_mean"), finite_scalar(p_std, "p_std")
    if p_std < 0:
        raise ValueError("p_std must be nonnegative")
    with full_precision(clean.device):
        options = {"generator": generator, "device": clean.device, "dtype": clean.dtype}
        log_sigma = torch.randn((clean.shape[0], 1, 1, 1), **options) * p_std + p_mean
        sigma = log_sigma.exp()
        noise = torch.randn(clean.shape, **options)
        prediction = denoise(model, clean + sigma * noise, sigma)
        weight = sigma.reciprocal().square() + model.config.sigma_data**-2
        return (weight * (prediction - clean).square()).mean()
