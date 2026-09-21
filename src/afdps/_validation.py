"""Shared runtime checks for inference kernels; no casting or device moves."""

import math
from contextlib import AbstractContextManager, nullcontext
from numbers import Real
from typing import Optional

import torch
from torch import Tensor


def full_precision(device: torch.device) -> AbstractContextManager:
    """Disable autocast, including on older PyTorch without MPS autocast."""
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except RuntimeError as error:
        if device.type == "mps" and "unsupported autocast device_type" in str(error):
            return nullcontext()
        raise


def finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def check_count(value: int, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def check_generator(
    generator: torch.Generator, device: Optional[torch.device] = None
) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    if device is not None and generator.device != device:
        raise ValueError("generator and particles must share device")


def check_positions(positions: Tensor) -> None:
    if not isinstance(positions, Tensor):
        raise TypeError("positions must be a tensor")
    if positions.ndim < 2 or positions.numel() == 0:
        raise ValueError("positions must have nonempty shape (N, *state_shape)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must use torch.float32 or torch.float64")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must contain only finite values")


def check_tensor(
    value: Tensor,
    reference: Tensor,
    name: str,
    *,
    shape: Optional[tuple[int, ...]] = None,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    expected = reference.shape if shape is None else shape
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {tuple(expected)}")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must preserve positions' dtype and device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
