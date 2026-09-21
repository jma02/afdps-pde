"""Small differentiable forward operators for analytic verification problems."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DenseLinearOperator:
    """Apply A to row-batched vectors: (N, d) -> (N, m), with A shaped (m, d).

    Intended for small tests, not for materializing high-dimensional imaging or
    PDE operators. Larger operators only need to satisfy ForwardOperator.
    """

    matrix: Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.numel() == 0:
            raise ValueError("matrix must have nonempty shape (m, d)")
        if self.matrix.dtype not in (torch.float32, torch.float64):
            raise TypeError("matrix must use torch.float32 or torch.float64")
        if not torch.isfinite(self.matrix).all():
            raise ValueError("matrix must contain only finite values")

    def __call__(self, positions: Tensor) -> Tensor:
        if positions.ndim != 2 or positions.shape[1] != self.matrix.shape[1]:
            raise ValueError("positions must have shape (N, matrix.shape[1])")
        if (
            positions.dtype != self.matrix.dtype
            or positions.device != self.matrix.device
        ):
            raise ValueError("positions and matrix must share dtype and device")
        return positions @ self.matrix.T
