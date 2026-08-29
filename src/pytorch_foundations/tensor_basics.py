from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSummary:
    shape: tuple[int, ...]
    dtype: str
    ndim: int
    numel: int


def summarize_tensor(tensor: torch.Tensor) -> TensorSummary:
    """Return the shape and storage details worth checking before training."""
    return TensorSummary(
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        ndim=tensor.ndim,
        numel=tensor.numel(),
    )


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Standardize each feature column using its sample mean and deviation."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if features.shape[0] < 2:
        raise ValueError("at least two rows are required to normalize features")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")

    mean = features.mean(dim=0, keepdim=True)
    standard_deviation = features.std(dim=0, keepdim=True)
    return (features - mean) / standard_deviation.clamp_min(torch.finfo(features.dtype).eps)
