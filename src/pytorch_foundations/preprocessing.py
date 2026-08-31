from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_features(features: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if len(features) == 0:
        raise ValueError("at least one feature row is required")
    if not torch.is_floating_point(features):
        raise ValueError("features must use a floating-point dtype")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")


@dataclass(frozen=True)
class FeatureScaler:
    """Standardize feature columns using statistics fitted on training data."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> FeatureScaler:
        _validate_features(features)
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False)
        scale = torch.where(scale > torch.finfo(features.dtype).eps, scale, torch.ones_like(scale))
        return cls(mean=mean, scale=scale)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features)
        if features.shape[1] != self.mean.shape[0]:
            raise ValueError("feature dimension does not match fitted scaler")
        return (features - self.mean) / self.scale
