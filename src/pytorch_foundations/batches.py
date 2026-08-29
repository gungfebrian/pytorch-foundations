from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a deterministic mini-batch loader for feature/label tensors."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if len(features) == 0:
        raise ValueError("features must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
