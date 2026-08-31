from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    seed: int
    feature_dim: int
    n_classes: int


def _validate_metadata(metadata: CheckpointMetadata) -> None:
    if metadata.epoch < 1:
        raise ValueError("checkpoint epoch must be positive")
    if metadata.feature_dim < 1:
        raise ValueError("checkpoint feature_dim must be positive")
    if metadata.n_classes < 1:
        raise ValueError("checkpoint n_classes must be positive")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
) -> None:
    """Save model weights and the minimum metadata needed to identify a run."""
    _validate_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "metadata": asdict(metadata)},
        path,
    )


def load_checkpoint(path: Path, model: nn.Module) -> CheckpointMetadata:
    """Load weights into ``model`` and return the saved run metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    metadata = CheckpointMetadata(**payload["metadata"])
    _validate_metadata(metadata)
    return metadata
