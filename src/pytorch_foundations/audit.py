from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import TurtleObservations


@dataclass(frozen=True)
class DatasetAudit:
    n_samples: int
    feature_dim: int
    n_classes: int
    n_sessions: int
    samples_per_class: tuple[int, ...]
    sessions_per_class: tuple[int, ...]


def audit_observations(observations: TurtleObservations) -> DatasetAudit:
    """Return shape and grouping information for a set of observations."""
    if observations.features.ndim != 2:
        raise ValueError("features must be a 2D tensor")

    n_samples = len(observations.features)
    if len(observations.labels) != n_samples or len(observations.session_ids) != n_samples:
        raise ValueError("features, labels, and session_ids must have the same length")
    if n_samples == 0:
        raise ValueError("at least one sample is required")
    if observations.labels.dtype != torch.long:
        raise ValueError("labels must use torch.long dtype")
    if observations.session_ids.dtype != torch.long:
        raise ValueError("session_ids must use torch.long dtype")
    if torch.any(observations.labels < 0):
        raise ValueError("labels must be non-negative")

    classes = torch.unique(observations.labels, sorted=True)
    samples_per_class = tuple(int((observations.labels == label).sum().item()) for label in classes)
    sessions_per_class = tuple(
        int(torch.unique(observations.session_ids[observations.labels == label]).numel())
        for label in classes
    )

    return DatasetAudit(
        n_samples=n_samples,
        feature_dim=int(observations.features.shape[1]),
        n_classes=len(classes),
        n_sessions=int(torch.unique(observations.session_ids).numel()),
        samples_per_class=samples_per_class,
        sessions_per_class=sessions_per_class,
    )
