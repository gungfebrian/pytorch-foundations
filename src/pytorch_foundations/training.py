from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from .batches import make_dataloader
from .data import TurtleObservations
from .model import TurtleClassifier


@dataclass(frozen=True)
class TrainingMetrics:
    epochs: int
    final_train_loss: float
    train_accuracy: float
    test_accuracy: float


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validate_observations(observations: TurtleObservations) -> None:
    if observations.features.ndim != 2:
        raise ValueError("features must be a 2D tensor")

    num_rows = len(observations.features)
    if len(observations.labels) != num_rows or len(observations.session_ids) != num_rows:
        raise ValueError("features, labels, and session_ids must have the same length")
    if num_rows < 2:
        raise ValueError("at least two samples are required for training")


def _validate_indices(name: str, indices: torch.Tensor, upper_bound: int) -> None:
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    if len(indices) == 0:
        raise ValueError(f"{name} must not be empty")
    if indices.dtype != torch.long:
        raise ValueError(f"{name} must use torch.long dtype")
    if torch.any(indices < 0) or torch.any(indices >= upper_bound):
        raise ValueError(f"{name} contains out-of-range indices")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def train_and_evaluate(
    observations: TurtleObservations,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    epochs: int = 40,
    learning_rate: float = 0.03,
    seed: int = 42,
    batch_size: int | None = None,
) -> TrainingMetrics:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive when provided")

    _validate_observations(observations)
    _validate_indices("train_indices", train_indices, len(observations.labels))
    _validate_indices("test_indices", test_indices, len(observations.labels))

    set_deterministic_seed(seed)

    train_features = observations.features[train_indices]
    train_labels = observations.labels[train_indices]
    test_features = observations.features[test_indices]
    test_labels = observations.labels[test_indices]

    feature_dim = train_features.shape[1]
    n_classes = int(observations.labels.max().item()) + 1
    model = TurtleClassifier(feature_dim=feature_dim, n_classes=n_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_train_loss = 0.0
    model.train()
    train_loader = None
    if batch_size is not None:
        train_loader = make_dataloader(
            train_features,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )

    for _ in range(epochs):
        if train_loader is None:
            batches = [(train_features, train_labels)]
        else:
            batches = train_loader

        for batch_features, batch_labels in batches:
            optimizer.zero_grad()
            train_logits = model(batch_features)
            loss = loss_fn(train_logits, batch_labels)
            loss.backward()
            optimizer.step()
            final_train_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        train_logits = model(train_features)
        test_logits = model(test_features)
        train_accuracy = _accuracy(train_logits, train_labels)
        test_accuracy = _accuracy(test_logits, test_labels)

    return TrainingMetrics(
        epochs=epochs,
        final_train_loss=float(final_train_loss),
        train_accuracy=float(train_accuracy),
        test_accuracy=float(test_accuracy),
    )
