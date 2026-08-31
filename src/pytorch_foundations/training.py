from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from .batches import make_dataloader
from .data import TurtleObservations
from .metrics import classification_metrics
from .model import TurtleClassifier
from .preprocessing import FeatureScaler


@dataclass(frozen=True)
class TrainingMetrics:
    epochs: int
    epochs_ran: int
    best_epoch: int
    final_train_loss: float
    train_accuracy: float
    validation_accuracy: float | None
    test_accuracy: float
    test_macro_precision: float
    test_macro_recall: float
    test_macro_f1: float
    test_confusion_matrix: tuple[tuple[int, ...], ...]
    features_normalized: bool


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


def _assert_disjoint(first: torch.Tensor, second: torch.Tensor) -> None:
    if torch.isin(first, second).any():
        raise ValueError("training, validation, and test indices must be disjoint")


def train_and_evaluate(
    observations: TurtleObservations,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    epochs: int = 40,
    learning_rate: float = 0.03,
    seed: int = 42,
    batch_size: int | None = None,
    normalize: bool = False,
    validation_indices: torch.Tensor | None = None,
    early_stopping_patience: int | None = None,
) -> TrainingMetrics:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive when provided")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when provided")
    if early_stopping_patience is not None and validation_indices is None:
        raise ValueError("validation_indices are required for early stopping")

    _validate_observations(observations)
    _validate_indices("train_indices", train_indices, len(observations.labels))
    _validate_indices("test_indices", test_indices, len(observations.labels))
    if validation_indices is not None:
        _validate_indices("validation_indices", validation_indices, len(observations.labels))
        _assert_disjoint(train_indices, validation_indices)
        _assert_disjoint(test_indices, validation_indices)

    set_deterministic_seed(seed)

    train_features = observations.features[train_indices]
    train_labels = observations.labels[train_indices]
    test_features = observations.features[test_indices]
    test_labels = observations.labels[test_indices]
    validation_features = None
    validation_labels = None
    if validation_indices is not None:
        validation_features = observations.features[validation_indices]
        validation_labels = observations.labels[validation_indices]

    if normalize:
        scaler = FeatureScaler.fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

    feature_dim = train_features.shape[1]
    n_classes = int(observations.labels.max().item()) + 1
    model = TurtleClassifier(feature_dim=feature_dim, n_classes=n_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_train_loss = 0.0
    epochs_ran = 0
    best_epoch = epochs
    best_validation_accuracy = None
    best_state: dict[str, torch.Tensor] | None = None
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

    for epoch in range(1, epochs + 1):
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

        epochs_ran = epoch
        if validation_features is not None and validation_labels is not None:
            model.eval()
            with torch.no_grad():
                validation_logits = model(validation_features)
                current_validation_accuracy = _accuracy(validation_logits, validation_labels)
            model.train()

            if (
                best_validation_accuracy is None
                or current_validation_accuracy > best_validation_accuracy
            ):
                best_validation_accuracy = current_validation_accuracy
                best_epoch = epoch
                best_state = {
                    name: parameter.detach().clone()
                    for name, parameter in model.state_dict().items()
                }
            elif (
                early_stopping_patience is not None
                and epoch - best_epoch >= early_stopping_patience
            ):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_logits = model(train_features)
        validation_accuracy = None
        if validation_features is not None and validation_labels is not None:
            validation_accuracy = _accuracy(model(validation_features), validation_labels)
        test_logits = model(test_features)
        train_accuracy = _accuracy(train_logits, train_labels)
        test_metrics = classification_metrics(test_logits, test_labels, n_classes=n_classes)

    return TrainingMetrics(
        epochs=epochs,
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        final_train_loss=float(final_train_loss),
        train_accuracy=float(train_accuracy),
        validation_accuracy=validation_accuracy,
        test_accuracy=test_metrics.accuracy,
        test_macro_precision=test_metrics.macro_precision,
        test_macro_recall=test_metrics.macro_recall,
        test_macro_f1=test_metrics.macro_f1,
        test_confusion_matrix=tuple(
            tuple(int(value) for value in row) for row in test_metrics.confusion_matrix.tolist()
        ),
        features_normalized=normalize,
    )
