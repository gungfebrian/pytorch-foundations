from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: torch.Tensor


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int | None = None,
) -> ClassificationMetrics:
    """Calculate accuracy, macro scores, and a confusion table of counts."""
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same number of rows")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long dtype")

    class_count = logits.shape[1] if n_classes is None else n_classes
    if class_count < 1:
        raise ValueError("n_classes must be positive")
    if torch.any(labels < 0) or torch.any(labels >= class_count):
        raise ValueError("labels must be within the class range")

    predictions = logits.argmax(dim=1)
    confusion = torch.bincount(
        labels * class_count + predictions,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    true_positives = confusion.diag().to(torch.float32)
    predicted_counts = confusion.sum(dim=0).to(torch.float32)
    actual_counts = confusion.sum(dim=1).to(torch.float32)
    precision = true_positives / predicted_counts.clamp_min(1.0)
    recall = true_positives / actual_counts.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)

    return ClassificationMetrics(
        accuracy=float((predictions == labels).float().mean().item()),
        macro_precision=float(precision.mean().item()),
        macro_recall=float(recall.mean().item()),
        macro_f1=float(f1.mean().item()),
        confusion_matrix=confusion,
    )
