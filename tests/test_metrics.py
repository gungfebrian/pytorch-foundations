import math

import pytest
import torch

from pytorch_foundations.metrics import ClassificationMetrics, classification_metrics


def test_classification_metrics_returns_confusion_matrix_and_macro_scores():
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    metrics = classification_metrics(logits, labels)

    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.confusion_matrix.tolist() == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert math.isclose(metrics.accuracy, 0.75)
    assert math.isclose(metrics.macro_precision, 5 / 6, rel_tol=1e-6)
    assert math.isclose(metrics.macro_recall, 5 / 6, rel_tol=1e-6)
    assert math.isclose(metrics.macro_f1, 7 / 9, rel_tol=1e-6)


def test_classification_metrics_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same number"):
        classification_metrics(torch.zeros((2, 3)), torch.zeros(1, dtype=torch.long))


def test_classification_metrics_rejects_out_of_range_labels():
    with pytest.raises(ValueError, match="range"):
        classification_metrics(torch.zeros((2, 3)), torch.tensor([0, 3], dtype=torch.long))
