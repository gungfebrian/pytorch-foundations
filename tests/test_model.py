import pytest
import torch

from pytorch_foundations.model import TurtleClassifier


def test_classifier_returns_one_logit_per_class():
    model = TurtleClassifier(feature_dim=24, n_classes=8)

    logits = model(torch.zeros(5, 24))

    assert logits.shape == (5, 8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"feature_dim": 0, "n_classes": 8}, "feature_dim"),
        ({"feature_dim": 24, "n_classes": 0}, "n_classes"),
    ],
)
def test_classifier_rejects_invalid_dimensions(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TurtleClassifier(**kwargs)
