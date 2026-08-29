import math

import pytest
import torch

from pytorch_foundations.data import TurtleObservations, generate_observations, image_level_split
from pytorch_foundations.training import train_and_evaluate


def test_train_and_evaluate_returns_finite_bounded_metrics():
    data = generate_observations(
        n_turtles=4,
        sessions_per_turtle=2,
        images_per_session=3,
        feature_dim=6,
        seed=5,
    )
    train_idx, test_idx = image_level_split(data, test_fraction=0.25, seed=7)

    metrics = train_and_evaluate(data, train_idx, test_idx, epochs=2, seed=11)

    assert metrics.epochs == 2
    assert 0.0 <= metrics.train_accuracy <= 1.0
    assert 0.0 <= metrics.test_accuracy <= 1.0
    assert 0.0 <= metrics.test_macro_precision <= 1.0
    assert 0.0 <= metrics.test_macro_recall <= 1.0
    assert 0.0 <= metrics.test_macro_f1 <= 1.0
    assert len(metrics.test_confusion_matrix) == 4
    assert math.isfinite(metrics.final_train_loss)


def test_train_and_evaluate_is_deterministic_for_the_same_seed():
    data = generate_observations(
        n_turtles=4,
        sessions_per_turtle=2,
        images_per_session=3,
        feature_dim=6,
        seed=13,
    )
    train_idx, test_idx = image_level_split(data, test_fraction=0.25, seed=17)

    first = train_and_evaluate(data, train_idx, test_idx, epochs=2, seed=19)
    second = train_and_evaluate(data, train_idx, test_idx, epochs=2, seed=19)

    assert first == second


def test_train_and_evaluate_rejects_empty_train_split():
    data = generate_observations(seed=23)
    test_idx = torch.arange(len(data.labels))
    train_idx = torch.tensor([], dtype=torch.long)

    with pytest.raises(ValueError, match="train_indices"):
        train_and_evaluate(data, train_idx, test_idx)


def test_train_and_evaluate_rejects_empty_test_split():
    data = generate_observations(seed=29)
    train_idx = torch.arange(len(data.labels))
    test_idx = torch.tensor([], dtype=torch.long)

    with pytest.raises(ValueError, match="test_indices"):
        train_and_evaluate(data, train_idx, test_idx)


@pytest.mark.parametrize("epochs", [0, -1])
def test_train_and_evaluate_requires_positive_epochs(epochs):
    data = generate_observations(seed=31)
    train_idx, test_idx = image_level_split(data, seed=37)

    with pytest.raises(ValueError, match="epochs"):
        train_and_evaluate(data, train_idx, test_idx, epochs=epochs)


@pytest.mark.parametrize("learning_rate", [0.0, -0.01])
def test_train_and_evaluate_requires_positive_learning_rate(learning_rate):
    data = generate_observations(seed=41)
    train_idx, test_idx = image_level_split(data, seed=43)

    with pytest.raises(ValueError, match="learning_rate"):
        train_and_evaluate(data, train_idx, test_idx, learning_rate=learning_rate)


def test_train_and_evaluate_rejects_misaligned_observation_lengths():
    malformed = TurtleObservations(
        features=torch.zeros((2, 3), dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        session_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    train_idx = torch.tensor([0], dtype=torch.long)
    test_idx = torch.tensor([1], dtype=torch.long)

    with pytest.raises(ValueError, match="same length"):
        train_and_evaluate(malformed, train_idx, test_idx)


def test_train_and_evaluate_supports_deterministic_mini_batches():
    data = generate_observations(
        n_turtles=4,
        sessions_per_turtle=2,
        images_per_session=3,
        feature_dim=6,
        seed=47,
    )
    train_idx, test_idx = image_level_split(data, test_fraction=0.25, seed=53)

    first = train_and_evaluate(data, train_idx, test_idx, epochs=2, batch_size=4, seed=59)
    second = train_and_evaluate(data, train_idx, test_idx, epochs=2, batch_size=4, seed=59)

    assert first == second
