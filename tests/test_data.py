import pytest
import torch

from pytorch_foundations.data import (
    TurtleObservations,
    generate_observations,
    image_level_split,
    session_aware_split,
)


def test_generate_observations_is_deterministic_and_aligned():
    first = generate_observations(seed=7)
    second = generate_observations(seed=7)

    assert torch.equal(first.features, second.features)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.session_ids, second.session_ids)
    assert first.features.ndim == 2
    assert first.features.dtype == torch.float32
    assert first.labels.dtype == torch.long
    assert first.session_ids.dtype == torch.long
    assert len(first.features) == len(first.labels) == len(first.session_ids)


def test_image_level_split_returns_non_empty_disjoint_partitions():
    data = generate_observations(seed=9)

    train_idx, test_idx = image_level_split(data, test_fraction=0.25, seed=11)

    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))


def test_session_aware_split_keeps_sessions_disjoint_across_partitions():
    data = generate_observations(seed=13)

    train_idx, test_idx = session_aware_split(data, test_fraction=0.25, seed=17)

    train_sessions = set(data.session_ids[train_idx].tolist())
    test_sessions = set(data.session_ids[test_idx].tolist())

    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert train_sessions.isdisjoint(test_sessions)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_turtles": 0}, "n_turtles"),
        ({"sessions_per_turtle": 0}, "sessions_per_turtle"),
        ({"images_per_session": 0}, "images_per_session"),
        ({"feature_dim": 0}, "feature_dim"),
    ],
)
def test_generate_observations_rejects_non_positive_sizes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_observations(**kwargs)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.25, 1.25])
def test_split_functions_require_a_proper_fraction(fraction):
    data = generate_observations(seed=23)

    with pytest.raises(ValueError, match="test_fraction"):
        image_level_split(data, test_fraction=fraction)

    with pytest.raises(ValueError, match="test_fraction"):
        session_aware_split(data, test_fraction=fraction)


def test_split_rejects_malformed_scalar_feature_observations():
    malformed = TurtleObservations(
        features=torch.tensor(1.0, dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        session_ids=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="features must be a 2D tensor"):
        image_level_split(malformed)
