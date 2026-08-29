import pytest

from pytorch_foundations.data import (
    generate_observations,
    session_aware_train_validation_test_split,
)


def test_session_aware_three_way_split_keeps_all_sessions_disjoint():
    data = generate_observations(seed=61)

    train_idx, validation_idx, test_idx = session_aware_train_validation_test_split(
        data, validation_fraction=0.2, test_fraction=0.25, seed=67
    )

    train_sessions = set(data.session_ids[train_idx].tolist())
    validation_sessions = set(data.session_ids[validation_idx].tolist())
    test_sessions = set(data.session_ids[test_idx].tolist())

    assert train_sessions.isdisjoint(validation_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert validation_sessions.isdisjoint(test_sessions)
    assert min(len(train_idx), len(validation_idx), len(test_idx)) > 0


def test_session_aware_three_way_split_is_deterministic():
    data = generate_observations(seed=71)

    first = session_aware_train_validation_test_split(data, seed=73)
    second = session_aware_train_validation_test_split(data, seed=73)

    assert all(first_idx.equal(second_idx) for first_idx, second_idx in zip(first, second))


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction"),
    [(0.0, 0.25), (0.2, 0.0), (0.8, 0.3)],
)
def test_session_aware_three_way_split_rejects_invalid_fractions(
    validation_fraction, test_fraction
):
    data = generate_observations(seed=79)

    with pytest.raises(ValueError, match="fractions"):
        session_aware_train_validation_test_split(
            data,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
