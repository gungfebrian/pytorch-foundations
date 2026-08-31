import pytest
import torch

from pytorch_foundations.audit import DatasetAudit, audit_observations
from pytorch_foundations.data import TurtleObservations, generate_observations


def test_audit_observations_reports_class_and_session_distribution():
    data = generate_observations(
        n_turtles=3,
        sessions_per_turtle=2,
        images_per_session=4,
        feature_dim=5,
        seed=101,
    )

    audit = audit_observations(data)

    assert audit == DatasetAudit(
        n_samples=24,
        feature_dim=5,
        n_classes=3,
        n_sessions=6,
        samples_per_class=(8, 8, 8),
        sessions_per_class=(2, 2, 2),
    )


def test_audit_observations_rejects_misaligned_tensors():
    data = TurtleObservations(
        features=generate_observations(n_turtles=1, sessions_per_turtle=2).features,
        labels=generate_observations(n_turtles=1, sessions_per_turtle=1).labels,
        session_ids=generate_observations(n_turtles=1, sessions_per_turtle=2).session_ids,
    )

    with pytest.raises(ValueError, match="same length"):
        audit_observations(data)


def test_audit_observations_requires_non_negative_labels():
    data = TurtleObservations(
        features=generate_observations(n_turtles=1, sessions_per_turtle=1).features,
        labels=torch.tensor([-1] * 8, dtype=torch.long),
        session_ids=generate_observations(n_turtles=1, sessions_per_turtle=1).session_ids,
    )

    with pytest.raises(ValueError, match="non-negative"):
        audit_observations(data)
