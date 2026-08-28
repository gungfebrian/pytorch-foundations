from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TurtleObservations:
    features: torch.Tensor
    labels: torch.Tensor
    session_ids: torch.Tensor


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_test_fraction(test_fraction: float) -> None:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")


def _validate_observations(observations: TurtleObservations) -> None:
    if observations.features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    num_rows = len(observations.features)
    if len(observations.labels) != num_rows or len(observations.session_ids) != num_rows:
        raise ValueError("features, labels, and session_ids must have the same length")
    if num_rows < 2:
        raise ValueError("at least two samples are required to split observations")


def generate_observations(
    n_turtles: int = 8,
    sessions_per_turtle: int = 4,
    images_per_session: int = 8,
    feature_dim: int = 24,
    seed: int = 42,
) -> TurtleObservations:
    for name, value in (
        ("n_turtles", n_turtles),
        ("sessions_per_turtle", sessions_per_turtle),
        ("images_per_session", images_per_session),
        ("feature_dim", feature_dim),
    ):
        _validate_positive_int(name, value)

    generator = torch.Generator().manual_seed(seed)
    identity_vectors = torch.randn(
        (n_turtles, feature_dim),
        generator=generator,
        dtype=torch.float32,
    )

    features: list[torch.Tensor] = []
    labels: list[int] = []
    session_ids: list[int] = []

    for turtle_idx in range(n_turtles):
        turtle_identity = identity_vectors[turtle_idx]
        for session_idx in range(sessions_per_turtle):
            session_style = 1.5 * torch.randn(feature_dim, generator=generator, dtype=torch.float32)
            global_session_id = turtle_idx * sessions_per_turtle + session_idx

            for _ in range(images_per_session):
                image_noise = 0.15 * torch.randn(
                    feature_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
                features.append(turtle_identity + session_style + image_noise)
                labels.append(turtle_idx)
                session_ids.append(global_session_id)

    return TurtleObservations(
        features=torch.stack(features).to(torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        session_ids=torch.tensor(session_ids, dtype=torch.long),
    )


def image_level_split(
    observations: TurtleObservations,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_observations(observations)
    _validate_test_fraction(test_fraction)

    num_samples = len(observations.labels)
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(num_samples, generator=generator)
    test_count = min(max(1, round(num_samples * test_fraction)), num_samples - 1)

    test_idx = torch.sort(shuffled_indices[:test_count]).values
    train_idx = torch.sort(shuffled_indices[test_count:]).values
    return train_idx, test_idx


def session_aware_split(
    observations: TurtleObservations,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_observations(observations)
    _validate_test_fraction(test_fraction)

    generator = torch.Generator().manual_seed(seed)
    test_session_ids: list[int] = []

    for label in torch.unique(observations.labels, sorted=True).tolist():
        label_mask = observations.labels == label
        sessions = torch.unique(observations.session_ids[label_mask], sorted=True)

        if len(sessions) < 2:
            raise ValueError(
                "each turtle must have at least two sessions for session-aware splitting"
            )

        session_order = torch.randperm(len(sessions), generator=generator)
        shuffled_sessions = sessions[session_order]
        num_test_sessions = min(
            max(1, round(len(shuffled_sessions) * test_fraction)),
            len(shuffled_sessions) - 1,
        )
        test_session_ids.extend(shuffled_sessions[:num_test_sessions].tolist())

    test_session_tensor = torch.tensor(test_session_ids, dtype=torch.long)
    test_mask = torch.isin(observations.session_ids, test_session_tensor)
    test_idx = torch.nonzero(test_mask, as_tuple=False).flatten()
    train_idx = torch.nonzero(~test_mask, as_tuple=False).flatten()

    return torch.sort(train_idx).values, torch.sort(test_idx).values
