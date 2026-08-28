from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import generate_observations, image_level_split, session_aware_split
from .training import TrainingMetrics, train_and_evaluate


def _dataset_summary(data) -> dict[str, int]:
    return {
        "n_samples": len(data.labels),
        "feature_dim": int(data.features.shape[1]),
        "n_classes": int(data.labels.unique().numel()),
        "n_sessions": int(data.session_ids.unique().numel()),
    }


def _overlapping_sessions(data, train_idx, test_idx) -> int:
    train_sessions = set(data.session_ids[train_idx].tolist())
    test_sessions = set(data.session_ids[test_idx].tolist())
    return len(train_sessions & test_sessions)


def _serialize_metrics(
    metrics: TrainingMetrics, overlapping_sessions: int
) -> dict[str, float | int]:
    payload = asdict(metrics)
    payload["overlapping_sessions"] = overlapping_sessions
    return payload


def run_experiment(
    epochs: int = 40,
    seed: int = 42,
    output: Path | None = None,
) -> dict[str, object]:
    data = generate_observations(seed=seed)

    image_train_idx, image_test_idx = image_level_split(data, seed=seed)
    session_train_idx, session_test_idx = session_aware_split(data, seed=seed)

    image_metrics = train_and_evaluate(
        data,
        image_train_idx,
        image_test_idx,
        epochs=epochs,
        seed=seed,
    )
    session_metrics = train_and_evaluate(
        data,
        session_train_idx,
        session_test_idx,
        epochs=epochs,
        seed=seed,
    )

    result: dict[str, object] = {
        "dataset": _dataset_summary(data),
        "image_level_split": _serialize_metrics(
            image_metrics,
            _overlapping_sessions(data, image_train_idx, image_test_idx),
        ),
        "session_aware_split": _serialize_metrics(
            session_metrics,
            _overlapping_sessions(data, session_train_idx, session_test_idx),
        ),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PyTorch foundations leakage experiment.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for each split.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for all steps.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    try:
        result = run_experiment(epochs=args.epochs, seed=args.seed, output=args.output)
    except ValueError as exc:
        parser.error(str(exc))

    dataset = result["dataset"]
    print(
        "Dataset:"
        f" {dataset['n_samples']} samples,"
        f" {dataset['n_classes']} classes,"
        f" {dataset['n_sessions']} sessions,"
        f" feature_dim={dataset['feature_dim']}"
    )

    for heading, key in (
        ("Image-Level Split", "image_level_split"),
        ("Session-Aware Split", "session_aware_split"),
    ):
        metrics = result[key]
        print(heading)
        print(f"  overlapping sessions: {metrics['overlapping_sessions']}")
        print(f"  test accuracy: {metrics['test_accuracy']:.3f}")
        print(f"  final train loss: {metrics['final_train_loss']:.3f}")

    if args.output is not None:
        print(f"Saved JSON report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
