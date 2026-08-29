"""PyTorch foundations package."""

from .batches import make_dataloader
from .data import (
    TurtleObservations,
    generate_observations,
    image_level_split,
    session_aware_split,
    session_aware_train_validation_test_split,
)
from .model import TurtleClassifier
from .tensor_basics import TensorSummary, normalize_features, summarize_tensor
from .training import TrainingMetrics, set_deterministic_seed, train_and_evaluate

__all__ = [
    "TensorSummary",
    "TrainingMetrics",
    "TurtleClassifier",
    "TurtleObservations",
    "generate_observations",
    "image_level_split",
    "make_dataloader",
    "normalize_features",
    "session_aware_split",
    "session_aware_train_validation_test_split",
    "set_deterministic_seed",
    "summarize_tensor",
    "train_and_evaluate",
]
