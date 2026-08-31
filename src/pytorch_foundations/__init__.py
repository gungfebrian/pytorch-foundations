"""PyTorch foundations package."""

from .audit import DatasetAudit, audit_observations
from .batches import make_dataloader
from .checkpoints import CheckpointMetadata, load_checkpoint, save_checkpoint
from .data import (
    TurtleObservations,
    generate_observations,
    image_level_split,
    session_aware_split,
    session_aware_train_validation_test_split,
)
from .metrics import ClassificationMetrics, classification_metrics
from .model import TurtleClassifier
from .model_summary import ModelSummary, summarize_model
from .preprocessing import FeatureScaler
from .tensor_basics import TensorSummary, normalize_features, summarize_tensor
from .training import TrainingMetrics, set_deterministic_seed, train_and_evaluate

__all__ = [
    "CheckpointMetadata",
    "ClassificationMetrics",
    "DatasetAudit",
    "FeatureScaler",
    "ModelSummary",
    "TensorSummary",
    "TrainingMetrics",
    "TurtleClassifier",
    "TurtleObservations",
    "audit_observations",
    "classification_metrics",
    "generate_observations",
    "image_level_split",
    "load_checkpoint",
    "make_dataloader",
    "normalize_features",
    "save_checkpoint",
    "session_aware_split",
    "session_aware_train_validation_test_split",
    "set_deterministic_seed",
    "summarize_model",
    "summarize_tensor",
    "train_and_evaluate",
]
