from __future__ import annotations

import torch
from torch import nn


class TurtleClassifier(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be at least 1")
        if n_classes < 1:
            raise ValueError("n_classes must be at least 1")

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 48),
            nn.ReLU(),
            nn.Linear(48, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
