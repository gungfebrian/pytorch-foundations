from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelSummary:
    parameter_count: int
    trainable_parameter_count: int


def summarize_model(model: nn.Module) -> ModelSummary:
    """Count all parameters and the subset updated by the optimizer."""
    parameters = list(model.parameters())
    return ModelSummary(
        parameter_count=sum(parameter.numel() for parameter in parameters),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    )
