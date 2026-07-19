from __future__ import annotations

from typing import TYPE_CHECKING

from automl_image.types import TrainConfig

if TYPE_CHECKING:
    from torch import nn


def build_model(config: TrainConfig, n_classes: int) -> "nn.Module":
    raise NotImplementedError


def build_param_groups(model: "nn.Module", config: TrainConfig) -> list[dict]:
    raise NotImplementedError
