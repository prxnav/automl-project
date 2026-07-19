from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from automl_image.types import DatasetSpec, TrainConfig

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def retrain_full(
    configs: list[TrainConfig], spec: DatasetSpec, *, epochs: int, device: str
) -> list[Path]:
    raise NotImplementedError


def soft_vote(
    checkpoints: list[Path], loader: DataLoader, *, tta: bool = True, tta_views: int = 4
) -> np.ndarray:
    raise NotImplementedError


def write_predictions(
    probs: np.ndarray, class_mapping: dict[int, int], out: Path
) -> None:
    raise NotImplementedError
