from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from automl_image.types import DatasetSpec, Fidelity, TrainConfig

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def load(dataset_path: Path, cache_dir: Path | None = None) -> DatasetSpec:
    raise NotImplementedError


def build_dataloaders(
    spec: DatasetSpec,
    config: TrainConfig,
    fidelity: Fidelity,
    split_indices,
    *,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    raise NotImplementedError
