from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from automl_image.types import DatasetSpec, MetaFeatures


@dataclass(frozen=True)
class SplitPlan:
    strategy: Literal["holdout", "cv"]
    seed: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()


def make_split(
    spec: DatasetSpec,
    *,
    strategy: Literal["holdout", "cv"],
    val_fraction: float = 0.2,
    n_folds: int = 5,
    seed: int = 42,
) -> SplitPlan:
    raise NotImplementedError


def choose_strategy(meta: MetaFeatures) -> Literal["holdout", "cv"]:
    raise NotImplementedError
