from __future__ import annotations

from typing import TYPE_CHECKING

from automl_image.types import MetaFeatures, RunResult, TrainConfig

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


def get_initial_design(
    meta: MetaFeatures,
    space: ConfigurationSpace,
    *,
    k: int = 8,
    exclude_dataset: str | None = None,
) -> list[TrainConfig]:
    raise NotImplementedError


def get_surrogate_warm_start_data(
    meta: MetaFeatures, exclude_dataset: str | None = None
) -> list[RunResult]:
    raise NotImplementedError
