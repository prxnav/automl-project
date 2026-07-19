from __future__ import annotations

from typing import TYPE_CHECKING

from automl_image.types import DatasetSpec, MetaFeatures, TrainConfig

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace


def build_space(
    dataset: DatasetSpec,
    meta: MetaFeatures,
    *,
    backbones: list[str] | None = None,
    vram_gb: float = 6.0,
) -> "ConfigurationSpace":
    raise NotImplementedError


def config_from_configuration(c: "Configuration", seed: int = 0) -> TrainConfig:
    raise NotImplementedError


def configuration_from_config(c: TrainConfig, space: "ConfigurationSpace") -> "Configuration":
    raise NotImplementedError
