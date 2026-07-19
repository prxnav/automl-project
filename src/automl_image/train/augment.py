from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from automl_image.types import DatasetSpec, Fidelity, TrainConfig

if TYPE_CHECKING:
    from torchvision.transforms import Compose


def build_transforms(
    config: TrainConfig, fidelity: Fidelity, spec: DatasetSpec
) -> "tuple[Compose, Compose]":
    raise NotImplementedError


def build_mixup_fn(config: TrainConfig, n_classes: int) -> Callable | None:
    raise NotImplementedError
