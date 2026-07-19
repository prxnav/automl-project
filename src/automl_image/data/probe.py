from __future__ import annotations

from automl_image.data.splits import SplitPlan
from automl_image.types import DatasetSpec


def run_probe(
    spec: DatasetSpec,
    split: SplitPlan,
    *,
    backbone: str = "resnet18",
    epochs: int = 1,
    resolution: int = 96,
    device: str = "cuda:0",
) -> tuple[float, float, float]:
    raise NotImplementedError
