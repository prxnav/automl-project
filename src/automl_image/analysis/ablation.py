from __future__ import annotations

from typing import TYPE_CHECKING

from automl_image.types import TrainConfig

if TYPE_CHECKING:
    import pandas as pd


def ablation_path(
    default: TrainConfig, incumbent: TrainConfig, evaluate
) -> pd.DataFrame:
    raise NotImplementedError
