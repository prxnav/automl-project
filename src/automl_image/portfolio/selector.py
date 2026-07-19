from __future__ import annotations

from typing import TYPE_CHECKING

from automl_image.types import TrainConfig

if TYPE_CHECKING:
    import pandas as pd


def choose(
    front: "pd.DataFrame",
    *,
    rule: str = "knee",
    constraint: dict | None = None,
    top_k: int = 3,
) -> list[TrainConfig]:
    raise NotImplementedError
