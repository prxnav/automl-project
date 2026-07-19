from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def fanova_importance(runs: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    raise NotImplementedError
