from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def paired_wilcoxon(a: "pd.Series", b: "pd.Series") -> tuple[float, float]:
    raise NotImplementedError


def bootstrap_ci(x, n: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    raise NotImplementedError


def summarize(store, group_cols) -> "pd.DataFrame":
    raise NotImplementedError
