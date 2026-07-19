from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def build_benchmark_table(store_root: Path) -> "pd.DataFrame":
    raise NotImplementedError
