from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from automl_image.runstore import RunStore

if TYPE_CHECKING:
    import pandas as pd


def build_ledger(store: RunStore) -> "pd.DataFrame":
    raise NotImplementedError


def budget_report(run_dir: Path) -> "pd.DataFrame":
    raise NotImplementedError
