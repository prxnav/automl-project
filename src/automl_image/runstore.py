from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from automl_image.types import EpochRecord, RunResult

if TYPE_CHECKING:
    import pandas as pd


def validate_result(result: RunResult) -> None:
    raise NotImplementedError


class RunStore:
    def __init__(self, root: Path, experiment: str):
        raise NotImplementedError

    def log(self, result: RunResult) -> str:
        raise NotImplementedError

    def log_epoch(self, run_id: str, rec: EpochRecord) -> None:
        raise NotImplementedError

    def load(self, **filters) -> pd.DataFrame:
        raise NotImplementedError

    def learning_curves(self, **filters) -> pd.DataFrame:
        raise NotImplementedError

    def ledger(self) -> pd.DataFrame:
        raise NotImplementedError
