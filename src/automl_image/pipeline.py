from __future__ import annotations

from pathlib import Path
from typing import Literal


class BudgetGuard:
    def __init__(self, budget_hours: float) -> None:
        raise NotImplementedError

    def remaining_seconds(self) -> float:
        raise NotImplementedError

    def can_afford(self, estimated_seconds: float) -> bool:
        raise NotImplementedError


def run_pipeline(
    dataset_path: Path,
    out_dir: Path,
    budget_hours: float = 24.0,
    *,
    searcher: Literal["bohb", "hyperband", "random", "optuna_asha", "default"] = "bohb",
    warm_start: bool = True,
    lc_extrapolation: bool = True,
    multi_objective: bool = True,
    seeds: int = 1,
    dry_run: bool = False,
) -> None:
    raise NotImplementedError
