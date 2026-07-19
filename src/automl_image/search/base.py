from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from automl_image.runstore import RunStore
from automl_image.search.fidelity import FidelitySchedule
from automl_image.types import Fidelity, RunResult, TrainConfig

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from automl_image.pipeline import BudgetGuard
    from automl_image.search.lc_extrapolation import LCPruner


class Searcher(Protocol):
    def run(
        self,
        evaluate: Callable[[TrainConfig, Fidelity], RunResult],
        space: ConfigurationSpace,
        schedule: FidelitySchedule,
        budget: BudgetGuard,
        *,
        initial_design: list[TrainConfig] | None = None,
        warm_start_data: list[RunResult] | None = None,
        lc_pruner: LCPruner | None = None,
        store: RunStore,
    ) -> list[RunResult]: ...

    def save_state(self, path: Path) -> None: ...

    def load_state(self, path: Path) -> None: ...
