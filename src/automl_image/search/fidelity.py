from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from automl_image.types import Fidelity, MetaFeatures

if TYPE_CHECKING:
    from automl_image.pipeline import BudgetGuard


@dataclass
class FidelitySchedule:
    dimension: Literal["epochs", "epochs_resolution", "epochs_subset"]
    rungs: list[Fidelity]
    eta: int = 3


def build_schedule(meta: MetaFeatures, budget: BudgetGuard) -> FidelitySchedule:
    raise NotImplementedError
