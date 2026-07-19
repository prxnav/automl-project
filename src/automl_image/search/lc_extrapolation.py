from __future__ import annotations

from typing import Literal

from automl_image.types import EpochRecord


class LCPruner:
    def __init__(
        self,
        incumbent_acc: float,
        min_epochs: int = 3,
        confidence: float = 0.95,
        model: Literal["pow4", "mmf", "exp"] = "pow4",
    ) -> None:
        raise NotImplementedError

    def should_stop(self, curve: list[EpochRecord], max_epochs: int) -> bool:
        raise NotImplementedError

    def predict_final(self, curve: list[EpochRecord], max_epochs: int) -> tuple[float, float]:
        raise NotImplementedError
