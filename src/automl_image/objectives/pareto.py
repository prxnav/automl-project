from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from automl_image.types import RunResult

if TYPE_CHECKING:
    import pandas as pd


def is_dominated(a: tuple[float, float], b: tuple[float, float]) -> bool:
    raise NotImplementedError


def pareto_front(points: np.ndarray, maximize: tuple[bool, bool]) -> np.ndarray:
    raise NotImplementedError


def hypervolume(front: np.ndarray, reference: tuple[float, float]) -> float:
    raise NotImplementedError


def hypervolume_over_time(
    results: list[RunResult], reference: tuple[float, float]
) -> "pd.DataFrame":
    raise NotImplementedError


def knee_point(front: np.ndarray) -> int:
    raise NotImplementedError
