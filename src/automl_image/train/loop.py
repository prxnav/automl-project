from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from automl_image.types import DatasetSpec, EpochRecord, Fidelity, RunResult, TrainConfig


def evaluate_config(
    config: TrainConfig,
    fidelity: Fidelity,
    dataset: DatasetSpec,
    *,
    device: str = "cuda:0",
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    time_limit_s: float | None = None,
    lc_callback: Callable[[EpochRecord], bool] | None = None,
    dry_run: bool = False,
) -> RunResult:
    raise NotImplementedError
