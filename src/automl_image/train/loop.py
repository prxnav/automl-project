from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

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
    lc_callback: Optional[Callable[[EpochRecord], bool]] = None,
    dry_run: bool = False,
) -> RunResult:
    raise NotImplementedError
