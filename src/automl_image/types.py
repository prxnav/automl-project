from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    n_classes: int
    n_train: int
    n_val: int
    n_test: int
    class_counts: dict[int, int]
    native_resolution: tuple[int, int]
    channels: int
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]


@dataclass(frozen=True)
class MetaFeatures:
    n_samples: int
    n_classes: int
    samples_per_class_mean: float
    samples_per_class_min: int
    class_balance_entropy: float
    median_resolution: int
    resolution_std: float
    aspect_ratio_mean: float
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]
    mean_image_entropy: float
    probe_acc_1epoch: float
    probe_loss_slope: float
    probe_seconds: float

    def to_vector(self) -> list[float]:
        raise NotImplementedError

    @staticmethod
    def feature_names() -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Fidelity:
    epochs: int
    resolution: int = 224
    data_fraction: float = 1.0

    def cost_weight(self) -> float:
        return self.epochs * (self.resolution / 224) ** 2 * self.data_fraction


@dataclass(frozen=True)
class TrainConfig:
    backbone: str
    ft_strategy: Literal[
        "linear_probe", "last_block", "last_n_blocks", "full", "discriminative_lr"
    ]
    n_unfrozen_blocks: int = 0
    augment: Literal[
        "none", "flip_crop", "randaugment", "trivialaugment", "autoaugment"
    ] = "randaugment"
    randaugment_n: int = 2
    randaugment_m: int = 9
    mixup: bool = False
    mixup_alpha: float = 0.2
    cutmix: bool = False
    cutmix_alpha: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    lr_schedule: Literal["cosine", "step", "constant", "plateau"] = "cosine"
    label_smoothing: float = 0.0
    batch_size: int = 32
    dropout: float = 0.0
    seed: int = 0

    @classmethod
    def default(cls) -> "TrainConfig":
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def hash(self) -> str:
        raise NotImplementedError


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    seconds: float
    peak_vram_mb: float


@dataclass
class RunResult:
    config: TrainConfig
    fidelity: Fidelity
    dataset: str
    val_acc: float
    val_macro_f1: float
    train_loss: float
    learning_curve: list[EpochRecord]
    n_params: int
    flops_g: float
    latency_ms: Optional[float]
    wallclock_seconds: float
    gpu_seconds: float
    peak_vram_mb: float
    device: str
    status: Literal["ok", "oom", "diverged", "timeout", "early_stopped", "error"]
    error: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    run_id: str = ""
