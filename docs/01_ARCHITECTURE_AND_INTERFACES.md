# 01 — Architecture & Interface Contract

> **This file is the contract.** It is frozen at the first integration checkpoint (M1). Members build against these signatures, not against each other's code. If a signature must change, it changes here first, in a PR, with both other members tagged.

## 1.1 Repository layout

```
automl-image/
├── README.md
├── environment.yml              # pinned; A owns
├── pyproject.toml               # package config, ruff+pytest settings
├── docs/                        # this folder
├── configs/
│   ├── search_space.yaml        # bounds + defaults + conditionals (B owns, A reviews)
│   ├── budget.yaml              # 24h allocation, fidelity rungs
│   ├── datasets/*.yaml          # per-dataset paths & overrides
│   └── best_configs/            # frozen incumbents that reproduce reported numbers
├── src/automl_image/
│   ├── types.py                 # ← shared dataclasses. A owns. EVERYONE imports this.
│   ├── data/                    # A
│   │   ├── ingest.py
│   │   ├── splits.py
│   │   ├── metafeatures.py
│   │   └── probe.py
│   ├── train/                   # A
│   │   ├── loop.py
│   │   ├── augment.py
│   │   ├── model_factory.py
│   │   ├── evaluate.py
│   │   └── ensemble.py
│   ├── search/                  # B
│   │   ├── space.py
│   │   ├── fidelity.py
│   │   ├── smac_driver.py
│   │   ├── optuna_driver.py
│   │   ├── lc_extrapolation.py
│   │   └── scheduler.py
│   ├── portfolio/               # C
│   │   ├── benchmark_table.py
│   │   ├── warm_start.py
│   │   └── selector.py
│   ├── objectives/              # C
│   │   ├── cost.py
│   │   └── pareto.py
│   ├── analysis/                # C
│   │   ├── fanova.py
│   │   ├── ablation.py
│   │   ├── ledger.py
│   │   ├── stats.py
│   │   └── plots.py
│   ├── runstore.py              # C owns; A and B write through it
│   └── pipeline.py              # A owns; orchestrates 1→8
├── scripts/
│   ├── run_automl.py            # ONE-CLICK
│   ├── train_config.py
│   ├── build_portfolio.py
│   ├── run_baseline.py
│   └── make_figures.py
├── tests/
└── results/                     # gitignored except figures/ and ledger.csv
```

## 1.2 Shared types — `src/automl_image/types.py` (Member A writes this FIRST, day 1, hour 1)

Everything else is blocked on this file. It has **no dependencies beyond the standard library and pydantic/dataclasses**.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

Split = Literal["train", "val", "test"]

@dataclass(frozen=True)
class DatasetSpec:
    """Everything downstream needs to know about a dataset."""
    name: str
    root: Path
    n_classes: int
    n_train: int
    n_val: int
    n_test: int
    class_counts: dict[int, int]      # train split
    native_resolution: tuple[int, int] # median (h, w)
    channels: int
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]

@dataclass(frozen=True)
class MetaFeatures:
    """Cheap descriptors. Must be computable in < 60 s on any dataset."""
    n_samples: int
    n_classes: int
    samples_per_class_mean: float
    samples_per_class_min: int
    class_balance_entropy: float       # normalised to [0, 1]
    median_resolution: int             # min(h, w)
    resolution_std: float
    aspect_ratio_mean: float
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]
    mean_image_entropy: float
    probe_acc_1epoch: float            # from probe.py
    probe_loss_slope: float            # from probe.py
    probe_seconds: float
    def to_vector(self) -> list[float]: ...
    @staticmethod
    def feature_names() -> list[str]: ...

@dataclass(frozen=True)
class Fidelity:
    """One point in fidelity space. Higher = more expensive."""
    epochs: int
    resolution: int = 224
    data_fraction: float = 1.0
    def cost_weight(self) -> float:
        """Relative cost estimate, used by the cost-aware acquisition."""
        return self.epochs * (self.resolution / 224) ** 2 * self.data_fraction

@dataclass(frozen=True)
class TrainConfig:
    """A fully-specified point in the search space. Serialisable to YAML."""
    backbone: str
    ft_strategy: Literal["linear_probe", "last_block", "last_n_blocks", "full", "discriminative_lr"]
    n_unfrozen_blocks: int = 0          # conditional on ft_strategy == last_n_blocks
    augment: Literal["none", "flip_crop", "randaugment", "trivialaugment", "autoaugment"] = "randaugment"
    randaugment_n: int = 2              # conditional
    randaugment_m: int = 9              # conditional
    mixup: bool = False
    mixup_alpha: float = 0.2            # conditional
    cutmix: bool = False
    cutmix_alpha: float = 1.0           # conditional
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    lr_schedule: Literal["cosine", "step", "constant", "plateau"] = "cosine"
    label_smoothing: float = 0.0
    batch_size: int = 32
    dropout: float = 0.0
    seed: int = 0

    @classmethod
    def default(cls) -> "TrainConfig": ...
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig": ...
    def to_dict(self) -> dict[str, Any]: ...
    def hash(self) -> str:   # stable 12-char hash, excluding seed

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
    """The single return type of every training evaluation. THE key contract."""
    config: TrainConfig
    fidelity: Fidelity
    dataset: str
    val_acc: float                     # primary objective (maximise)
    val_macro_f1: float
    train_loss: float
    learning_curve: list[EpochRecord]  # length == fidelity.epochs (or fewer if stopped)
    n_params: int                      # cost objective input
    flops_g: float                     # cost objective input
    latency_ms: Optional[float]        # measured on reference device, may be None
    wallclock_seconds: float
    gpu_seconds: float
    peak_vram_mb: float
    device: str                        # e.g. "rtx5070-desktop"
    status: Literal["ok", "oom", "diverged", "timeout", "early_stopped", "error"]
    error: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    run_id: str = ""                   # assigned by runstore
```

### Failure contract
A training evaluation **never raises** into the optimizer. On OOM it retries once with `batch_size // 2` and gradient accumulation; on second failure it returns `RunResult(status="oom", val_acc=0.0, ...)`. On divergence (NaN loss) it returns `status="diverged", val_acc=0.0`. Member B's searchers must treat `val_acc=0.0` as a legitimate bad observation, not a crash.

## 1.3 The four cross-team interfaces

Only four function/class boundaries cross member lines. Everything else is internal.

### I1 — Training evaluation (A → B, A → C)

```python
# src/automl_image/train/loop.py
def evaluate_config(
    config: TrainConfig,
    fidelity: Fidelity,
    dataset: DatasetSpec,
    *,
    device: str = "cuda:0",
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,       # for rung promotion: continue, don't restart
    time_limit_s: float | None = None,
    lc_callback: "Callable[[EpochRecord], bool] | None" = None,
) -> RunResult:
    """Train `config` at `fidelity` and return the result.

    `lc_callback` is called after every epoch; if it returns True, training stops
    early and the result is returned with status="early_stopped". This is the hook
    Member B's learning-curve extrapolation plugs into.

    `resume_from` enables Successive-Halving promotion without re-training earlier
    epochs. If given, the epoch counter continues from the checkpoint.
    """
```

**This is the most important function in the repo.** Member A must have a *stub* version of it returning fake but plausible `RunResult`s (`--dry-run` mode) checked in on **day 1**, so B and C can build against it immediately.

### I2 — Search space (B → A, B → C)

```python
# src/automl_image/search/space.py
from ConfigSpace import ConfigurationSpace, Configuration

def build_space(
    dataset: DatasetSpec,
    meta: MetaFeatures,
    *,
    backbones: list[str] | None = None,
    vram_gb: float = 6.0,
) -> ConfigurationSpace:
    """Return the ConfigSpace, with conditionals and forbidden clauses that
    prevent VRAM-infeasible (backbone, batch_size, resolution) triples."""

def config_from_configuration(c: Configuration, seed: int = 0) -> TrainConfig: ...
def configuration_from_config(c: TrainConfig, space: ConfigurationSpace) -> Configuration: ...
```

### I3 — Warm-start / portfolio (C → B)

```python
# src/automl_image/portfolio/warm_start.py
def get_initial_design(
    meta: MetaFeatures,
    space: ConfigurationSpace,
    *,
    k: int = 8,
    exclude_dataset: str | None = None,   # leave-one-dataset-out guard
) -> list[TrainConfig]:
    """Top-k portfolio configs, ranked by meta-feature-similarity-weighted
    performance across offline benchmark datasets. Falls back to a
    space-filling (Sobol/LHS) design if no benchmark table exists."""

def get_surrogate_warm_start_data(
    meta: MetaFeatures, exclude_dataset: str | None = None
) -> list[RunResult]:
    """Prior observations to seed SMAC's random-forest surrogate."""
```

**Contract note:** `get_initial_design` must work — returning a space-filling design — *before* any offline benchmark exists. B is never blocked on C.

### I4 — Run store (A, B → C)

```python
# src/automl_image/runstore.py
class RunStore:
    def __init__(self, root: Path, experiment: str): ...
    def log(self, result: RunResult) -> str:        # returns run_id, appends to parquet
    def log_epoch(self, run_id: str, rec: EpochRecord) -> None:
    def load(self, **filters) -> "pandas.DataFrame":
    def learning_curves(self, **filters) -> "pandas.DataFrame":
    def ledger(self) -> "pandas.DataFrame":         # gpu_seconds by phase/member/machine
```

Every evaluation, everywhere — search, baselines, ablations, offline benchmark — goes through `RunStore.log`. **If it is not in the run store it does not exist**, and it cannot appear on the poster.

## 1.4 The orchestrator — `pipeline.py` (A owns)

```python
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
```

The `warm_start` / `lc_extrapolation` / `multi_objective` / `searcher` flags exist **because every ablation in `07_EXPERIMENT_PROTOCOL.md` is exactly one of these flags flipped**. Build them as flags from the start; do not retrofit.

Stages (matching plan §4):
1. `ingest.load(dataset_path)` → `DatasetSpec`
2. `splits.make_split(...)` → fixed stratified hold-out (or CV if meta says so)
3. `metafeatures.compute(...)` + `probe.run(...)` → `MetaFeatures` (budget 0.25 h)
4. `warm_start.get_initial_design(...)` → initial design
5. `search.run(...)` → list of `RunResult` (budget ~18–19 h, hard wall-clock stop)
6. `pareto.front(...)` + `selector.choose(...)` → final config(s)
7. `train.loop` full-fidelity retrain of top-k + `ensemble.soft_vote` + TTA (outside 24 h)
8. write `submission.csv`, `best_config.yaml`, `report.html`, `ledger.csv`

**Checkpointing:** stage boundaries write `out_dir/state.json`. Re-running with the same `out_dir` resumes. Non-negotiable — "pipeline crashes mid-24 h run" is on the risk register.

## 1.5 Output contract

```
results/<run_name>/
├── submission.csv          # exam format: id,label  (A verifies against the template repo)
├── best_config.yaml
├── pareto_front.csv        # config_hash, val_acc, n_params, flops_g, latency_ms, selected
├── runs.parquet            # every RunResult
├── learning_curves.parquet
├── ledger.csv              # phase, member, machine, gpu_seconds, wallclock_s, peak_vram
├── report.html             # fANOVA + ablation + cost plots
└── state.json              # resume marker
```
