# Run Store Schema

Agreed by A, B, C on day 1; enforced by `validate_result()` inside
`RunStore.log`. TODO(all): sign off in the first sync.

## `runs_*.parquet` columns

Flattened `RunResult`:

| Column | Dtype | Notes |
|---|---|---|
| `run_id` | str | assigned by RunStore |
| `config.*` | mixed | one column per TrainConfig field (`config.backbone`, `config.lr`, …) |
| `config_hash` | str | 12-char, seed-excluded |
| `fidelity.epochs` / `.resolution` / `.data_fraction` | int/int/float | |
| `seed` | int | |
| `dataset` | str | fashion \| flowers \| emotions \| skin_cancer |
| `val_acc`, `val_macro_f1`, `train_loss` | float | primary metric = val_acc (D8) |
| `n_epochs_run` | int | len(learning_curve) |
| `n_params`, `flops_g`, `latency_ms` | int/float/float? | cost objective inputs (D1) |
| `wallclock_seconds`, `gpu_seconds`, `peak_vram_mb` | float | compute pillar |
| `device`, `status`, `error`, `checkpoint_path` | str | status per failure contract |

Plus provenance (added by RunStore):

| Column | Notes |
|---|---|
| `experiment` | naming convention: `{dataset}__{arm}__seed{N}__{YYYYMMDD}` |
| `arm` | random \| hyperband \| bohb \| optuna_asha \| default \| strong_default |
| `phase` | `offline` \| `search` \| `baseline` \| `ablation` \| `final` |
| `member`, `machine` | from `.env.local` (`MEMBER`, `MACHINE_NAME`) |
| `timestamp` | ISO-8601 |
| `git_sha` | captured automatically; no SHA → not on the poster |

## `learning_curves_*.parquet` columns

`run_id`, `experiment`, `epoch`, `train_loss`, `val_loss`, `val_acc`,
`val_macro_f1`, `seconds`, `peak_vram_mb` (= flattened `EpochRecord`).

## Sharding

Each worker appends to `runs_{MACHINE_NAME}_{pid}.parquet`; `load()` globs and
concatenates. No database, no lock server.
