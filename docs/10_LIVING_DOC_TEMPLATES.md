# 10 — Templates for the Living Docs

Split these into separate files under `docs/` on day 1. They are cheap to maintain and expensive to reconstruct.

---

## `docs/DECISION_LOG.md`
```
| Date | Decision | Changed from | Reason | Agreed by |
|---|---|---|---|---|
| 2026-07-18 | Cost objective = params/FLOPs primary, latency logged only | (open, plan §14.1) | Reproducibility across 3 heterogeneous machines | A, B, C |
```

## `docs/CONTRIBUTIONS.md`
```
## Member A — <name>
### Week of Jul 18
- Implemented: types.py, ingest, splits, training loop (commits: abc123..def456)
- Ran: BL1 and BL4 baselines on ds1–ds3, 3 seeds
- Reviewed: PR #12 (B, BOHB searcher)
```
Update weekly. The poster's individual-contributions section is generated from this.

## `docs/EXPERIMENTS.md`
```
| ID | Hypothesis | Expected outcome (pre-registered) | Arms | Datasets | Seeds | Budget/arm | Owner | Status | Result |
|---|---|---|---|---|---|---|---|---|---|
| E-RANK | fidelity assumption | ρ ≥ 0.7 for epochs, ≥ 0.6 for resolution | 30 configs × 4 rungs | ds1–3 | 2 | — | B | running | |
```

## `docs/SUBMISSIONS.md`
```
| # | Date | Git SHA | Config hash | Val acc | Test score | Why this submission |
|---|---|---|---|---|---|---|
| 1 | | | | | | Pipeline's own knee-point + top-3 ensemble |
```
Max 3 rows. Ever.

## `docs/GPU_QUEUE.md`
```
| Machine | Job | Owner | Started | Expected finish | Status |
|---|---|---|---|---|---|
| rtx3060 | offline benchmark grid ds1 | C | Jul 18 23:00 | Jul 19 09:00 | running |
```

## `docs/LEDGER.md`
Snapshot of `analysis/ledger.py` output at each milestone:
```
| Phase | Member | Machine | GPU-hours | Wall-clock h | Peak VRAM GB |
|---|---|---|---|---|---|
| offline_benchmark | C | rtx3060 | | | |
| baselines | B | rtx5070d | | | |
| search_dev | B | rtx5070d | | | |
| ablations | B,C | all | | | |
| exam_run | B | rtx5070d | | | |
| **total** | | | | | |
```

## `docs/RUNSTORE_SCHEMA.md`
Agreed by A, B, C on day 1. Column list, dtypes, the `experiment` naming convention (`{dataset}__{arm}__seed{N}__{YYYYMMDD}`), and the `phase` enum (`offline|search|baseline|ablation|final`).

## `docs/DATA_FORMAT.md`
Written by A after reading the course template repo, before writing `ingest.py`: directory layout, label encoding, exact `submission.csv` header and dtype, the reference baseline number.

## `docs/BENCHMARK_STATUS.md`
C updates daily during the offline grid:
```
| Dataset | Configs done | Rungs done | Seeds | Curves saved | ETA |
```

## `docs/DRYRUN_REPORT.md`
Produced at M4 by A. Stage timings vs `budget.yaml`, GPU-hours, peak VRAM, incumbent trajectory, what was cut when the budget expired, crash-resume verification result. This report is poster material for the compute pillar.
