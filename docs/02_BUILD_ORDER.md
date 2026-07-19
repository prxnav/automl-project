# 02 — Build Order & Milestones

## 2.1 Dependency graph

```
                     ┌──────────────────────────┐
    HOUR 0           │ A0: types.py             │   ← blocks everyone
                     │ A0b: repo + env + CI     │
                     └────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  ┌───────────┐            ┌─────────────┐          ┌──────────────┐
  │ A1 ingest │            │ B1 space.py │          │ C1 runstore  │
  │ A2 splits │            │    (needs   │          │ C2 cost.py   │
  └─────┬─────┘            │  DatasetSpec)│         │ C3 pareto.py │
        ▼                  └──────┬──────┘          └──────┬───────┘
  ┌───────────────┐               │                        │
  │ A3 train loop │◄──────────────┘                        │
  │  = I1 contract│                                        │
  └─────┬─────────┘                                        │
        │  (A3-STUB available hour 4 → B and C unblocked)   │
        ▼                                                   │
  ┌───────────────┐   ┌──────────────┐   ┌──────────────────▼─────┐
  │ A4 metafeat.  │   │ B2 fidelity  │   │ C4 offline benchmark   │
  │ A5 probe      │──►│ B3 SMAC/BOHB │◄──│ C5 warm_start (I3)     │
  └───────────────┘   │ B4 scheduler │   └────────────────────────┘
                      │ B5 LC extrap.│
                      └──────┬───────┘
                             ▼
                   ┌──────────────────┐
                   │ A6 pipeline.py   │  ← M2 integration
                   └────────┬─────────┘
                            ▼
              ┌──────────────────────────┐
              │ C6 fANOVA / ablation     │
              │ C7 ledger / plots / stats│
              │ A7 ensemble + TTA        │
              └────────────┬─────────────┘
                           ▼
                   ┌────────────────┐
                   │ M4 FREEZE      │
                   │ 24 h dry run   │
                   └────────────────┘
```

## 2.2 Milestones (the only dates that matter)

| ID | Milestone | Definition of done (binary, testable) | Target (compressed) | Target (full) |
|---|---|---|---|---|
| **M0** | Skeleton green | `types.py` merged; repo + `environment.yml` install clean on all 3 machines; `pytest` runs; `evaluate_config` **stub** returns fake `RunResult`; `RunStore.log` writes parquet | Jul 18 EOD | Jul 07 |
| **M1** | Real training loop | `scripts/train_config.py --config configs/best_configs/default.yaml --dataset practice1` finishes and prints val acc; a `submission.csv` is produced and passes format check | Jul 19 midday | Jul 09 |
| **M2** | Search skeleton | `run_automl.py --searcher random --budget 0.5h` and `--searcher hyperband` both complete on a practice dataset and log ≥20 runs to the run store | Jul 19 EOD | Jul 12 |
| **M3** | Differentiators in | BOHB + Pareto tracking + warm-start + LC extrapolation all runnable as flags; each flag has an off-switch for ablation | Jul 20 EOD | Jul 17 |
| **M4** | **Method freeze + dry run** | Full 24 h contract dry-run on a practice dataset completes unattended, resumes after a forced kill, respects the wall-clock budget, emits all 7 output artefacts | **Jul 21** | Jul 21 |
| **M5** | Exam run | Frozen pipeline run on the exam training split; predictions generated; ≤3 submissions logged | Jul 22–27 | same |
| **M6** | Analysis complete | All baselines + 4 ablations × ≥3 seeds in the run store; significance tests run; all poster figures regenerate from `make_figures.py` | Jul 30 | same |
| **M7** | Submission | Repo frozen, README exact, poster PDF uploaded | **Aug 03, 23:59 CET** | same |

**M4 is the hard gate.** The exam requires the *final* run to respect the one-click + 24 h contract. Nothing may be added to the pipeline after M4 except bug fixes — and each bug fix requires re-running the dry run.

## 2.3 Compressed ordering (Jul 18–21), if you take Option COMPRESSED

**Day 1 (Jul 18)** — everyone in the same room or on a call.
- Hour 0–1: A merges `types.py` and the repo skeleton. B and C read it and object *now* if anything is missing.
- Hour 1–4: A builds the stub `evaluate_config` + `RunStore`. B builds `space.py` against `types.py`. C builds `cost.py` + `pareto.py` (pure functions, no dependencies — fully testable with synthetic data).
- Hour 4–EOD: A builds the real training loop. B wires SMAC3 Hyperband against the stub. C builds the ledger and the figure scripts against synthetic runs.
- **M0 gate before anyone sleeps.**

**Day 2 (Jul 19)** — M1 then M2.
- A: real loop working on all 3 GPUs, VRAM table measured, `submission.csv` format verified against the exam template repo.
- B: swap stub → real; launch the random-search and Hyperband baselines overnight on the practice datasets.
- C: launch the offline benchmark grid overnight (this is the long pole for the portfolio — start it as early as physically possible; it only needs A3, not B).

**Day 3 (Jul 20)** — M3.
- B: BOHB + LC extrapolation + cost-aware fidelity choice.
- C: portfolio warm-start from whatever the overnight benchmark produced, leave-one-dataset-out; Pareto + hypervolume tracking wired into the pipeline.
- A: pipeline orchestration, checkpoint/resume, ensemble + TTA.
- Evening: full integration; every ablation flag verified to actually change behaviour.

**Day 4 (Jul 21)** — M4. Freeze. Dry run. Kill it halfway on purpose and verify resume. Then stop touching the code.

Everything after Jul 21 is: the exam run, ablations on practice datasets (allowed — that's Phase-I analysis, and it uses the *frozen* code), significance testing, figures, poster.

## 2.4 What runs overnight, always

The three GPUs must never idle between Jul 18 and Jul 30. Maintain a queue in `docs/GPU_QUEUE.md`:

| Machine | Owner by default | Standing job |
|---|---|---|
| RTX 5070 desktop (12 GB) | B | Search experiments (BOHB / HB / random arms) |
| RTX 5070 laptop (~8 GB) | A | Training-loop dev, baselines, final retrains |
| RTX 3060 (6/12 GB) | C | Offline benchmark grid + latency reference measurements |

Nightly rule: whoever finishes last starts a queued job before closing the laptop. Log the job in `GPU_QUEUE.md` with expected finish time.
