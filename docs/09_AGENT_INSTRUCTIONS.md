# 09 — Instructions for the Coding Agent

Paste this file (or link it) at the start of every agent session. It is the standing context; the per-task prompt is the variable part.

## 9.1 Standing context

You are implementing the AutoML image-classification system described in `docs/`. Read, in order: `01_ARCHITECTURE_AND_INTERFACES.md` (the contract), `02_BUILD_ORDER.md` (where this task sits), and the member file for whoever's task you are doing.

**Non-negotiables:**
1. `src/automl_image/types.py` is the shared vocabulary. Never redefine `TrainConfig`, `RunResult`, `Fidelity`, `MetaFeatures`, or `DatasetSpec` locally; import them.
2. Never change a cross-team interface (I1–I4 in §1.3) without saying so explicitly at the top of your output. Silently changing a signature breaks two other people's work.
3. Every training evaluation returns a `RunResult` and logs through `RunStore`. If it isn't in the run store, it doesn't exist.
4. Training evaluations never raise into the optimizer — follow the failure contract in §1.2 (`status="oom"|"diverged"|"error"`, `val_acc=0.0`).
5. Test labels are never read outside the final prediction step.
6. Ablation switches are constructor/CLI flags, present from the first version. Never hard-code a component "on".
7. Pin every new dependency in `environment.yml` with an exact version, and say why it's needed.

## 9.2 Task prompt template

```
TASK: <task ID from the member file, e.g. B4>
FILE(S): <exact paths>
CONTEXT: docs/01_ARCHITECTURE_AND_INTERFACES.md §<x>, docs/0<n>_MEMBER_<X>_*.md task <ID>
DEPENDS ON: <task IDs that must already exist; assume their interfaces, not their internals>
DELIVERABLE:
  - implementation
  - unit tests in tests/test_<module>.py
  - docstrings stating the lecture concept implemented (needed for the poster)
CONSTRAINTS: <VRAM / runtime / no-GPU-in-tests / etc.>
DONE WHEN: <the binary criterion from the member file>
```

## 9.3 Code conventions

- Python 3.11, full type hints, `ruff` clean.
- Dataclasses for data, `Protocol` for interfaces, no inheritance hierarchies deeper than one level.
- **No global state.** Seeds, devices, and paths are passed explicitly. Three machines and parallel workers make hidden globals a debugging nightmare.
- Logging via `logging`, not `print` (except in `scripts/`, where `rich` is fine).
- Every function that consumes GPU time measures and returns its own cost. The compute pillar depends on instrumentation being ubiquitous, not bolted on later.
- Docstrings on public functions include a `Lecture:` line naming the deck and concept — e.g. `Lecture: Speedup — successive halving with eta=3`. At poster time this makes the week-citation table a grep, not an archaeology project.

## 9.4 Testing rules

- Tests are **CPU-only** and complete in under 60 s total.
- Any test needing data uses `tests/fixtures/make_fake_dataset.py`. No test depends on a real dataset.
- Any test needing a training result uses `evaluate_config(dry_run=True)`.
- Numerical utilities (Pareto, hypervolume, knee, rank correlation, LC fits) get hand-computed test cases. These are where silent wrongness hides — a wrong hypervolume produces a plausible-looking plot.

## 9.5 What to do when blocked

- Depend on an interface that doesn't exist yet → write the stub matching §1.3, mark it `# STUB: owned by <member>, task <ID>`, and note it in your output.
- The interface as specified doesn't work → **stop and report**, with a proposed alternative signature. Do not improvise a different one.
- A task seems to need more than ~300 lines → say so and propose a split.

## 9.6 Explicit anti-patterns

| Don't | Because |
|---|---|
| Re-download `timm` weights inside the pipeline | Wastes the 24 h budget on something explicitly excluded from it |
| Restart training from scratch on rung promotion | Throws away most of the multi-fidelity speedup |
| Use `print` for metrics | Metrics must land in the run store to be analysable |
| Add a feature that isn't a P0/P1 task | Scope creep is the #1 named risk |
| Silently catch and ignore exceptions | Failures must appear as `status` in the run store, not vanish |
| Hand-edit a figure | Every figure regenerates from `make_figures.py` or it isn't reproducible |
| Compare methods by number of evaluations | The three GPUs differ ~2× in speed; compare GPU-seconds |
| Tune anything on the test set | Instant, unrecoverable grade damage |

## 9.7 Suggested agent task sequence (mirrors `02_BUILD_ORDER.md`)

```
A0  types.py + repo skeleton + CI                 [no deps]
C1  runstore.py                                    [A0]
A3s evaluate_config stub (dry_run)                 [A0]
B1  space.py + search_space.yaml                   [A0]
C2  cost.py + backbone_costs.yaml                  [A0]
C3  pareto.py + selector.py                        [A0]
A1  ingest.py       A2 splits.py                   [A0]
A4  augment.py                                     [A0]
A3  evaluate_config real                           [A1,A2,A4,B1]
B2  fidelity.py + E-RANK experiment script         [A3,B1]
B3  searchers (random, hyperband, bohb, optuna)    [A3s,B1,B2,C1]
A5  metafeatures.py + probe.py                     [A1,A3]
C4  benchmark_table.py + build_portfolio.py        [A3,C1]
B4  lc_extrapolation.py                            [A3,B3]
C5  warm_start.py                                  [C4,A5]
B5  cost-aware acquisition                         [B3,C2]
A6  pipeline.py + scripts/                         [all above]
A7  ensemble.py + TTA + submission writer          [A3,C3]
C6  fanova.py + ablation.py                        [B3,C1]
C7  ledger.py + stats.py                           [C1]
C8  make_figures.py                                [C1,C3,C6,C7]
```

Each line is one agent session. Do not batch more than two into a single session — the review cost outweighs the speed gain, and interface drift compounds.
