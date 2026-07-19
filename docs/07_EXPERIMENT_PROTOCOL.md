# 07 — Experiment Protocol (the 80%-weighted pillar)

This is the document that determines the grade. Every experiment below must be *runnable as a flag on the existing pipeline* — that is why the ablation switches exist in `pipeline.py` from day one.

## 7.1 The experiment register

Maintain `docs/EXPERIMENTS.md` as a table. One row per experiment, filled *before* it runs:

| ID | Hypothesis | Arms | Datasets | Seeds | Budget/arm | Owner | Status | Result |
|---|---|---|---|---|---|---|---|---|

Pre-registering the expected outcome (one sentence) before running is cheap and makes the "clear hypotheses" claim on the poster true rather than retrofitted.

## 7.2 Baselines — mandatory (plan §6)

| ID | Baseline | What it isolates | Owner |
|---|---|---|---|
| BL1 | Default config, no search | Does search buy anything at all? | A |
| BL2 | Random search, **same wall-clock** as BOHB | The honest AutoML baseline | B |
| BL3 | Reference off-the-shelf AutoML (the exam's provided weak number) | The bar we must clear | B |
| BL4 | Strong tuned-timm default (expert recipe, no search) | The *hard* bar — beating this is the real claim | A |
| BL5 | Single-best portfolio config | Lower bound for algorithm selection | C |
| BL6 | Virtual-best (per-dataset oracle) | Upper bound for algorithm selection | C |

**Equal-budget discipline:** BL2 and every search arm get the *same wall-clock on the same machine class*. Comparing methods at unequal budgets invalidates the entire H1 claim. Record budget per arm in the register.

## 7.3 Ablations — one flag each

| ID | Ablation | Flag | Confirms | Owner |
|---|---|---|---|---|
| AB1 | No warm-start (cold start) | `--no-warm-start` | H3 | C |
| AB2 | No multi-fidelity (full-fidelity search, same wall-clock) | `--searcher random_full` / `--single-fidelity` | H1 | B |
| AB3 | No LC extrapolation | `--no-lc` | LC value + false-kill cost | B |
| AB4 | Single-objective (accuracy only) | `--single-objective` | H2 (does MO cost accuracy?) | C |
| AB5 | No cost-aware acquisition | `--no-cost-aware` | efficiency claim | B |

Each ablation is a poster figure bar. Each runs on **all practice datasets × ≥3 seeds**. AB5 is P2 if time is short — drop it before dropping AB1–AB4.

## 7.4 Variance & significance

- **Every reported number: mean ± std over ≥3 seeds.** No exceptions, including the "best config" number.
- **Method comparisons: paired Wilcoxon signed-rank** over (dataset × seed) pairs; the pairing is valid because the val split is fixed per (dataset, seed) — that is exactly why A2 makes the split a pure function.
- Report **n and p**, and state the power limitation honestly: with 3 datasets × 3 seeds, n=9. Say "we observe a consistent improvement; with n=9 we cannot claim strong significance" rather than reporting p=0.04 as if it settled anything. Graders reward calibration.
- Bootstrap 95% CIs on the accuracy differences as the primary evidence; the test is secondary.

## 7.5 Splits discipline (the leakage firewall)

1. Search **only ever** sees train → val.
2. The val split is fixed per (dataset, seed) and identical across configs.
3. Final config selection happens **offline, on validation** — never by looking at a test score.
4. Test labels are withheld; **≤ 3 submissions total**, tracked in `docs/SUBMISSIONS.md` with date, config hash, git sha, predicted-val-acc, actual score, and *why* the submission was made.
5. Any code path in `src/search/` or `src/portfolio/` that touches a test loader is a bug and has a test guarding against it.

## 7.6 Overfitting-to-practice-sets guard

The brief's stated goal is generalisation; the plan names practice-set overfitting as the #1 pitfall.

- All meta-components (portfolio, warm-start, LC predictor) are evaluated **leave-one-dataset-out**. A dataset never contributes to its own warm-start.
- Warm-start is a **prior on the initial design**, never a hard restriction on the space. Assert in code that the searcher can reach any point of the space regardless of warm-start.
- No design decision is justified by a single practice dataset. If a choice helps on one dataset and hurts on another, report both and pick the robust option.
- State this explicitly on the poster — the graders are looking for it.

## 7.7 The 24 h budget experiment

Beyond the exam run itself, run **one full dry-run** of the 24 h contract on a practice dataset (M4). Report:
- Stage-by-stage wall-clock vs the `budget.yaml` allocation.
- What was cut when the budget ran out and whether the incumbent had converged.
- Peak VRAM and GPU-hours consumed.
- Crash-resume verification: kill the process at hour *k*, resume, confirm the final result is equivalent.

This dry-run report *is* the compute-pillar evidence. It is worth more than any accuracy number.

## 7.8 Honest negative results to include (not hide)

The brief rewards good empirical practice; negative results are evidence of it. Actively report:
- Any fidelity dropped for low rank correlation.
- The LC pruner's false-kill rate on offline curves.
- If warm-start does not beat cold start on 3 datasets — say so, and explain the meta-training-set size limitation.
- If the multi-objective arm loses accuracy vs single-objective — quantify the price of the Pareto front.
- Any dataset where the default config beat the search.

A poster with four confirmed hypotheses and no caveats reads as unexamined. One with three confirmed, one refuted, and a clear explanation reads as science.
