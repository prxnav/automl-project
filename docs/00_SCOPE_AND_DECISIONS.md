# 00 — Scope & Locked Decisions

## 0.1 What we are building, in one sentence

A one-command AutoML system for image classification that runs multi-fidelity Bayesian optimization (BOHB) over `pretrained backbone × fine-tuning strategy × augmentation × optimizer HPs`, warm-started from an offline portfolio, tracking `(accuracy, inference cost)` as a multi-objective problem, and emitting `submission.csv` + a Pareto front + an interpretability/cost report — inside 24 h of search on a new dataset.

## 0.2 Deliverables (what "done" means)

| Deliverable | Owner | Deadline |
|---|---|---|
| Runnable repo, pinned deps, `README` with exact commands | A (integrator) | Aug 03 |
| `python scripts/run_automl.py --dataset <path> --budget 24h --out results/` works end-to-end | A + B | Jul 21 (dry-run), Aug 03 (final) |
| `python scripts/train_config.py --config <yaml>` reproduces a reported number | A | Jul 21 |
| Exam-dataset predictions, ≤3 submissions | B (runs), all (decide) | Phase II |
| Poster with method, results, compute report, week citations, individual contributions | C leads, all write own section | Aug 03 |
| Cost ledger covering every GPU-hour spent | C | continuous |

## 0.3 Decisions LOCKED (Section 14 of the plan, resolved)

These were "open decisions". They are now closed so that parallel work can start. Changing one requires all three members to agree in writing in `docs/DECISION_LOG.md`.

| # | Decision | Locked choice | Rationale |
|---|---|---|---|
| D1 | Cost objective for multi-objective | **Primary: parameter count + FLOPs (deterministic).** Secondary: measured latency on one fixed reference device (the RTX 3060), logged for every config but not optimized against. | Deterministic objective keeps the Pareto front reproducible across three heterogeneous machines; latency is still reported so the "real deployment concern" story survives. |
| D2 | Primary HPO tool | **SMAC3 primary, Optuna cross-check.** Both behind one `Searcher` interface. | Course-aligned; Optuna gives ASHA + NSGA-II for free as a comparison arm. |
| D3 | Backbone portfolio (8) | `resnet18`, `resnet50`, `efficientnet_b0`, `efficientnet_b3`, `convnext_tiny`, `mobilenetv3_large_100`, `vit_tiny_patch16_224`, `vit_small_patch16_224` | Spans ~1.5M→50M params and the accuracy/cost spectrum; all available pretrained in `timm`; all fit 6 GB with AMP at batch 32 / 224 px except `efficientnet_b3` and `vit_small` which need grad accumulation. |
| D4 | Fidelity set | **Epochs primary** (rungs 1/3/9/27). **Resolution secondary** (96→160→224), enabled only if rank-correlation ≥ 0.6 in the M2 experiment. Data-subset = fallback for datasets > 100k images. | Plan §5.4; the correlation measurement is itself a poster figure. |
| D5 | Stretch goals | **In:** meta-learned LC predictor *only if* P0+P1 green by Jul 26. **Out:** DAC learned LR policy (replaced by a *heuristic* reduce-on-plateau policy framed in MDP language — cheap, still cites the deck). **Out:** LLM-API selector unless a member has spare time in Phase II. | Scope creep is the #1 listed risk. |
| D6 | Validation strategy | **Stratified hold-out 80/20 during search**, fixed seed, identical split for every config (paired comparisons). **5-fold CV** only for final reported numbers and for datasets with < 2000 training images (decided automatically by meta-features). | Plan §5.1. |
| D7 | Baseline framing | **Build from components**, and additionally report a *strong tuned-timm default* as a baseline we must beat. No Auto-PyTorch dependency. | Fewer moving parts; the strong baseline is cheap to produce and makes the improvement claim honest. |
| D8 | Primary search metric | **Top-1 accuracy on held-out val.** Macro-F1 logged alongside for imbalanced datasets. | Exam metric. |
| D9 | Experiment tracking | **CSV/Parquet run-store as the source of truth**, W&B as an optional viewer. | The cost ledger and all poster figures must be regenerable offline without a network. |

## 0.4 Hypotheses → who owns the evidence

| Hypothesis | Owner of the experiment | Poster figure |
|---|---|---|
| **H1** multi-fidelity + cost-aware search beats default & random search per unit compute | **B** | accuracy vs cumulative GPU-seconds |
| **H2** non-trivial accuracy/cost Pareto trade-off exists | **C** | Pareto front + hypervolume-over-time |
| **H3** meta-feature portfolio warm-start beats cold start, without practice-set overfitting | **C** (portfolio) + **B** (integration into search) | time-to-target-accuracy, warm vs cold; single-best vs virtual-best bounds |
| **H0** (implicit) low-fidelity ranks predict high-fidelity ranks | **A** produces the curves, **B** analyses | rank-correlation scatter per fidelity |

## 0.5 Non-goals (say no to these)

- Training backbones from scratch. Pretrained only.
- Neural architecture search beyond the 8-backbone categorical choice.
- Chasing leaderboard accuracy at the expense of ablations. The grade is 80% methodology.
- Any dependency that cannot be pinned in `environment.yml`.
- More than 3 test submissions. Ever.

## 0.6 Role summary

| Member | Title | Owns (dirs) | Owns (lecture decks for poster) |
|---|---|---|---|
| **A** | Data & Training Core / Integrator | `src/data/`, `src/train/`, `src/pipeline.py`, `scripts/` | Basics of HPO (validation & resampling, config-space plumbing) |
| **B** | Search & Fidelity | `src/search/` | BO for HPO; Speedup (SH/HB/BOHB, LC extrapolation) |
| **C** | Meta-learning, Multi-objective & Analysis | `src/portfolio/`, `src/objectives/`, `src/analysis/` | Algorithm Selection; Multi-criteria; Interpretability |

Each member writes their own "individual contributions" poster paragraph and keeps `docs/CONTRIBUTIONS.md` updated weekly — individual grades can differ, so this is graded evidence, not admin.
