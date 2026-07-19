# 04 — Member B: Search, Fidelity & Optimization

**Owns:** `src/automl_image/search/`, `configs/search_space.yaml`, `configs/budget.yaml`
**Poster decks you cite:** *BO for HPO* (surrogates, acquisition, exploration/exploitation, SMAC) · *Speedup* (multi-fidelity, SH, Hyperband, BOHB, LC extrapolation)
**Machine:** RTX 5070 desktop (12 GB) — the search box
**Owns hypothesis H1** and the headline poster figure (accuracy vs cumulative GPU-seconds).

You develop against Member A's **stub** `evaluate_config(dry_run=True)`. You should be able to write and test your *entire* stack without a GPU. Do that — it makes your work independent of A's schedule.

---

## B1 — Configuration space · **P0 · 3 h**

`src/automl_image/search/space.py`, implementing **I2**.

**The space** (from plan §5.3, bounds are yours to justify on the poster):

| HP | Type | Range / choices | Default | Conditional on |
|---|---|---|---|---|
| `backbone` | categorical | the 8 from D3 | `efficientnet_b0` | — |
| `ft_strategy` | categorical | linear_probe, last_block, last_n_blocks, full, discriminative_lr | `full` | — |
| `n_unfrozen_blocks` | int | 1–4 | 2 | `ft_strategy == last_n_blocks` |
| `augment` | categorical | none, flip_crop, randaugment, trivialaugment, autoaugment | `randaugment` | — |
| `randaugment_n` | int | 1–3 | 2 | `augment == randaugment` |
| `randaugment_m` | int | 5–15 | 9 | `augment == randaugment` |
| `mixup` | bool | — | False | — |
| `mixup_alpha` | float | 0.1–1.0 | 0.2 | `mixup == True` |
| `cutmix` | bool | — | False | — |
| `cutmix_alpha` | float | 0.5–2.0 | 1.0 | `cutmix == True` |
| `lr` | float, **log** | 1e-5 – 1e-2 | 3e-4 | — |
| `weight_decay` | float, **log** | 1e-6 – 1e-1 | 1e-4 | — |
| `warmup_ratio` | float | 0.0–0.2 | 0.05 | — |
| `lr_schedule` | categorical | cosine, step, constant, plateau | cosine | — |
| `label_smoothing` | float | 0.0–0.2 | 0.0 | — |
| `batch_size` | ordinal | 16, 32, 64, 128 | 32 | — |
| `dropout` | float | 0.0–0.5 | 0.0 | — |

**Requirements**
- Use **ConfigSpace conditionals** natively (`EqualsCondition`, `GreaterThanCondition`) — this is a graded item ("configuration spaces with conditional hyperparameters", *Basics* deck). Do not fake conditionals with ignored values.
- **Forbidden clauses for VRAM feasibility.** Build a small table `configs/vram_table.yaml` from a measurement script (`scripts/measure_vram.py`, run once per machine per backbone × batch × resolution) and generate `ForbiddenAndConjunction` clauses so the optimizer never proposes an infeasible config. Alternative if that's too rigid: allow it but auto-enable gradient accumulation in A's loop. **Locked: forbid the truly impossible, grad-accumulate the marginal.**
- Serialise the whole space to `configs/search_space.yaml` with a `justification:` string per hyperparameter. The poster requires bounds *and* defaults to be motivated — write the justification when you set the bound, not the night before.
- `TrainConfig.default()` and the ConfigSpace default configuration must be **identical**. Assert this in a test.

---

## B2 — Fidelity design + the rank-correlation experiment · **P0 · 2 h code + overnight compute**

`src/automl_image/search/fidelity.py`
```python
@dataclass
class FidelitySchedule:
    dimension: Literal["epochs", "epochs_resolution", "epochs_subset"]
    rungs: list[Fidelity]        # e.g. 1, 3, 9, 27 epochs
    eta: int = 3
def build_schedule(meta: MetaFeatures, budget: BudgetGuard) -> FidelitySchedule
```
- Rungs: `eta=3`, `min_budget=1` epoch, `max_budget=27` epochs (or fewer if the dataset is tiny — derive `max_budget` from meta-features: `max(9, min(30, 3000 // (n_samples//1000 + 1)))`, tune this rule and document it).
- Combined fidelity: rung *i* uses resolution `[96, 128, 160, 224][i]` **only if** the M2 correlation experiment justifies it (D4).

### The experiment (E-RANK) — this is a poster figure, do it early
1. Sample 30 configs uniformly from the space.
2. Evaluate every one at every rung on each practice dataset (this is *also* the seed of C's offline benchmark table — coordinate with C so it runs once, not twice).
3. Compute **Spearman rank correlation** between rung-*i* val acc and max-rung val acc, per dataset, per fidelity dimension.
4. Report the correlation table + a scatter. **If a fidelity's ρ < 0.6, drop it** and say so on the poster. Reporting a dropped fidelity is a *strength* — it's the "each fidelity choice brings its own nuance" ask, evidenced.

This experiment directly validates the assumption the entire multi-fidelity design rests on. It is the highest-value-per-GPU-hour thing you will run.

---

## B3 — Searcher abstraction + the four arms · **P0 · 5 h**

`src/automl_image/search/base.py`
```python
class Searcher(Protocol):
    def run(self, evaluate: Callable[[TrainConfig, Fidelity], RunResult],
            space: ConfigurationSpace,
            schedule: FidelitySchedule,
            budget: BudgetGuard,
            *,
            initial_design: list[TrainConfig] | None = None,
            warm_start_data: list[RunResult] | None = None,
            lc_pruner: "LCPruner | None" = None,
            store: RunStore) -> list[RunResult]: ...
    def save_state(self, path: Path) -> None: ...
    def load_state(self, path: Path) -> None: ...
```

Implement **four arms behind this one interface** — they are the ablation and the H1 comparison:

| Arm | Class | Notes |
|---|---|---|
| `random` | `RandomSearcher` | Full fidelity, same wall-clock. The honest AutoML baseline. |
| `hyperband` | `SMACHyperbandSearcher` | SMAC3 `Hyperband` intensifier + **random** config selector |
| `bohb` | `SMACBOHBSearcher` | SMAC3 `MultiFidelityFacade` (RF surrogate + Hyperband intensifier) — **the primary method** |
| `optuna_asha` | `OptunaSearcher` | TPE + `HyperbandPruner`/ASHA; the cross-check (D2) |

**Implementation notes**
- SMAC3 2.x: use `MultiFidelityFacade` with `Scenario(configspace, walltime_limit=..., min_budget=..., max_budget=...)`. Your target function signature is `f(config, seed, budget) -> float | dict`. Convert `budget` (SMAC's float) → `Fidelity` via `FidelitySchedule`.
- **Return the negative accuracy** (SMAC minimises). Wrap this conversion in one place and unit-test the sign. Sign errors here are the classic 3-hour bug.
- **Every** evaluation, from every arm, is logged via `store.log(result)` with `experiment=` naming that identifies the arm, seed, and dataset. C's analysis reads only from there.
- **Checkpoint the searcher state every rung** to `out_dir/search_state/`. `save_state`/`load_state` must survive process death — SMAC3's own run-history persistence plus your own metadata.
- **Parallelism:** one worker per GPU. Use SMAC3's dask integration or, if that fights you, a simpler file-lock-based job queue (a `pending.jsonl` + `claimed/` dir). **Schedule by wall-clock, not by count** — the three machines are ~2× apart in speed (time-heterogeneity, flagged in the *Basics* deck; mention it on the poster).
  - *Pragmatic fallback if distributed BOHB eats your day:* run each arm/seed as an independent single-GPU process on a different machine (embarrassingly parallel across seeds and arms). This is 90% of the benefit for 10% of the risk. **Take the fallback if parallel BOHB isn't working by end of Day 2.**

---

## B4 — Learning-curve extrapolation & early termination · **P1 · 3 h**

`src/automl_image/search/lc_extrapolation.py`
```python
class LCPruner:
    def __init__(self, incumbent_acc: float, min_epochs: int = 3, confidence: float = 0.95,
                 model: Literal["pow4","mmf","exp"] = "pow4"): ...
    def should_stop(self, curve: list[EpochRecord], max_epochs: int) -> bool
    def predict_final(self, curve, max_epochs) -> tuple[float, float]  # (mean, std)
```
- Fit a parametric saturating family to the partial val-acc curve with `scipy.optimize.curve_fit`:
  - `pow4`: `a - b * x**(-c)` · `mmf`: `(a*b + c*x**d)/(b + x**d)` · `exp`: `a - b*exp(-c*x)`
  - Fit all three, weight by fit residual, take the ensemble mean and spread as uncertainty.
- Stop when `P(final < incumbent) > confidence`, using the fit spread as the uncertainty estimate. Never stop before `min_epochs=3` points.
- Plugs into A's `lc_callback` — the pruner is created per-run by the searcher with the current incumbent.
- **Honesty requirement (plan §5.6):** log every pruning decision to `results/pruning_log.csv` with the predicted final acc. Then on the *offline benchmark curves* (C's table, which are complete), replay the pruner and measure **how often it would have killed a config that ended up in the top 10%**. Report that false-kill rate on the poster. Negative results are explicitly rewarded.

---

## B5 — Cost-aware fidelity/acquisition · **P1 · 2 h**

- Track `accuracy_gain_per_gpu_second` per rung and per backbone class in the run store.
- Simplest defensible implementation: **cost-weighted acquisition** — divide SMAC's expected improvement by the predicted evaluation cost (`fidelity.cost_weight() × per-backbone seconds/epoch, measured`). This is the "EI per unit cost" (EIpu) idea; cite it and say plainly that it's a cost-normalised acquisition, not a full multi-fidelity BO.
- Also: **budget-aware rung admission** — with `T` seconds left, do not start a rung whose estimated cost exceeds `T - safety_margin`. Coordinates with A's `BudgetGuard`.
- Ablation flag: `--no-cost-aware`.

---

## B6 — Baseline runs · **P0 · your compute, C's analysis**

Produce, on every practice dataset, ≥3 seeds each:
1. **Default config**, no search (uses A's `train_config.py`).
2. **Random search**, identical wall-clock budget to BOHB.
3. **Hyperband** (random sampling, multi-fidelity) — isolates "does the model-based part help?"
4. **BOHB** — the method.
5. **BOHB + warm-start** — with C's portfolio.
6. **Strong tuned-timm default** (D7) — a hand-tuned recipe you'd write as an expert.

Every one goes to the run store with consistent `experiment` naming: `{dataset}__{arm}__seed{N}__{YYYYMMDD}`. C cannot do the significance tests without this naming being consistent, so agree the schema with C on day 1 and write it in `docs/RUNSTORE_SCHEMA.md`.

---

## B7 — Stretch · **P2**
- **DEHB** as a fifth arm (plan §5.5.5). Only if M4 passed and Phase II is calm.
- Meta-learned LC predictor (with C): train a small GBM on the offline curves to predict final accuracy from `(meta-features, config, partial curve)`. This is a genuine creativity point and ties H3 to §5.6 — but it is P2 for a reason.

## B8 — Your poster content
- Search-space table with justified bounds and defaults.
- The **rank-correlation figure** (E-RANK) validating the multi-fidelity assumption.
- The **headline figure**: mean val accuracy vs cumulative GPU-seconds, one line per arm, shaded ±1 std over seeds. This one plot carries H1 and half the compute pillar.
- LC-pruner false-kill rate (honest negative result).
