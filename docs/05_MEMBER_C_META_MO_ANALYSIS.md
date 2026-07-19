# 05 — Member C: Meta-learning, Multi-objective & Analysis

**Owns:** `src/automl_image/portfolio/`, `objectives/`, `analysis/`, `runstore.py`, all poster figures
**Poster decks you cite:** *Algorithm Selection* (meta-features, portfolios, single-best vs virtual-best) · *Multi-criteria* (Pareto, hypervolume, a-posteriori selection) · *Interpretability* (fANOVA, ablation)
**Machine:** RTX 3060 — also the **reference device for all latency measurements** (D1); never change this machine mid-project or the latency numbers become incomparable.
**Owns hypotheses H2 and H3, the cost ledger, and — because you own the figures — the visual half of the poster.**

Your work splits cleanly into: things that need no GPU and no teammates (`objectives/`, `analysis/`, `runstore.py` — build these on day 1 with synthetic data), and things that need the offline benchmark (`portfolio/`).

---

## C1 — Run store · **P0 · 2 h · blocks the analysis of everything**

`src/automl_image/runstore.py`, implementing **I4**.

- Backing store: append-only **Parquet** per experiment (`results/<experiment>/runs.parquet`, `learning_curves.parquet`) with a **file lock** — three machines and multiple workers write concurrently. Simplest robust design: each worker appends to its own `runs_{worker_id}.parquet` shard; `load()` globs and concatenates. Avoid a database.
- Schema is the flattened `RunResult` plus: `experiment`, `arm`, `seed`, `dataset`, `member`, `machine`, `timestamp`, `git_sha`, `phase` (`offline|search|baseline|ablation|final`).
- `git_sha` is captured automatically. When you later ask "which commit produced this number", you will need it.
- Write `docs/RUNSTORE_SCHEMA.md` with A and B on day 1 and make the naming convention mandatory.
- Optional W&B mirror behind `--wandb` — **never** a hard dependency (D9).

---

## C2 — Cost objectives · **P0 · 1.5 h**

`src/automl_image/objectives/cost.py`
```python
def n_params(model) -> int
def flops_g(model, resolution: int) -> float          # fvcore or ptflops
def measure_latency_ms(model, resolution, *, device, batch_size=1,
                       warmup=20, iters=100) -> float # median, not mean
def cost_scalar(result: RunResult, mode="params") -> float
```
- Build a **static table** `configs/backbone_costs.yaml`: for each of the 8 backbones × {96,128,160,224} px → params, FLOPs, measured latency on the RTX 3060. Measure it **once** (30 min), then cost lookups inside the 24 h run are free. This is a real budget optimisation and worth a poster sentence.
- Latency: median over 100 iterations after 20 warmup iters, `torch.cuda.synchronize()` around each, batch size 1 (deployment-realistic) *and* 32 (throughput-realistic) — report both, optimise against neither (D1).
- Ensembles: cost is additive. The final ensemble gets its own point on the Pareto plot.

---

## C3 — Pareto machinery · **P0 · 2 h**

`src/automl_image/objectives/pareto.py`
```python
def is_dominated(a: tuple[float, float], b: tuple[float, float]) -> bool
def pareto_front(points: np.ndarray, maximize: tuple[bool, bool]) -> np.ndarray  # indices
def hypervolume(front: np.ndarray, reference: tuple[float, float]) -> float
def hypervolume_over_time(results: list[RunResult], reference) -> pd.DataFrame
def knee_point(front: np.ndarray) -> int
```
- Objectives: **maximise** `val_acc`, **minimise** `cost_scalar`. Reference point for hypervolume: `(acc=0.0, cost=max_cost_in_portfolio)` — fixed and documented so hypervolume is comparable across runs and datasets.
- `knee_point`: maximum distance to the line joining the two extreme points of the normalised front (the standard construction). This is the **default a-posteriori selection rule** (plan §5.8).
- Pure functions, no I/O, 100% unit-tested with hand-computed examples. You can and should finish this on day 1 without touching a GPU.

`src/automl_image/portfolio/selector.py`
```python
def choose(front: pd.DataFrame, *, rule="knee", constraint: dict | None = None,
           top_k: int = 3) -> list[TrainConfig]
```
- Rules: `knee` (default), `max_acc`, `constrained` (e.g. `{"max_latency_ms": 10}` → best accuracy meeting it).
- The `constrained` rule *is* the brief-sanctioned user-interaction window (§5.12). Implement it as a CLI flag; that's sufficient to claim the concept honestly. An LLM-API selector is P2 and only after everything else is green.

---

## C4 — Offline benchmark table · **P0 · 1 h code + a lot of overnight GPU**

`scripts/build_portfolio.py` / `src/automl_image/portfolio/benchmark_table.py`

- Grid: **the 30 sampled configs from B's E-RANK experiment** (share the runs — coordinate with B, run it once) **plus** the 8 backbones at default HPs, **plus** ~40 additional random configs, across **all practice datasets**, at **all fidelity rungs**, ≥2 seeds. All curves saved.
- **Start this the moment A's real training loop passes M1.** It is pure background compute, it blocks the portfolio (H3), and GPU-hours are the one thing you cannot parallelise on the last day. Nothing else you do has a longer lead time.
- Output: `results/offline_benchmark/` (parquet) + `docs/BENCHMARK_STATUS.md` with a progress table you update daily.
- This offline compute is **outside the 24 h budget** (plan §7) but **must still be reported** on the poster. Ledger it.

---

## C5 — Warm-start / portfolio / algorithm selection · **P1 · 4 h · H3**

`src/automl_image/portfolio/warm_start.py`, implementing **I3**.

1. **Portfolio construction (greedy)**: from the benchmark table, greedily pick the config that most improves *average rank across datasets* given the already-picked set — the standard greedy portfolio construction from the *Algorithm Selection* deck. Size `k=8`.
2. **Meta-feature similarity**: standardise meta-feature vectors, cosine or Euclidean k-NN over the offline datasets; re-weight the portfolio ranking by similarity to the new dataset. With only 3 practice datasets this is thin — **say so on the poster**. Honest framing: "with 3 meta-training datasets the similarity weighting is under-determined; we therefore keep the warm-start a *prior* over the initial design rather than a selector, and we report the leave-one-dataset-out result with that caveat."
3. **Leave-one-dataset-out is mandatory** (plan §5.7, the #1 pitfall). `exclude_dataset` must be honoured everywhere, and a unit test must assert that a dataset never appears in its own warm-start source data.
4. **Bounds analysis** (this is the clean, standard, poster-ready result):
   - **Single-best**: the one config with best mean performance across datasets — the lower bound of what selection buys you.
   - **Virtual-best (oracle)**: per-dataset best config — the upper bound.
   - Your selector must land between them. Report where. If it lands at single-best, that's an honest negative result and still scores.
5. `get_surrogate_warm_start_data` → prior observations for SMAC's RF surrogate (B consumes).
6. **Fallback path**: with no benchmark table, return a Sobol/LHS space-filling design. B must never be blocked on you.

---

## C6 — Interpretability: fANOVA + ablation · **P1 · 3 h**

`src/automl_image/analysis/fanova.py`
- Run fANOVA on SMAC3's run history (SMAC ships fANOVA support; else use the standalone `fanova` package on the exported run history).
- Output: per-hyperparameter and pairwise-interaction importance, **per dataset**. The interesting claim is the *variation across datasets* ("augmentation mattered only on dataset X") — tie it back to meta-features. That connection is what turns a plot into a finding.
- Caveat to state: fANOVA needs enough evaluations; report how many the analysis used, and don't run it on a 20-evaluation run.

`src/automl_image/analysis/ablation.py`
- **Greedy ablation path** (Fawcett & Hoos, in the *Interpretability* deck): start at the default config, at each step flip the single hyperparameter that most improves toward the incumbent, until you reach the incumbent. Produces the "which changes mattered, in order" staircase plot.
- Distinguish this from the **component ablations** (§07) — same word, different things. Label them clearly on the poster: "hyperparameter ablation path" vs "system component ablation".

---

## C7 — Compute ledger & statistics · **P0 · 2 h**

`src/automl_image/analysis/ledger.py`
```python
def build_ledger(store: RunStore) -> pd.DataFrame   # phase × member × machine × gpu_hours × wallclock × peak_vram
def budget_report(run_dir: Path) -> pd.DataFrame    # the 24h run, stage by stage vs budget
```
- Must account for **every** GPU-hour: offline benchmark, dev runs, baselines, ablations, the exam run. The brief asks for it; "we tracked it from day one" is worth more than a large or small number.
- Add a `docs/LEDGER.md` snapshot updated at each milestone so nobody has to reconstruct it on Aug 2.

`src/automl_image/analysis/stats.py`
```python
def paired_wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]
def bootstrap_ci(x, n=10000, alpha=0.05) -> tuple[float, float]
def summarize(store, group_cols) -> pd.DataFrame   # mean ± std over seeds
```
- Every reported number is **mean ± std over ≥3 seeds**. Every method comparison is a **paired Wilcoxon signed-rank** over (dataset × seed) pairs. State n and the p-value; with 3 datasets × 3 seeds, n=9 — say plainly that this is low power rather than over-claiming significance. Under-claiming reads as rigor; over-claiming reads as sloppiness.

---

## C8 — Figures · **P1 · 3 h · `scripts/make_figures.py`**

One script regenerates **every** poster figure from the run store. No hand-edited plots, ever.

| Fig | Content | Hypothesis |
|---|---|---|
| F1 | Accuracy vs cumulative GPU-seconds, one line per arm, ±1 std | H1 |
| F2 | Low- vs high-fidelity rank correlation scatter + ρ table | multi-fidelity assumption |
| F3 | Pareto front (acc vs params/FLOPs), all evaluated configs greyed, front highlighted, knee marked, ensemble point marked | H2 |
| F4 | Hypervolume over wall-clock time, per arm | H2 |
| F5 | Warm-start vs cold-start: time-to-target-accuracy, with single-best and virtual-best bound lines | H3 |
| F6 | fANOVA importance bars, grouped per dataset | interpretability |
| F7 | Greedy ablation staircase, default → incumbent | interpretability |
| F8 | Component ablation bars (no-warm-start / no-MF / no-LC / single-obj) with error bars | all |
| F9 | Compute ledger stacked bar: GPU-hours by phase | compute pillar |

Style: consistent palette, colourblind-safe, readable at poster distance (min 14 pt in final size), every axis labelled with units, every figure caption states n and the seed count.

---

## C9 — Poster ownership · **P0 in the final week**

You own the layout and figure production; **each member writes their own method section and their own contribution paragraph.** Required elements per the brief: problem & modality, method diagram, search space justification, dataset analysis + meta-features, key results (F1/F3/F6/F8), exam test score, **number of submissions used**, **full compute report**, **lecture-week citation table** (plan §11), individual contributions.

Start the poster skeleton at M3, not at M6. A skeleton with placeholder figures forces you to discover which figures are missing while there is still GPU time to produce them.
