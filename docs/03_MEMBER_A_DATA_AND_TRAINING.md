# 03 — Member A: Data & Training Core (+ Integrator)

**Owns:** `src/automl_image/types.py`, `data/`, `train/`, `pipeline.py`, `scripts/`, `environment.yml`, CI
**Poster decks you cite:** *Basics of HPO* — validation & resampling, config-space realisation, the black-box view
**Machine:** RTX 5070 laptop (~8 GB) — deliberately the *smallest* practical card, so VRAM constraints get found early
**You are also the integrator:** you own `main` staying green and you run the M4 dry run.

---

## A0 — Repo skeleton & shared types · **P0 · 1.5 h · blocks everyone**

Create the repo per `01_ARCHITECTURE_AND_INTERFACES.md §1.1`, then write `types.py` exactly as specified in §1.2.

**Tasks**
1. `pyproject.toml` with package `automl_image`, ruff + pytest config, `requires-python = "==3.11.*"`.
2. `environment.yml` with **pinned** versions. Starting point (verify against the exam template repo's constraints):
   ```yaml
   name: automl-image
   channels: [pytorch, nvidia, conda-forge]
   dependencies:
     - python=3.11
     - pytorch=2.4.*        # must match CUDA available on RTX 50-series
     - torchvision=0.19.*
     - pip
     - pip:
       - timm==1.0.9
       - smac==2.2.0
       - ConfigSpace==1.2.0
       - optuna==4.0.0
       - scipy==1.14.*
       - pandas==2.2.*
       - pyarrow==17.*
       - scikit-learn==1.5.*
       - matplotlib==3.9.*
       - seaborn==0.13.*
       - fvcore==0.1.5.post20221221   # FLOPs counting
       - pyyaml==6.0.*
       - rich==13.*
   ```
   ⚠️ RTX 50-series (Blackwell, sm_120) needs a recent CUDA build. **Verify torch actually sees the GPU on all three machines before anything else** — if the 5070s need a nightly/cu124+ wheel, that changes the pin and it must be discovered on day 1, not day 4.
3. `types.py` — all dataclasses from §1.2, with `to_dict`/`from_dict`/`hash` implemented and unit-tested for round-trip.
4. `.github/workflows/ci.yml` (or a `make check` target if no GitHub Actions): ruff + pytest on CPU only, no GPU tests in CI.
5. `.gitignore`: `results/**` except `results/figures/**` and `results/*/ledger.csv`.

**Definition of done:** `pip install -e .` + `pytest` green on all three machines; `python -c "import torch; print(torch.cuda.get_device_name(0))"` works on all three.

---

## A1 — Dataset ingestion · **P0 · 2 h**

`src/automl_image/data/ingest.py`

```python
def load(dataset_path: Path, cache_dir: Path | None = None) -> DatasetSpec
def build_dataloaders(spec, config, fidelity, split_indices, *, num_workers=None) -> tuple[DataLoader, DataLoader]
```

**Requirements**
- Support the exam's format **first** (check the course GitHub template — do this before writing a line of code) and `ImageFolder` layout as a fallback.
- Handle: grayscale → 3-channel conversion, corrupt/unreadable images (skip + log count), variable image sizes, non-contiguous class labels (remap to `0..n-1` and store the mapping so `submission.csv` uses original labels).
- Cache a manifest (`parquet`: path, label, height, width) so ingestion is O(1) on repeat runs. The 24 h budget cannot afford re-scanning a large dataset every run.
- `num_workers` default = `min(8, os.cpu_count() - 1)`; expose as env var because the three machines differ.

**Test:** synthetic dataset generator in `tests/fixtures/make_fake_dataset.py` (100 images, 4 classes, mixed sizes, one corrupt file, one grayscale). All data tests use it — no test may depend on a real dataset.

---

## A2 — Splits & validation protocol · **P0 · 1.5 h**

`src/automl_image/data/splits.py`

```python
def make_split(spec: DatasetSpec, *, strategy: Literal["holdout","cv"], val_fraction=0.2,
               n_folds=5, seed=42) -> SplitPlan
def choose_strategy(meta: MetaFeatures) -> Literal["holdout","cv"]
```

**Requirements (this is the leakage firewall — D6)**
- **Stratified** by class always.
- The split is a **pure function of `(dataset, seed)`** — identical for every config in a search. Persist it to `out_dir/split.json` and assert on reload that the hash matches. Every config sees the identical val set → comparisons are paired → C's Wilcoxon test is valid.
- `choose_strategy` rule (locked, document on poster): CV if `meta.n_samples < 2000` **or** `meta.samples_per_class_min < 20`; else hold-out.
- Test labels are never loaded by anything other than the final predictor. Add a test that asserts `SplitPlan` has no overlap between train/val indices, and that nothing in `src/search/` or `src/portfolio/` imports the test loader.

---

## A3 — Training loop · **P0 · 5 h · the single most important deliverable**

`src/automl_image/train/loop.py` implementing **I1** exactly.

**Step 1 (hour 1, do this before the real loop): the stub.**
```python
def evaluate_config(..., dry_run: bool = False) -> RunResult
```
When `dry_run=True`, return a synthetic `RunResult` whose `val_acc` is a deterministic smooth function of `(backbone rank, lr distance from 3e-4, epochs)` plus seeded noise, with a plausible saturating learning curve and `sleep(0.01 * fidelity.cost_weight())`. **Push this within 4 hours of project start.** B and C then develop and unit-test their entire stack without a GPU.

**Step 2: the real loop.** Requirements:
- `timm.create_model(backbone, pretrained=True, num_classes=n)`; **cache the pretrained weights once** to a shared dir (`TIMM_HOME`) — re-downloading inside the 24 h run is unacceptable and the weights are excluded from the budget anyway.
- **AMP** (`torch.amp.autocast` + `GradScaler`) always on. `channels_last` memory format.
- Fine-tuning strategies: implement all five from `TrainConfig.ft_strategy`. `discriminative_lr` = layer-wise LR decay factor 0.75 across `timm`'s `group_matcher` groups.
- Schedules: cosine with warmup, step, constant, and `plateau` (this one is the **heuristic-DAC** policy for the stretch citation — reduce LR ×0.3 after 2 epochs without val improvement; log every LR change so C can plot the policy's trajectory).
- **Rung-aware checkpointing:** save `{model, optimizer, scheduler, epoch, rng_state}` at the end of every fidelity rung to `checkpoint_dir/{config.hash()}_{seed}.pt`. `resume_from` must continue training, not restart. **Successive Halving is ~3× cheaper with this and ~1× without it** — do not skip it.
- **`lc_callback` after every epoch**, receiving the `EpochRecord`; if it returns `True`, stop, set `status="early_stopped"`, return the partial curve.
- **OOM handling** per §1.2 failure contract. Also: catch OOM, `torch.cuda.empty_cache()`, halve batch size, enable grad accumulation to preserve effective batch size, retry **once**.
- **Timing:** `gpu_seconds` measured with `torch.cuda.Event` around the training region; `wallclock_seconds` around everything including data loading; `peak_vram_mb` from `torch.cuda.max_memory_allocated`. C's whole compute pillar depends on these three numbers being right.
- Determinism: seed `torch`, `numpy`, `random`, set `torch.backends.cudnn.deterministic` under a `--strict-repro` flag (off by default because it costs ~15% speed; on for the final reported runs).

**Test:** with the fake dataset, `evaluate_config` on `resnet18`, 2 epochs, CPU, must finish < 60 s and return a valid `RunResult` with a 2-entry learning curve.

---

## A4 — Augmentation module · **P0 · 1.5 h**

`src/automl_image/train/augment.py`

```python
def build_transforms(config: TrainConfig, fidelity: Fidelity, spec: DatasetSpec) -> tuple[Compose, Compose]
def build_mixup_fn(config: TrainConfig, n_classes: int) -> Callable | None
```
- Use `timm.data.create_transform` where possible; `torchvision.transforms.RandAugment/TrivialAugmentWide/AutoAugment` otherwise.
- **Resolution comes from `fidelity`, not from config** — that's what makes resolution a fidelity dimension.
- Normalisation uses the **backbone's** pretrained mean/std (`timm` `default_cfg`), not the dataset's, when fine-tuning from ImageNet. Log this choice; it's a real methodological detail worth a poster line.
- Val transform is always deterministic resize+center-crop at the *evaluation* resolution. Decide and document: when training at 96 px, do we evaluate at 96 px (consistent, cheap) or 224 px (matches final)? **Locked: evaluate at the training resolution during search, at 224 px for final.** The rank-correlation experiment (B2) measures whether that's OK.

---

## A5 — Meta-features & probe · **P1 · 3 h** (B and C consume this; C's warm-start needs it)

`src/automl_image/data/metafeatures.py` — all fields in `MetaFeatures`, pure statistics, computed on a **sample of ≤2000 images** so it stays under 60 s.

`src/automl_image/data/probe.py`
```python
def run_probe(spec, split, *, backbone="resnet18", epochs=1, resolution=96, device="cuda:0") -> tuple[float, float, float]
```
- Trains `resnet18` linear-probe for 1 epoch at 96 px on ≤20% of data. Returns `(acc, loss_slope, seconds)`.
- **Reuse it:** the probe result is logged to the run store as a legitimate lowest-fidelity `RunResult` and handed to the searcher as a free observation. Plan §5.2 explicitly calls this out; don't waste it.
- Hard timeout 5 min; on timeout return `(0.0, 0.0, elapsed)` and let downstream fall back to defaults.

---

## A6 — Pipeline orchestration & one-click scripts · **P0 · 4 h**

`src/automl_image/pipeline.py` per §1.4, plus:

- `scripts/run_automl.py` — argparse exactly matching the documented contract:
  `--dataset --out --budget 24h --searcher --seeds --no-warm-start --no-lc --single-objective --dry-run --resume`
- `scripts/train_config.py --config <yaml> --dataset <path>` — reproduces a single config, full fidelity, prints val acc + writes submission.
- `scripts/run_baseline.py --baseline {default,random,strong_default} --budget <h>` — Member C uses this for §07 baselines.

**Budget enforcement (the 24 h contract):**
- A `BudgetGuard` class: knows the wall-clock deadline, exposes `remaining_seconds()`, and refuses to launch an evaluation whose *estimated* cost exceeds the remaining budget minus the safety margin.
- Cost estimate from `fidelity.cost_weight() × measured_seconds_per_unit_on_this_machine` (calibrated from the probe + first few runs).
- Stage budgets from `configs/budget.yaml`: ingest+meta 0.25 h, search 18.5 h, analysis 0.5 h, margin 3.5 h. If search overruns, it is cut off, not extended — the incumbent is already valid.
- On SIGTERM/SIGINT: flush the run store, write `state.json`, exit 0. The pipeline must survive a laptop lid closing.

**Resume:** `--resume` reads `state.json` and skips completed stages; the searcher resumes from its own persisted state (B's responsibility, but you define the handshake: `state.json["search_state_path"]`).

---

## A7 — Final assembly: full-fidelity retrain, ensemble, TTA · **P1 · 2.5 h**

`src/automl_image/train/ensemble.py`
```python
def retrain_full(configs: list[TrainConfig], spec, *, epochs, device) -> list[Path]
def soft_vote(checkpoints, loader, *, tta: bool = True, tta_views: int = 4) -> np.ndarray
def write_submission(probs, index, class_mapping, out: Path) -> None
```
- Ensemble size **2–3** (locked, D5 rationale: honesty about inference cost — the ensemble's params/FLOPs go on the Pareto plot as its own point).
- TTA = horizontal flip + 2 scales, averaged probabilities. Report single-model vs ensemble vs ensemble+TTA as three rows — the plan requires this transparency.
- **`write_submission` must produce the exact exam format.** Verify against the template repo's example file byte-for-byte on headers and label encoding. Add a `tests/test_submission_format.py` that checks it. Getting this wrong on Aug 3 would be the stupidest possible way to lose points.

---

## A8 — Integrator duties · **continuous**

- Own `main`. Every PR needs one review from another member. `main` is always green.
- Run the **M4 dry run** and publish its `ledger.csv` in `docs/`.
- Own `README.md`: exact commands, exact versions, hardware used, how to reproduce every reported number.
- Maintain `docs/DECISION_LOG.md` — every deviation from `00_SCOPE_AND_DECISIONS.md` with date, reason, who agreed.

## A9 — Your poster content
- The validation-protocol figure (why hold-out during search, CV for final; leakage firewall).
- The engineering-efficiency numbers: AMP + checkpoint-resume speedups (measure them — "SH promotion via checkpoint resume saved X GPU-hours" is a compute-pillar sentence).
- Compute report table (produced with C).
