# 06 — Infrastructure, Tooling & Team Systems

Everything here is "not the science, but the science fails without it." Owner in brackets.

## 6.1 Version control [A]

- One repo, `main` protected. Branches: `feat/<member>-<topic>`, e.g. `feat/b-bohb-searcher`.
- PRs require one review. Small and frequent; a 900-line PR on Jul 20 will not get reviewed properly.
- **Commit message convention:** `[A|B|C] area: what`. At the end you need "who did what" evidence — `git log --author` plus this prefix generates the individual-contributions section automatically. This is graded; it costs nothing to do from commit 1.
- Tag `v1.0-freeze` at M4. The exam run is executed from that tag, nothing else.
- Large files (checkpoints, parquet) are **gitignored**. Share via a shared drive / rsync, not git.

## 6.2 Environment [A]

- `environment.yml` pinned to exact versions (the brief requires versions).
- Every machine runs `conda env create -f environment.yml` from the same file. Divergence between machines is a source of unreproducible results — check `pip freeze | md5sum` matches on all three and paste it in `docs/ENV_CHECK.md`.
- `TIMM_HOME` set to a shared path so pretrained weights download once per machine.
- `.env.local` per machine (gitignored) with `MACHINE_NAME`, `MEMBER`, `DATA_ROOT`, `RESULTS_ROOT`, `NUM_WORKERS`. `RunStore` reads `MACHINE_NAME` and `MEMBER` automatically — this is how the ledger gets per-member attribution for free.

## 6.3 The three GPUs [all]

| Machine | VRAM | Default owner | Standing role |
|---|---|---|---|
| RTX 5070 desktop | 12 GB | B | Search arms (BOHB / HB / random) |
| RTX 5070 laptop (Omen 16) | ~8 GB | A | Dev, baselines, final full-fidelity retrains |
| RTX 3060 | 6/12 GB | C | Offline benchmark grid; **reference device for all latency measurements** |

Rules:
- Batch/resolution feasibility is defined by the **smallest** card in play for that experiment. `scripts/measure_vram.py` fills `configs/vram_table.yaml` per machine.
- Time-heterogeneity is real (~2× spread). Never compare "number of evaluations"; always compare **GPU-seconds** or wall-clock. Record `device` on every `RunResult` and mention heterogeneity on the poster (the *Basics* deck flags it).
- `docs/GPU_QUEUE.md`: a simple table of `machine | job | started | expected finish | owner`. Update before you walk away. Idle GPUs between Jul 18 and Jul 30 are the most expensive mistake available to this team.
- **Backup compute:** Google Colab and bwUniCluster. Verify *now* that the code runs on at least one of them (a 15-minute test), so it's a real fallback rather than a hope. Risk register item #1.

## 6.4 Experiment tracking & the run store [C]

- Source of truth: the Parquet run store (C1). W&B optional viewer only.
- `docs/RUNSTORE_SCHEMA.md` — agreed day 1, enforced by a `validate_result()` call inside `RunStore.log`.
- Every logged run carries `git_sha`. If a number cannot be traced to a commit, it does not go on the poster.

## 6.5 Data management [A]

- `DATA_ROOT/practice/{ds1,ds2,ds3}/` and `DATA_ROOT/exam/`.
- Checksum each dataset after copying to each machine (`sha256sum` of the manifest) and record in `docs/DATA_CHECKSUMS.md`. Three machines silently holding different data versions is a classic, invisible, result-destroying bug.
- **Exam test labels do not exist.** Add a pre-commit hook or a test that fails if any file under `src/search/` or `src/portfolio/` references a test split.

## 6.6 Testing [A owns the harness, each member tests their own code]

- `pytest`, CPU-only, < 60 s total. Runs in CI on every PR.
- Required tests:
  - `types.py` round-trip serialisation + `TrainConfig.hash()` stability.
  - Split determinism + no train/val overlap + stratification preserved.
  - `evaluate_config` on the fake dataset (CPU, resnet18, 2 epochs).
  - Submission format byte-check against the template example.
  - ConfigSpace default == `TrainConfig.default()`.
  - Sign convention: searcher minimises negative accuracy.
  - Pareto/hypervolume/knee against hand-computed cases.
  - Leave-one-dataset-out: dataset never in its own warm-start data.
- **Smoke test** `make smoke`: full pipeline, `--dry-run`, 2-minute budget, fake dataset. Must pass before every merge to `main` and before the M4 dry run.

## 6.7 Reproducibility checklist [A, verified at M4]

- [ ] Fixed seeds everywhere; seed logged per run.
- [ ] Pinned deps; `pip freeze` hash identical across machines.
- [ ] Every reported number regenerable by `scripts/make_figures.py` from the committed run store.
- [ ] `configs/best_configs/*.yaml` reproduce the reported models via `train_config.py`.
- [ ] Hardware logged per run.
- [ ] README states exact commands, expected runtime, expected output files.

## 6.8 Team process [all]

- **Daily 15-min sync** (Jul 18–Aug 3), same time, three questions: what's green, what's blocked, what's running on your GPU tonight.
- **Blocking rule:** if you are blocked on another member for > 2 h, build against a stub and say so in the sync. Nobody waits.
- `docs/DECISION_LOG.md` — dated entries for every deviation from `00_SCOPE_AND_DECISIONS.md`.
- `docs/CONTRIBUTIONS.md` — each member appends weekly. Individual grades can differ; this is your evidence.
- **Scope-creep veto:** any member can veto a new feature before M4 by pointing at a red P0/P1 task. No debate needed.

## 6.9 Third-party / external systems checklist

| System | Needed for | Owner | Status to confirm by Jul 18 EOD |
|---|---|---|---|
| Course GitHub template repo | Data format, submission format, reference baseline number | A | Cloned, format documented in `docs/DATA_FORMAT.md` |
| Submission portal | ≤3 test submissions | B | Access confirmed, submission mechanics understood *before* Phase II |
| `timm` pretrained weights | Backbones | A | All 8 downloaded and cached on all 3 machines (offline safety) |
| Colab / bwUniCluster | Backup compute | C | One successful test run |
| Poster template | Final deliverable | C | Downloaded, dimensions known |
| Poster-availability sheet | Admin | any | Filled |
| Shared drive for checkpoints/parquet | Cross-machine result merging | C | Created, all three can write |
