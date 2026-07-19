# 08 — Integration, Freeze, Phase II & Poster

## 8.1 Integration checkpoints

Integration is scheduled, not opportunistic. Three sessions, all three members present, screen shared.

| Checkpoint | When | Agenda | Exit criterion |
|---|---|---|---|
| **INT-1** | End of M0 | Walk through `types.py` and the four interfaces line by line. Everyone objects *now*. | Contract frozen; each member confirms their stubs compile against it |
| **INT-2** | End of M2 | B's searcher driving A's real loop, writing to C's run store. Run 20 evaluations live. | ≥20 real runs in the store; C loads them and produces F1 with real data |
| **INT-3** | End of M3 | Every ablation flag flipped once, verified to change behaviour (not silently ignored). | `make smoke` green with each flag; all 9 figures generate (possibly ugly) from real data |

**INT-3's "verified to change behaviour" is not a formality.** An ablation flag that is accepted but ignored produces identical bars, and you will not notice until you are staring at the poster. For each flag, assert a behavioural difference: `--no-warm-start` changes the first 8 configs evaluated; `--no-lc` changes the count of `status="early_stopped"` runs; `--single-objective` empties the Pareto CSV of dominated-tracking.

## 8.2 The freeze (M4, Jul 21)

**Freeze means:** tag `v1.0-freeze`. After this, `src/` changes only for bugs that break the pipeline. Every such fix requires a re-run of the dry run and a line in `docs/DECISION_LOG.md`.

**What may continue after freeze:** analysis code (`analysis/`, `make_figures.py`), the poster, additional ablation *runs* using frozen code, documentation.

### The M4 dry run — exact procedure (A executes, all watch)
1. Clean machine, fresh conda env from `environment.yml`.
2. `git checkout v1.0-freeze`.
3. `python scripts/run_automl.py --dataset $DATA_ROOT/practice/ds3 --out results/dryrun --budget 24h`
   (If 24 h of wall-clock isn't available before Jul 21, run at `--budget 6h` and *additionally* verify the BudgetGuard arithmetic on a simulated 24 h with the dry-run stub. Say which you did on the poster.)
4. At a random point, `kill -9` the process. Then `--resume`. Verify it continues and the final artefacts are complete.
5. Verify all 7 output artefacts exist and `submission.csv` passes the format test.
6. Publish `docs/DRYRUN_REPORT.md`: stage timings vs budget, GPU-hours, peak VRAM, incumbent trajectory, what got cut.

**If the dry run fails, the freeze does not happen and nothing else proceeds until it passes.** This is the one gate with no workaround.

## 8.3 Phase II — the exam dataset (Jul 21 – Aug 3)

### Rules of engagement
- Run the frozen pipeline. Do not tune anything based on the exam data beyond what the pipeline does automatically.
- **≤ 3 submissions.** Plan them before making any:
  1. **Submission 1** — the pipeline's own choice (knee-point / top-k ensemble). Made after the full run completes.
  2. **Submission 2** — *only if* a defensible reason exists (e.g. submission 1 revealed a format problem, or the ensemble vs single-model decision was genuinely close on validation and worth one probe). Write the reason in `docs/SUBMISSIONS.md` **before** submitting.
  3. **Submission 3** — hold in reserve for disaster recovery. Assume you will not use it.
- Every submission logged with date, git sha, config hash, val accuracy, test score, justification.
- The poster reports the submission count. Using 1 of 3 is a *positive* signal about discipline. Using 3 with weak reasons reads as test-set tuning.

### Phase II schedule
| Dates | Activity | Owner |
|---|---|---|
| Jul 21–22 | Exam data ingested, format verified, meta-features computed and sanity-checked | A |
| Jul 22–24 | The 24 h pipeline run (with margin for one restart) | B runs, all monitor |
| Jul 24–25 | Pareto review, final selection, full-fidelity retrain + ensemble, **Submission 1** | all decide together |
| Jul 24–30 | In parallel: remaining ablations + baselines on practice datasets, all seeds | B, C |
| Jul 28–30 | fANOVA, significance tests, ledger finalised, all 9 figures final | C |
| Jul 30–Aug 2 | Poster assembly, README, repo cleanup, reproducibility checklist | all |
| Aug 3 | Final read-through, submit **before 23:59 CET** | A submits |

Do not plan to submit on Aug 3. Plan to submit Aug 2 and use Aug 3 as margin.

## 8.4 Poster plan

Layout (against the provided template), with owners:

| Section | Content | Owner |
|---|---|---|
| Problem & modality | Task, dataset, metric, why image | A |
| Method overview | The §4 architecture diagram, redrawn cleanly | C (draws), B (content) |
| Search space & justification | Table with bounds + defaults + why | B |
| Dataset analysis | Meta-features across practice + exam datasets | A |
| Key results | F1 (cost vs accuracy), F3 (Pareto), F6 (fANOVA), F8 (ablations) | C |
| Multi-fidelity validation | F2 rank correlation, incl. any dropped fidelity | B |
| Meta-learning | F5 warm-start vs cold, single-best/virtual-best bounds | C |
| Exam result | Test score + **# submissions used** | B |
| Compute report | F9 ledger + the 24 h budget table + dry-run findings | C |
| **Lecture-week citations** | The §11 mapping table — required by the brief | all |
| **Individual contributions** | One short paragraph each, name-attributed | each member writes their own |
| Honest limitations | The negative results from §7.8 | all |

**Week-citation table** (put a version of this on the poster verbatim — the brief requires naming the weeks):

| Deck | Concept | Where in our system | Whose work |
|---|---|---|---|
| Basics of HPO | Config spaces incl. conditionals; validation & resampling; random search | `search/space.py`, `data/splits.py`, BL2 | A, B |
| BO for HPO | RF surrogate, acquisition, exploration/exploitation, SMAC | `search/smac_driver.py` | B |
| Speedup | Multi-fidelity, SH, Hyperband, BOHB, LC extrapolation, meta-learning | `search/fidelity.py`, `lc_extrapolation.py`, `portfolio/` | B, C |
| Multi-criteria | Pareto dominance, hypervolume, a-posteriori selection | `objectives/pareto.py`, `portfolio/selector.py` | C |
| Interpretability | fANOVA, ablation studies | `analysis/fanova.py`, `ablation.py` | C |
| Algorithm Selection | Meta-features, portfolios, single- vs virtual-best | `data/metafeatures.py`, `portfolio/warm_start.py` | A, C |
| DAC (light) | Dynamic configuration as contextual MDP → heuristic plateau LR policy | `train/loop.py` plateau schedule | A |

Seven decks, each doing real work. That is the creativity pillar, evidenced rather than asserted.

## 8.5 Final submission checklist (A verifies, Aug 2)

- [ ] `git checkout` of the submitted tag into a fresh dir + fresh env → `make smoke` passes
- [ ] README commands copy-pasteable and correct
- [ ] `environment.yml` versions exact
- [ ] `configs/best_configs/exam.yaml` present and reproduces the reported model
- [ ] `submission.csv` format verified against the template
- [ ] `docs/SUBMISSIONS.md` complete with count
- [ ] Ledger covers all compute, offline and online
- [ ] All 9 figures regenerate from `make_figures.py`
- [ ] `docs/CONTRIBUTIONS.md` complete; poster contributions section matches it
- [ ] Poster PDF at required dimensions, text readable at 1 m
- [ ] Poster-availability sheet filled
- [ ] Submitted **before** Aug 03, 23:59 CET
