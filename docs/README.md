# AutoML Image — Team Implementation Docs

This folder is the **execution layer** for `AutoML_Image_Project_Plan.md`. The original plan says *what and why*; these docs say *who builds what, in what order, against which interfaces*.

Copy this whole folder into the repo as `docs/` before writing any code.

## Read order

| # | File | Who reads it |
|---|---|---|
| 0 | `00_SCOPE_AND_DECISIONS.md` | Everyone, first, together |
| 1 | `01_ARCHITECTURE_AND_INTERFACES.md` | Everyone — **the contract**. Frozen before parallel work starts. |
| 2 | `02_BUILD_ORDER.md` | Everyone — dependency graph + milestones |
| 3 | `03_MEMBER_A_DATA_AND_TRAINING.md` | Member A (+ skim by B, C) |
| 4 | `04_MEMBER_B_SEARCH.md` | Member B (+ skim by A, C) |
| 5 | `05_MEMBER_C_META_MO_ANALYSIS.md` | Member C (+ skim by A, B) |
| 6 | `06_INFRASTRUCTURE.md` | Everyone — repo, env, git, tracking, GPUs |
| 7 | `07_EXPERIMENT_PROTOCOL.md` | Everyone — the 80% of the grade |
| 8 | `08_INTEGRATION_FREEZE_AND_PHASE_II.md` | Everyone at integration points |
| 9 | `09_AGENT_INSTRUCTIONS.md` | Given to the coding agent with every task |

## ⚠️ Calendar reality check

The source plan is written as if today is **Jul 06**. It is **Jul 18**. Phase I development closes **Jul 21**.

That means the four-sprint schedule in the plan (Sprints 0–3, Jul 06–21) has ~3 days left, not 15. Before starting, the team must pick one:

- **Option COMPRESSED (recommended)** — treat Jul 18–21 as a single "core-only" sprint: ingestion + training loop + BOHB search + Pareto tracking + cost ledger, and *drop* warm-start/portfolio, LC extrapolation, DAC, and the LLM selector to "reported as designed-but-not-run" or squeeze them into Phase II slack. Phase II (Jul 21–Aug 3) is 13 days and the brief only requires the *final* run to be frozen — analysis and ablations can legitimately continue on practice datasets while the exam run is going.
- **Option FULL** — keep the full scope but accept that ablations and significance testing land in Phase II, which risks the 80%-weighted rigor pillar.

`02_BUILD_ORDER.md` gives both a **Full** and a **Compressed (P0/P1/P2)** ordering. Every task in the member files carries a priority tag:

- **P0** = must exist or there is no submission
- **P1** = must exist or the rigor/creativity grade suffers materially
- **P2** = stretch, only after all P0+P1 are green

Do not start a P1 task while any P0 task is red.
