# Decision Log

Every deviation from `00_SCOPE_AND_DECISIONS.md`, dated, with all three members agreeing.

| Date | Decision | Changed from | Reason | Agreed by |
|---|---|---|---|---|
| 2026-07-18 | Cost objective = params/FLOPs primary, latency logged only | (open, plan §14.1) | Reproducibility across 3 heterogeneous machines | A, B, C |
| 2026-07-18 | Submission artefact is `predictions.npy` (np.save int labels, test.csv order), not `submission.csv` | plan/docs said submission.csv | The actual exam template repo grades a `.npy` pushed to the `test` branch (see repo README + docs/DATA_FORMAT.md) | pending A, B sign-off |
