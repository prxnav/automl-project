# Data & Submission Format (from the exam template repo)

Owner: A (Karan) — verify before writing `ingest.py`. Source: repo README +
`src/automl/datasets.py`.

## Dataset layout (per dataset, under `./data/`)

```
data/<name>/
├── images_train/000001.jpg ...
├── images_test/000001.jpg ...
├── description.md
├── train.csv        # columns: image_file_name, label
└── test.csv         # same; label = NaN for the exam dataset
```

Download: `python -m download_datasets` (or the dataset classes with
`download=True`). Zips hosted at ml.informatik.uni-freiburg.de
(vision-phase1.zip / vision-phase2.zip).

## Datasets

| Name | Classes | Train | Test | Ch | Resolution | Ref. acc |
|---|---|---|---|---|---|---|
| fashion | 10 | 60,000 | 10,000 | 1 | 28×28 | 0.88 |
| flowers | 102 (imbalanced) | 5,732 | 2,457 | 3 | 512×512 | 0.55 |
| emotions | 7 | 28,709 | 7,178 | 1 | 48×48 | 0.40 |
| **skin_cancer** (exam) | 7 (imbalanced) | 7,010 | 3,005 | 3 | 450×450 | 0.71 |

## Prediction format — ⚠️ NOT a CSV

**`predictions.npy`**: `np.save` of an integer label array, one entry per
test image, in `test.csv` row order. (The planning docs said
`submission.csv`; the real exam grades `.npy` — logged in DECISION_LOG.)

Auto-evaluation (Phase II, from Jul 21 00:00 CET): put the file at
`data/exam_dataset/predictions.npy` on the **`test` branch** (unrelated
history — copy the file across, don't merge), push, `git pull` for results in
`data/exam_dataset/test_out/`. That dir must contain nothing else. **Max 3
scored submissions** (docs/SUBMISSIONS.md). Never touch the workflow yaml.

## Final submission (Aug 3, 23:59 CET)

`final_test_preds.npy` + poster PDF `final_poster_vision_<team-name>.pdf` +
`run_instructions.md` (≤24 h fit command + predict command) + `team_info.txt`
(matriculation IDs only, no names).
