"""ONE-CLICK entry point.

    python scripts/run_automl.py --dataset data/fashion --budget 24 --out results/run1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from automl_image.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="One-click AutoML for image classification")
    p.add_argument("--dataset", type=Path, required=True, help="dataset root directory")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--budget", type=float, default=24.0, help="wall-clock budget in hours")
    p.add_argument("--searcher", choices=["bohb", "hyperband", "random", "optuna_asha", "default"],
                   default="bohb")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--no-warm-start", action="store_true", help="ablation: cold start")
    p.add_argument("--no-lc", action="store_true", help="ablation: no LC extrapolation")
    p.add_argument("--single-objective", action="store_true", help="ablation: accuracy only")
    p.add_argument("--dry-run", action="store_true", help="stubbed evaluations, no GPU")
    p.add_argument("--resume", action="store_true", help="resume from out/state.json")
    args = p.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        out_dir=args.out,
        budget_hours=args.budget,
        searcher=args.searcher,
        warm_start=not args.no_warm_start,
        lc_extrapolation=not args.no_lc,
        multi_objective=not args.single_objective,
        seeds=args.seeds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
