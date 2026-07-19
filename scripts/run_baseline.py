"""Baselines: default config (no search), random search (same wall-clock),
strong tuned-timm default (locked D7 — the honest bar we must beat).

    python scripts/run_baseline.py --baseline default --dataset data/fashion --budget 1
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["default", "random", "strong_default"], required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--budget", type=float, default=1.0, help="hours")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("results/baselines"))
    args = p.parse_args()
    raise NotImplementedError("Task A6/B6")


if __name__ == "__main__":
    main()
