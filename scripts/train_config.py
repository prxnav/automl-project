"""Reproduce a single config at full fidelity; print val acc, write predictions.

    python scripts/train_config.py --config configs/best_configs/default.yaml --dataset data/fashion

M1 gate: this finishing on a practice dataset. Also how every
configs/best_configs/*.yaml is verified to reproduce its reported number
(reproducibility checklist, docs/06 §6.7).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="TrainConfig yaml")
    p.add_argument("--dataset", type=Path, required=True, help="dataset root directory")
    p.add_argument("--out", type=Path, default=Path("results/train_config"))
    p.add_argument("--seed", type=int, default=0)
    p.parse_args()
    raise NotImplementedError("Task A6 (Member A)")


if __name__ == "__main__":
    main()
