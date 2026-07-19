"""Measure peak VRAM per (backbone × batch_size × resolution) on THIS machine
and fill configs/vram_table.yaml — the source of the ConfigSpace forbidden
clauses. Run once per machine.

    python scripts/measure_vram.py --machine rtx3060
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--machine", required=True, help="key used in configs/vram_table.yaml")
    p.add_argument("--out", default="configs/vram_table.yaml")
    p.parse_args()
    raise NotImplementedError("Task B1 (Member B)")


if __name__ == "__main__":
    main()
