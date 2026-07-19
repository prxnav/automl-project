"""Run the offline benchmark grid and build the warm-start portfolio.

Grid: 30 E-RANK configs (shared with B — run once) + 8 default-HP backbones +
~40 random configs × all practice datasets × all rungs × ≥2 seeds.
Runs on the RTX 3060 overnight; progress in docs/BENCHMARK_STATUS.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("results/offline_benchmark"))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    raise NotImplementedError("Task C4 (Member C)")


if __name__ == "__main__":
    main()
