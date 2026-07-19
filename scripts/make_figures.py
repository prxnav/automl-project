"""Regenerate EVERY poster figure (F1-F9, docs/05 C8) from the run store.
No hand-edited plots, ever — if a figure can't be regenerated from here, it
doesn't go on the poster.

    python scripts/make_figures.py --store results --out results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--figures", nargs="*", default=None,
                   help="subset like F1 F3; default all")
    p.parse_args()
    raise NotImplementedError("Task C8 (Member C)")


if __name__ == "__main__":
    main()
