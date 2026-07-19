"""Synthetic dataset generator — ALL data tests use this; no test may depend
on a real dataset (docs/06 §6.6, docs/09 §9.4).

Spec (docs/03 A1): ~100 images, 4 classes, mixed sizes, one corrupt file,
one grayscale image, exam directory layout (images_train/ images_test/ +
train.csv/test.csv with columns image_file_name,label).
"""
from __future__ import annotations

from pathlib import Path


def make_fake_dataset(root: Path, *, n_images: int = 100, n_classes: int = 4,
                      seed: int = 0) -> Path:
    """Create the fake dataset under `root` and return its path."""
    raise NotImplementedError("Task A1 (Member A)")
