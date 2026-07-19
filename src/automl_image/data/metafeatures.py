from __future__ import annotations

from automl_image.types import DatasetSpec, MetaFeatures


def compute(spec: DatasetSpec, *, sample_size: int = 2000, seed: int = 42) -> MetaFeatures:
    raise NotImplementedError
