"""TODO(A2):
- split determinism: pure function of (dataset, seed)
- no train/val overlap
- stratification preserved per class
- nothing in src/automl_image/search/ or portfolio/ imports the test loader
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(A2): implement with splits.py")
