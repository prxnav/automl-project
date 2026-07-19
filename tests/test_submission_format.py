"""TODO(A7): predictions.npy byte-check against the exam contract —
np.save'd integer array, one label per test image, test.csv row order,
original label encoding (see docs/DATA_FORMAT.md). Getting this wrong on
Aug 3 would be the stupidest possible way to lose points.
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(A7): implement with ensemble.write_predictions")
