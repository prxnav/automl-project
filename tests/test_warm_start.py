"""TODO(C5):
- leave-one-dataset-out: a dataset NEVER appears in its own warm-start data
  (plan §5.7, the #1 pitfall — this test is mandatory)
- fallback: with no benchmark table, get_initial_design returns a
  space-filling design of size k (B is never blocked on C)
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(C5): implement with warm_start.py")
