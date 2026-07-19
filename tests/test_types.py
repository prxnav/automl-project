"""TODO(A0): implement alongside types.py:
- TrainConfig round-trip: from_dict(to_dict(c)) == c
- hash(): stable across processes, 12 chars, EXCLUDES seed
  (same config, different seed → same hash)
- default() fields match configs/best_configs/default.yaml
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(A0): implement with types.py methods")


def test_trainconfig_roundtrip():
    ...


def test_hash_stable_and_excludes_seed():
    ...


def test_default_matches_yaml():
    ...
