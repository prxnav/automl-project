"""TODO(C1):
- log() writes a parquet shard; load() returns the row (use tmp_path)
- log_epoch + learning_curves round-trip
- ledger() groups gpu_seconds by phase/member/machine
- validate_result() rejects failure-contract violations
  (status="oom" with val_acc != 0.0)
- concurrent shards: two RunStore instances, load() sees both
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(C1): implement with runstore.py")
