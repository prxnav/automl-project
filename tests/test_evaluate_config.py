"""TODO(A3s): against the dry_run stub —
- returns a valid RunResult, status="ok", learning_curve length == epochs
- deterministic for the same (config, seed, dataset)
- lc_callback returning True → status="early_stopped", partial curve
TODO(A3): real loop on the fake dataset (CPU, resnet18, 2 epochs, < 60 s).
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(A3s): implement with the dry-run stub")
