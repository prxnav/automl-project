"""TODO(B1):
- ConfigSpace default configuration == TrainConfig.default()
- conditionals: n_unfrozen_blocks only active when ft_strategy==last_n_blocks etc.
- config_from_configuration / configuration_from_config round-trip
- forbidden clauses exclude known VRAM-infeasible triples
- sign convention: searcher minimises NEGATIVE accuracy (unit-test the
  conversion — the classic 3-hour bug, docs/04 B3)
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(B1): implement with space.py")
