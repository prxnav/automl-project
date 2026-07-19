"""TODO(C3): hand-computed cases — this is where silent wrongness hides
(a wrong hypervolume produces a plausible-looking plot):
- is_dominated on all four max/min orientation combos
- pareto_front on a 5-point set with known front
- hypervolume of a 2-3 point front against a hand-computed value
- knee_point on a front with an obvious knee
- hypervolume_over_time monotonically non-decreasing
"""
import pytest

pytestmark = pytest.mark.skip(reason="TODO(C3): implement with pareto.py")
