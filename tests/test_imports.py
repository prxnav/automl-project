"""Smoke test: every module in the contract layout imports cleanly.

Keeps CI green from day 0 and catches interface-drift import errors early.
All stub modules are import-safe (heavy deps only imported inside functions).
"""
import importlib

import pytest

MODULES = [
    "automl_image.types",
    "automl_image.runstore",
    "automl_image.pipeline",
    "automl_image.data.ingest",
    "automl_image.data.splits",
    "automl_image.data.metafeatures",
    "automl_image.data.probe",
    "automl_image.train.loop",
    "automl_image.train.augment",
    "automl_image.train.model_factory",
    "automl_image.train.evaluate",
    "automl_image.train.ensemble",
    "automl_image.search.space",
    "automl_image.search.fidelity",
    "automl_image.search.base",
    "automl_image.search.smac_driver",
    "automl_image.search.optuna_driver",
    "automl_image.search.lc_extrapolation",
    "automl_image.search.scheduler",
    "automl_image.portfolio.benchmark_table",
    "automl_image.portfolio.warm_start",
    "automl_image.portfolio.selector",
    "automl_image.objectives.cost",
    "automl_image.objectives.pareto",
    "automl_image.analysis.fanova",
    "automl_image.analysis.ablation",
    "automl_image.analysis.ledger",
    "automl_image.analysis.stats",
    "automl_image.analysis.plots",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_fidelity_cost_weight():
    """The one piece of real logic in types.py today."""
    from automl_image.types import Fidelity

    assert Fidelity(epochs=27).cost_weight() == 27.0
    assert Fidelity(epochs=9, resolution=112).cost_weight() == 9 * 0.25
    assert Fidelity(epochs=1, data_fraction=0.5).cost_weight() == 0.5
