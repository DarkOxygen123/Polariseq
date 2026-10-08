"""The in-memory mode warns when a load is predicted to exceed the budget.

The budget bounds the out-of-core mode only. When a matrix is about to be
held in memory, the planner's own estimate is checked against the budget and
a `BudgetWarning` says so if it will not fit — never an error, and at most
once per load.
"""

from __future__ import annotations

import warnings

import polariseq as ps
import pytest

FIXTURE = "tests/data/medium_10kx2k.h5ad"


@pytest.fixture
def ram_budget(monkeypatch):
    """Set the active budget's RAM ceiling for one test, restored after.

    Patches the module's budget directly rather than calling `set_budget`, so
    a test never touches the worker pool, which cannot be resized.
    """
    def _set(gb: float) -> None:
        monkeypatch.setattr(ps, "_BUDGET", ps.budget().with_overrides(ram_gb=gb))
    return _set


def _budget_warnings(record) -> list:
    return [w for w in record if issubclass(w.category, ps.BudgetWarning)]


def test_lazy_load_over_budget_warns_once(ram_budget) -> None:
    ram_budget(0.001)
    adata = ps.read_h5ad(FIXTURE)          # metadata only: nothing to warn about yet
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ps.pp.calculate_qc_metrics(adata)  # the matrix becomes resident here
        ps.pp.filter_cells(adata, min_genes=1)
    found = _budget_warnings(record)
    assert len(found) == 1
    msg = str(found[0].message)
    assert "medium_10kx2k.h5ad" in msg
    assert "does not bound the in-memory mode" in msg


def test_eager_load_over_budget_warns(ram_budget) -> None:
    ram_budget(0.001)
    with pytest.warns(ps.BudgetWarning):
        ps.read_h5ad(FIXTURE, lazy=False)


def test_load_within_budget_is_silent(ram_budget) -> None:
    ram_budget(100.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.calculate_qc_metrics(adata)
        ps.read_h5ad(FIXTURE, lazy=False)
    assert not _budget_warnings(record)


def test_warning_matches_the_plan(ram_budget) -> None:
    """The warning's figure is `ps.plan()`'s, so the two never disagree."""
    ram_budget(0.001)
    predicted = ps.plan(FIXTURE, storage="ram", verbose=False).peak_gb
    with pytest.warns(ps.BudgetWarning, match=f"{predicted:.1f} GB"):
        ps.read_h5ad(FIXTURE, lazy=False)
