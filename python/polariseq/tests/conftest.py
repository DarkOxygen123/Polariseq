"""Fixtures shared by the tests of the object API."""

from __future__ import annotations

import polariseq as ps
import polariseq._project as P
import pytest

# anndata 0.11 with pandas 2.2 refuses to write a string index unless told
# that nullable strings are acceptable; the tests write small h5ad files of
# their own, and without this they fail on such an environment (the cluster's)
# for a reason that has nothing to do with the library. Absent in older and
# newer anndata, where nothing needs setting.
try:
    import anndata

    anndata.settings.allow_write_nullable_strings = True
except (ImportError, AttributeError):
    pass


@pytest.fixture
def proj(tmp_path):
    """A project folder of its own, and the budget restored afterwards."""
    before = ps.budget()
    p = ps.project(tmp_path / "proj")
    yield p
    P._CURRENT = None
    ps.set_budget(ram_gb=before.ram_gb, spill_dir=before.spill_dir)
