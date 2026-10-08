"""Scarf's preprocessing in Polariseq: genes by flavor='scarf', the counts normalized again over
those genes (pp.renormalize), and each gene standardized before PCA. In memory and out of core the
recipe gives the same principal components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polariseq as ps
import pytest

FIXTURE = Path(__file__).resolve().parents[3] / "tests/data/medium_10kx2k.h5ad"
N_HVG, N_PCS = 300, 10


def _recipe(adata, counts=None):
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="scarf", subset=True)
    ps.pp.renormalize(adata, target_sum=1000, counts=counts)
    ps.pp.pca(adata, N_PCS, scale=True, seed=0)
    return np.asarray(adata.obsm["X_pca"], dtype=np.float64)


def test_in_memory_and_out_of_core_agree(proj) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not present")
    mem = ps.read_h5ad(FIXTURE, mode="memory")
    a = _recipe(mem)
    b = _recipe(ps.read_h5ad(FIXTURE, mode="out-of-core"))
    assert a.shape == b.shape
    for j in range(5):   # the leading components, up to sign
        r = abs(np.corrcoef(a[:, j], b[:, j])[0, 1])
        assert r > 0.999, f"component {j}: |r| = {r:.5f}"


def test_renormalize_scales_each_cell_over_the_current_genes() -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not present")
    adata = ps.read_h5ad(FIXTURE, mode="memory")
    raw = ps.read_h5ad(FIXTURE, mode="memory")
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="scarf", subset=True)
    ps.pp.renormalize(adata, target_sum=1000, log1p=False, counts=raw)
    x = adata.X.to_scipy() if hasattr(adata.X, "to_scipy") else adata.X
    totals = np.asarray(x.sum(axis=1)).ravel()
    nonzero = totals > 0
    np.testing.assert_allclose(totals[nonzero], 1000.0, rtol=1e-4)


def test_in_memory_without_a_copy_equals_from_the_raw_counts() -> None:
    """normalize_total records each cell's total, so renormalize turns the values back into counts in
    place: the result equals renormalizing the raw counts, without holding a copy of them."""
    if not FIXTURE.exists():
        pytest.skip("fixture not present")
    results = []
    for use_copy in (False, True):
        adata = ps.read_h5ad(FIXTURE, mode="memory")
        raw = ps.read_h5ad(FIXTURE, mode="memory") if use_copy else None
        ps.pp.normalize_total(adata, target_sum=1e4)
        ps.pp.log1p(adata)
        ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="scarf", subset=True)
        if use_copy:
            adata.uns["normalize"].pop("cell_totals")   # force the path through the raw counts
        ps.pp.renormalize(adata, target_sum=1000, counts=raw)
        x = adata.X.to_scipy() if hasattr(adata.X, "to_scipy") else adata.X
        results.append(x.toarray())
    np.testing.assert_allclose(results[0], results[1], rtol=1e-5, atol=1e-6)


def test_in_memory_with_changed_cells_and_no_counts_is_refused() -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not present")
    adata = ps.read_h5ad(FIXTURE, mode="memory")
    ps.pp.normalize_total(adata)
    adata = adata[np.arange(adata.n_obs // 2)]   # the cells changed after normalizing
    with pytest.raises(ValueError, match="counts="):
        ps.pp.renormalize(adata)
