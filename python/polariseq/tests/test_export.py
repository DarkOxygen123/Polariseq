"""``AnnData.to_anndata``: results leave Polariseq for Scanpy's ecosystem."""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def test_in_memory_result_round_trips(tmp_path) -> None:
    ad = pytest.importorskip("anndata")
    a = ps.read_h5ad(FIXTURE, lazy=False)
    ps.pp.normalize_total(a, target_sum=1e4)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=500, subset=True)
    ps.pp.pca(a, 20, seed=0)
    ps.pp.neighbors(a, n_neighbors=10, seed=0)
    ps.tl.leiden(a, seed=0)
    out = a.to_anndata()
    assert out.shape == a.shape
    np.testing.assert_array_equal(out.obsm["X_pca"], a.obsm["X_pca"])
    # Clusters leave as Scanpy keeps them: categorical strings, same labels.
    assert out.obs["leiden"].dtype.name == "category"
    np.testing.assert_array_equal(np.asarray(out.obs["leiden"]).astype(int),
                                  np.asarray(a.obs["leiden"]))
    assert out.obsp["connectivities"].shape == (a.n_obs, a.n_obs)
    out.write_h5ad(tmp_path / "x.h5ad")
    back = ad.read_h5ad(tmp_path / "x.h5ad")
    assert back.shape == a.shape


def test_out_of_core_result_has_no_matrix_but_every_gene() -> None:
    pytest.importorskip("anndata")
    ps.set_budget(ram_gb=2, threads=2)
    a = ps.process_diskbacked(FIXTURE, n_hvg=500, n_pcs=20, n_neighbors=10, seed=0)
    out = a.to_anndata()
    assert out.X is None
    assert out.n_obs == a.n_obs
    assert out.n_vars == len(a.var["n_cells"])
    assert out.var["highly_variable"].sum() == 500
    assert "X_pca" in out.obsm and "leiden" in out.obs


def test_filter_cells_computes_what_it_needs() -> None:
    """Without calculate_qc_metrics first, filter_cells must still filter."""
    a = ps.read_h5ad(FIXTURE, lazy=False)
    b = ps.read_h5ad(FIXTURE, lazy=False)
    ps.pp.calculate_qc_metrics(b)
    threshold = int(np.median(np.asarray(b.obs["n_genes"])))
    ps.pp.filter_cells(a, min_genes=threshold)
    ps.pp.filter_cells(b, min_genes=threshold)
    assert a.n_obs == b.n_obs < ps.read_h5ad(FIXTURE, lazy=False).n_obs
