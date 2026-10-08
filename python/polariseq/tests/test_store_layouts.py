"""The out-of-core store is written compact by default (byte planes deflated
per chunk of rows; see crates/polariseq-core/src/store.rs). It is lossless:
an analysis on a compact store must give the same numbers, bit for bit, as
one on a plain store, while the copy of the counts takes a fraction of the
disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polariseq as ps

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def _analyse(tmp_path: Path, layout: str, monkeypatch):
    monkeypatch.setenv("POLARISEQ_STORE", layout)
    ps.project(tmp_path / layout)
    a = ps.read_h5ad(FIXTURE, mode="out-of-core", name=layout)
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=100)
    ps.pp.filter_genes(a, min_cells=3)
    ps.pp.normalize_total(a, target_sum=1e4)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=500, subset=True)
    ps.pp.pca(a, n_components=20, seed=0)
    ps.pp.neighbors(a, n_neighbors=15, seed=0)
    ps.tl.leiden(a, seed=0)
    matrix = tmp_path / layout / "runs" / layout / "matrix"
    size = sum(f.stat().st_size for f in matrix.glob("counts.*"))
    files = {f.suffix for f in matrix.glob("counts.*")}
    return np.array(a.obsm["X_pca"]), np.asarray(a.obs["leiden"]), size, files


def test_compact_store_gives_the_same_results_on_less_disk(proj, tmp_path, monkeypatch) -> None:
    pca_p, lab_p, size_p, files_p = _analyse(tmp_path, "plain", monkeypatch)
    pca_c, lab_c, size_c, files_c = _analyse(tmp_path, "compact", monkeypatch)
    assert {".data", ".indices"} <= files_p and ".cz" not in files_p
    assert {".cz", ".czi"} <= files_c and ".data" not in files_c
    np.testing.assert_array_equal(pca_p, pca_c)
    np.testing.assert_array_equal(lab_p, lab_c)
    assert size_c * 2 < size_p, f"compact {size_c:,} bytes vs plain {size_p:,}"
