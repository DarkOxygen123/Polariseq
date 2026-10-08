"""SparseMatrix — the Rust-owned CSR handle behind ``AnnData.X``.

Every kernel is checked against a numpy/scipy reference on random count-like
data; HVG is checked against scanpy's own ``flavor='seurat'`` on the 10k
fixture, because agreeing with the reference implementation is the point.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest
import scipy.sparse as sp
from polariseq import SparseMatrix

FIXTURE_SMALL = "tests/data/sparse_100x200.h5ad"
FIXTURE_10K = "tests/data/medium_10kx2k.h5ad"


def rand_csr(n: int = 300, m: int = 120, density: float = 0.1, seed: int = 0) -> sp.csr_matrix:
    """Random count-like CSR with positive integer values."""
    rng = np.random.default_rng(seed)
    x = sp.random(n, m, density=density, format="csr", random_state=rng, dtype=np.float32)
    x.data = np.rint(x.data * 10 + 1).astype(np.float32)
    x.eliminate_zeros()
    x.sort_indices()
    return x


class TestConstruction:
    def test_scipy_round_trip(self) -> None:
        x = rand_csr()
        m = SparseMatrix.from_scipy(x)
        assert m.shape == x.shape
        assert m.nnz == x.nnz
        assert m.ndim == 2
        assert m.dtype == np.float32
        np.testing.assert_array_equal(m.toarray(), x.toarray())
        back = m.to_scipy()
        assert (back != x).nnz == 0

    def test_accepts_int64_indices(self) -> None:
        x = rand_csr()
        x.indices = x.indices.astype(np.int64)
        x.indptr = x.indptr.astype(np.int64)
        m = SparseMatrix.from_scipy(x)
        np.testing.assert_array_equal(m.toarray(), x.toarray())

    def test_from_dense_drops_zeros(self) -> None:
        d = np.array([[1, 0, 2], [0, 0, 0], [0, 3, 4]], dtype=np.float32)
        m = SparseMatrix.from_dense(d)
        assert m.nnz == 4
        np.testing.assert_array_equal(m.toarray(), d)

    def test_anndata_coerces_scipy_once(self) -> None:
        x = rand_csr()
        adata = ps.AnnData(X=x)
        assert isinstance(adata.X, SparseMatrix)
        adata.X = x  # assignment goes through the same coercion
        assert isinstance(adata.X, SparseMatrix)
        assert adata.shape == x.shape

    def test_reader_returns_rust_matrix(self) -> None:
        adata = ps.read_h5ad(FIXTURE_SMALL)
        assert isinstance(adata.X, SparseMatrix)
        assert adata.X.shape == (100, 200)
        assert "SparseMatrix" in repr(adata.X)

    def test_copy_is_independent(self) -> None:
        m = SparseMatrix.from_scipy(rand_csr())
        c = m.copy()
        c.log1p()
        assert not np.allclose(m.toarray(), c.toarray())


class TestIndexing:
    @pytest.fixture()
    def pair(self):
        x = rand_csr(200, 60, seed=3)
        return x, SparseMatrix.from_scipy(x)

    def _check(self, x, m, key) -> None:
        got = m[key].toarray()
        # SparseMatrix indexes rows x cols (outer, like AnnData); scipy would
        # treat two arrays as pointwise pairs, so build the reference in two steps.
        want = (x[key[0]][:, key[1]] if isinstance(key, tuple) else x[key]).toarray()
        assert got.shape == want.shape, key
        np.testing.assert_array_equal(got, want)

    def test_row_slice(self, pair) -> None:
        x, m = pair
        self._check(x, m, slice(10, 50))
        self._check(x, m, slice(None))
        self._check(x, m, (slice(5, 9), slice(None)))

    def test_column_slice_and_both(self, pair) -> None:
        x, m = pair
        self._check(x, m, (slice(None), slice(3, 7)))
        self._check(x, m, (slice(20, 40), slice(0, 10)))

    def test_boolean_masks(self, pair) -> None:
        x, m = pair
        rows = np.zeros(x.shape[0], dtype=bool)
        rows[::3] = True
        cols = np.zeros(x.shape[1], dtype=bool)
        cols[::2] = True
        self._check(x, m, rows)
        self._check(x, m, (rows, slice(None)))
        self._check(x, m, (slice(None), cols))
        self._check(x, m, (rows, cols))

    def test_integer_arrays_keep_order_and_duplicates(self, pair) -> None:
        x, m = pair
        rows = np.array([7, 3, 3, 150], dtype=np.int64)
        cols = [59, 0, 17]
        self._check(x, m, rows)
        self._check(x, m, (slice(None), cols))
        self._check(x, m, (rows, cols))
        # Numpy int32 indices (what pandas/numpy often hand back).
        self._check(x, m, rows.astype(np.int32))

    def test_single_integer_row(self, pair) -> None:
        x, m = pair
        self._check(x, m, 5)
        self._check(x, m, -1)

    def test_errors(self, pair) -> None:
        x, m = pair
        with pytest.raises(IndexError):
            m[x.shape[0]]
        with pytest.raises(ValueError):
            m[np.ones(3, dtype=bool)]
        with pytest.raises(ValueError):
            m[:, [1, 1]]
        with pytest.raises(ValueError):
            m[::2]


class TestReductions:
    def test_sum_and_mean_match_scipy(self) -> None:
        x = rand_csr(150, 40, seed=5)
        m = SparseMatrix.from_scipy(x)
        assert np.isclose(m.sum(), x.sum())
        np.testing.assert_allclose(m.sum(axis=1), np.asarray(x.sum(axis=1)).ravel(), rtol=1e-6)
        np.testing.assert_allclose(m.sum(axis=0), np.asarray(x.sum(axis=0)).ravel(), rtol=1e-6)
        np.testing.assert_allclose(m.mean(axis=1), np.asarray(x.mean(axis=1)).ravel(), rtol=1e-6)
        np.testing.assert_allclose(m.mean(axis=0), np.asarray(x.mean(axis=0)).ravel(), rtol=1e-6)
        assert np.isclose(m.mean(), x.mean())


class TestKernels:
    def test_qc_metrics_match_reference(self) -> None:
        x = rand_csr(400, 80, seed=11)
        d = x.toarray()
        mito = np.zeros(80, dtype=bool)
        mito[[2, 40, 79]] = True
        qc = SparseMatrix.from_scipy(x).qc_metrics(mito)
        np.testing.assert_allclose(qc["n_counts"], d.sum(axis=1), rtol=1e-6)
        np.testing.assert_array_equal(qc["n_genes"], (d > 0).sum(axis=1))
        np.testing.assert_allclose(qc["mt_counts"], d[:, mito].sum(axis=1), rtol=1e-6)
        np.testing.assert_array_equal(qc["gene_n_cells"], (d > 0).sum(axis=0))
        np.testing.assert_allclose(qc["gene_total_counts"], d.sum(axis=0), rtol=1e-9)
        assert SparseMatrix.from_scipy(x).qc_metrics(None)["mt_counts"] is None

    def test_calculate_qc_metrics_populates_obs_and_var(self) -> None:
        adata = ps.read_h5ad(FIXTURE_SMALL)
        adata.var_names = ["MT-" + n if i < 5 else n for i, n in enumerate(adata.var_names)]
        ps.pp.calculate_qc_metrics(adata)
        for key in ("n_counts", "n_genes", "pct_counts_mt"):
            assert key in adata.obs and len(adata.obs[key]) == 100
        for key in ("n_cells", "total_counts", "mean_counts"):
            assert key in adata.var and len(adata.var[key]) == 200
        assert (adata.obs["pct_counts_mt"] >= 0).all()

    def test_normalize_total_and_log1p(self) -> None:
        x = rand_csr(300, 50, seed=2)
        m = SparseMatrix.from_scipy(x)
        before = m.normalize_total(1e4)
        np.testing.assert_allclose(before, np.asarray(x.sum(axis=1)).ravel(), rtol=1e-6)
        sums = m.sum(axis=1)
        nonzero = before > 0
        np.testing.assert_allclose(sums[nonzero], 1e4, rtol=1e-4)
        m.log1p()
        assert m.nnz == x.nnz
        ref = x.toarray() / before[:, None].clip(1e-12) * 1e4
        ref[~nonzero] = 0
        np.testing.assert_allclose(m.toarray(), np.log1p(ref), rtol=1e-5, atol=1e-6)

    def test_hvg_matches_scanpy_seurat(self) -> None:
        sc = pytest.importorskip("scanpy")
        ad = pytest.importorskip("anndata")
        n_top = 300

        ours = ps.read_h5ad(FIXTURE_10K)
        ps.pp.normalize_total(ours, target_sum=1e4)
        ps.pp.log1p(ours)
        ps.pp.highly_variable_genes(ours, n_top_genes=n_top)
        mask_ps = np.asarray(ours.var["highly_variable"])
        assert mask_ps.sum() == n_top

        ref = ad.read_h5ad(FIXTURE_10K)
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        sc.pp.highly_variable_genes(ref, flavor="seurat", n_top_genes=n_top)
        mask_sc = np.asarray(ref.var["highly_variable"])

        overlap = (mask_ps & mask_sc).sum() / n_top
        assert overlap >= 0.95, f"HVG overlap with scanpy only {overlap:.1%}"
        # The per-gene statistics agree too, not just the selection.
        finite = np.isfinite(np.asarray(ref.var["dispersions"]))
        np.testing.assert_allclose(
            np.asarray(ours.var["means"])[finite], np.asarray(ref.var["means"])[finite], rtol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(ours.var["dispersions"])[finite],
            np.asarray(ref.var["dispersions"])[finite],
            rtol=1e-3,
            atol=1e-4,
        )

    def test_hvg_rejects_other_flavors(self) -> None:
        adata = ps.read_h5ad(FIXTURE_SMALL)
        with pytest.raises(NotImplementedError):
            ps.pp.highly_variable_genes(adata, n_top_genes=10, flavor="cell_ranger")

    def test_rank_genes_groups_by_reference(self) -> None:
        adata = ps.read_h5ad(FIXTURE_SMALL)
        adata.obs["grp"] = np.arange(100) % 3
        ps.tl.rank_genes_groups(adata, groupby="grp", n_genes=5)
        assert set(adata.uns["rank_genes_groups"]) == {"0", "1", "2"}
        assert len(adata.uns["rank_genes_groups"]["0"]["names"]) == 5
