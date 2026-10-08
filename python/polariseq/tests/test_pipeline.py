"""End-to-end pipeline integration tests.

Tests the full Polariseq pipeline from Python:
    read_h5ad → pp.pca → pp.neighbors → tl.leiden

And cross-validates each step against scanpy on the same data.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

FIXTURE = "tests/data/sparse_100x200.h5ad"


class TestPipeline:
    """Full-pipeline integration: each step validated against scanpy."""

    def test_read_h5ad_loads_matrix_and_metadata(self) -> None:
        adata = ps.read_h5ad(FIXTURE)
        assert adata.shape == (100, 200)
        assert adata.X.dtype == np.float32
        assert len(adata.obs_names) == 100
        assert len(adata.var_names) == 200
        assert "n_counts" in adata.obs
        assert "n_cells" in adata.var

    def test_pca_produces_correct_embedding(self) -> None:
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata, n_components=10)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (100, 10)
        assert adata.uns["pca"]["n_components"] == 10
        assert len(adata.uns["pca"]["variance_ratio"]) == 10

    def test_pca_centered_matches_scanpy(self) -> None:
        """Centered dense PCA should match scanpy to high precision."""
        # scanpy and anndata are comparison-only dependencies, not runtime
        # ones. Skip rather than fail where they are absent: INSTALL.md tells
        # a new user to run this suite to verify their build, and a red result
        # for a missing optional package tells them their install is broken
        # when it is fine.
        pytest.importorskip("anndata")
        pytest.importorskip("scanpy")
        import anndata as ad
        import scanpy as sc

        # Our PCA
        adata_ps = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata_ps, n_components=5, center=True)

        # scanpy PCA (densify for fair comparison)
        adata_sc = ad.read_h5ad(FIXTURE)
        sc.pp.pca(adata_sc, n_comps=5, svd_solver="arpack")

        # Variance ratios should match closely.
        our_vr = adata_ps.uns["pca"]["variance_ratio"]
        scanpy_vr = adata_sc.uns["pca"]["variance_ratio"]
        for j in range(5):
            assert abs(our_vr[j] - scanpy_vr[j]) < 1e-4, (
                f"PC{j} variance_ratio mismatch: {our_vr[j]} vs {scanpy_vr[j]}"
            )

    def test_neighbors_produces_correct_graph(self) -> None:
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata, n_components=10)
        ps.pp.neighbors(adata, n_neighbors=15)
        assert "distances" in adata.obsp
        # 100 cells × 14 neighbors each (self excluded).
        n_edges = adata.obsp["distances"].nnz
        assert n_edges == 100 * 14, f"expected 1400 edges, got {n_edges}"

    def test_neighbors_match_scanpy(self) -> None:
        """Neighbor sets should be identical to scanpy."""
        pytest.importorskip("anndata")
        pytest.importorskip("scanpy")
        import anndata as ad
        import scanpy as sc

        adata_ps = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata_ps, n_components=10)
        ps.pp.neighbors(adata_ps, n_neighbors=15)

        adata_sc = ad.read_h5ad(FIXTURE)
        sc.pp.pca(adata_sc, n_comps=10, svd_solver="arpack")
        sc.pp.neighbors(adata_sc, n_neighbors=15)

        # Compare neighbor sets for first 10 cells.
        ps_dist = adata_ps.obsp["distances"]
        sc_dist = adata_sc.obsp["distances"]
        total_overlap = 0
        total = 0
        for i in range(10):
            ps_set = set(ps_dist.getrow(i).indices.tolist())
            sc_set = set(sc_dist.getrow(i).indices.tolist())
            total_overlap += len(ps_set & sc_set)
            total += len(sc_set)
        overlap_frac = total_overlap / total
        assert overlap_frac > 0.95, (
            f"neighbor overlap only {overlap_frac:.1%} (expected >95%)"
        )

    def test_leiden_produces_clusters(self) -> None:
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata, n_components=10)
        ps.pp.neighbors(adata, n_neighbors=15)
        ps.tl.leiden(adata, resolution=1.0)
        assert "leiden" in adata.obs
        assert len(adata.obs["leiden"]) == 100
        assert adata.uns["leiden"]["n_communities"] >= 1
        # Labels should be contiguous (0, 1, 2, ...).
        labels = adata.obs["leiden"]
        max_label = labels.max()
        assert max_label + 1 == adata.uns["leiden"]["n_communities"]

    def test_full_pipeline_runs_without_error(self) -> None:
        """The canonical scRNA-seq pipeline, end to end."""
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata, n_components=10)
        ps.pp.neighbors(adata, n_neighbors=15)
        ps.tl.leiden(adata, resolution=1.0)

        # Every step produced its output.
        assert "X_pca" in adata.obsm
        assert "distances" in adata.obsp
        assert "leiden" in adata.obs
        assert adata.obsm["X_pca"].shape == (100, 10)
        assert adata.obs["leiden"].shape == (100,)

    def test_pipeline_is_deterministic(self) -> None:
        """Same seed → same clustering."""
        adata1 = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata1, n_components=10, seed=42)
        ps.pp.neighbors(adata1, n_neighbors=15)
        ps.tl.leiden(adata1, resolution=1.0, seed=42)

        adata2 = ps.read_h5ad(FIXTURE)
        ps.pp.pca(adata2, n_components=10, seed=42)
        ps.pp.neighbors(adata2, n_neighbors=15)
        ps.tl.leiden(adata2, resolution=1.0, seed=42)

        np.testing.assert_array_equal(
            adata1.obs["leiden"], adata2.obs["leiden"]
        )

def test_gzipped_fixture_reads_identically() -> None:
    """The chunk-parallel fast read (POLARISEQ_FASTREAD=1) must produce the
    same matrix as the library path on a gzip+shuffle file (the layout
    scanpy/anndata writes)."""
    import os

    import numpy as np
    import polariseq as ps

    gz = ps.read_h5ad("tests/data/sparse_gzip_100x200.h5ad", lazy=True)
    gz._materialize()
    os.environ["POLARISEQ_FASTREAD"] = "1"
    try:
        fast = ps.read_h5ad("tests/data/sparse_gzip_100x200.h5ad", lazy=True)
        fast._materialize()
    finally:
        os.environ.pop("POLARISEQ_FASTREAD", None)
    np.testing.assert_array_equal(gz.X.indptr, fast.X.indptr)
    np.testing.assert_array_equal(gz.X.indices, fast.X.indices)
    np.testing.assert_array_equal(gz.X.data, fast.X.data)

class TestHarmony:
    """Harmony as harmonypy 2 runs it: soft clustering with the diversity
    penalty, then a ridge correction per cluster. Hard k-means, used before,
    gave each batch its own clusters on well-separated groups with a large
    shift and corrected nothing there; soft assignment does not."""

    @staticmethod
    def _data(shift_scale: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_per, n_pcs, n_types = 800, 20, 4
        types = rng.integers(0, n_types, n_per * 2)
        batches = np.repeat([0, 1], n_per)
        centres = rng.normal(0, 6.0, (n_types, n_pcs))
        shift = rng.normal(0, shift_scale, (2, n_pcs))
        emb = (centres[types] + shift[batches]
               + rng.normal(0, 1.0, (n_per * 2, n_pcs))).astype(np.float32)
        return emb, batches, types

    @staticmethod
    def _adata(emb, batches):
        a = ps.AnnData(X=np.zeros((len(emb), 1), dtype=np.float32))
        a.obsm["X_pca"] = emb.copy()
        a.obs["batch"] = [str(b) for b in batches]
        return a

    @classmethod
    def _gap_removed(cls, emb, batches) -> float:
        """Fraction of the between-batch centroid distance the call removes."""
        a = cls._adata(emb, batches)
        before = np.linalg.norm(emb[batches == 0].mean(0) - emb[batches == 1].mean(0))
        ps.pp.harmony_integrate(a, batch_key="batch")
        out = np.asarray(a.obsm["X_pca_harmony"])
        after = np.linalg.norm(out[batches == 0].mean(0) - out[batches == 1].mean(0))
        return float(1.0 - after / before)

    def test_small_batch_effect_is_corrected(self) -> None:
        emb, batches, _ = self._data(0.5)
        assert self._gap_removed(emb, batches) > 0.5

    def test_well_separated_groups_with_a_large_shift_are_aligned(self) -> None:
        """Where hard k-means removed nothing; harmonypy removes ~66 %."""
        emb, batches, _ = self._data(3.0)
        assert self._gap_removed(emb, batches) > 0.5

    def test_the_corrected_embedding_is_added_and_the_pca_kept(self) -> None:
        emb, batches, _ = self._data(0.5)
        a = self._adata(emb, batches)
        ps.pp.harmony_integrate(a, batch_key="batch")
        np.testing.assert_array_equal(np.asarray(a.obsm["X_pca"]), emb)
        out = np.asarray(a.obsm["X_pca_harmony"])
        assert out.shape == emb.shape and not np.array_equal(out, emb)
        info = a.uns["harmony"]
        assert info["n_batches"] == 2 and info["n_clusters"] == 53
        assert info["iterations"] == len(info["objective"]) - 1 >= 1
        ps.pp.neighbors(a, use_rep="X_pca_harmony")
        assert a.obsp["connectivities"].shape == (len(emb), len(emb))

    def test_agrees_with_harmonypy(self) -> None:
        hm = pytest.importorskip("harmonypy")
        emb, batches, _ = self._data(1.5, seed=1)
        a = self._adata(emb, batches)
        ps.pp.harmony_integrate(a, batch_key="batch")
        ours = np.asarray(a.obsm["X_pca_harmony"])
        ref = np.asarray(hm.run_harmony(emb.astype(np.float64), {"batch": batches}, "batch",
                                        random_state=0, verbose=False).Z_corr)
        r = [np.corrcoef(ours[:, i], ref[:, i])[0, 1] for i in range(emb.shape[1])]
        assert min(r) > 0.99, r

    def test_several_batch_variables_and_a_seed(self) -> None:
        emb, batches, types = self._data(1.0)
        a = self._adata(emb, batches)
        a.obs["lane"] = [f"l{t % 3}" for t in range(len(emb))]
        ps.pp.harmony_integrate(a, ["batch", "lane"], seed=3, key_added="corrected")
        first = np.asarray(a.obsm["corrected"]).copy()
        assert a.uns["harmony"]["n_batches"] == [2, 3]
        ps.pp.harmony_integrate(a, ["batch", "lane"], seed=3, key_added="corrected")
        np.testing.assert_array_equal(np.asarray(a.obsm["corrected"]), first)

    def test_missing_or_single_batches_are_refused(self) -> None:
        emb, batches, _ = self._data(0.5)
        a = self._adata(emb, batches)
        a.obs["one"] = ["x"] * len(emb)
        with pytest.raises(ValueError, match=">= 2 batches"):
            ps.pp.harmony_integrate(a, "one")
        a.obs["gaps"] = [None if i == 5 else "x" if i % 2 else "y" for i in range(len(emb))]
        with pytest.raises(ValueError, match="missing values"):
            ps.pp.harmony_integrate(a, "gaps")


def test_filter_cells_max_pct_mt() -> None:
    """The mitochondrial cutoff of the standard QC (e.g. scanpy's PBMC 3k
    tutorial keeps cells under 5 %) drops exactly the cells above it."""
    import scipy.sparse as sp

    x = np.array([
        [10, 0, 0],   # 0 % mitochondrial
        [8, 2, 0],    # 20 %
        [9, 1, 0],    # 10 %
        [5, 0, 5],    # 0 %
    ], dtype=np.float32)
    adata = ps.AnnData(X=sp.csr_matrix(x), var_names=["A", "MT-CO1", "B"])
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, max_pct_mt=10.0)
    assert adata.n_obs == 3
    assert list(adata.obs_names) == ["cell_0", "cell_2", "cell_3"]


def test_filter_cells_max_pct_mt_needs_qc() -> None:
    adata = ps.AnnData(X=np.ones((2, 2), dtype=np.float32))
    with pytest.raises(KeyError, match="pct_counts_mt"):
        ps.pp.filter_cells(adata, max_pct_mt=5.0)
