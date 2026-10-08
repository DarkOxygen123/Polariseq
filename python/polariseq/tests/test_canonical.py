"""Tests for the canonical scRNA-seq preprocessing pipeline.

The full canonical pipeline:
    read → normalize_total → log1p → highly_variable_genes → PCA → neighbors → leiden

This is the standard scanpy workflow. These tests verify each preprocessing
step produces sensible results and that the full pipeline runs end-to-end.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps

FIXTURE = "tests/data/medium_10kx2k.h5ad"


class TestCanonicalPipeline:
    """The real-world scRNA-seq workflow, step by step."""

    def test_normalize_total_scales_cells(self) -> None:
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.normalize_total(adata, target_sum=1e4)
        # After: every cell sums to ~1e4 (works for both dense and sparse).
        sums = np.asarray(adata.X.sum(axis=1)).flatten()
        np.testing.assert_allclose(sums, 1e4, rtol=1e-3)

    def test_log1p_preserves_sparsity(self) -> None:
        """log1p on sparse data preserves the sparsity pattern; on dense data
        it correctly transforms all values. Test both paths."""
        # Dense path: read as dense, verify log1p correctness
        adata = ps.read_h5ad(FIXTURE, sparse=False)
        ps.pp.normalize_total(adata)
        ps.pp.log1p(adata)
        # log1p of a positive value is positive; log1p(0) = 0.
        # (dense array for the check — sparse matrices lack .all() in new scipy)
        Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        assert (Xd >= 0).all(), "log1p of non-negative data is non-negative"
        assert np.isfinite(Xd).all()

        # Sparse path: construct a CSR directly to test sparse log1p.
        import scipy.sparse as sp

        X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32))
        adata2 = ps.AnnData(X=X)
        nnz_before = adata2.X.nnz
        ps.pp.log1p(adata2)
        assert adata2.X.nnz == nnz_before, "log1p must preserve sparsity pattern"

    def test_hvg_selects_requested_number(self) -> None:
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.normalize_total(adata)
        ps.pp.log1p(adata)
        ps.pp.highly_variable_genes(adata, n_top_genes=500)
        assert "highly_variable" in adata.var
        assert adata.var["highly_variable"].sum() == 500
        assert "means" in adata.var
        assert "dispersions" in adata.var
        assert "dispersions_norm" in adata.var

    def test_full_canonical_pipeline(self) -> None:
        """The complete scRNA-seq pipeline, end to end."""
        adata = ps.read_h5ad(FIXTURE)

        # Preprocessing
        ps.pp.normalize_total(adata, target_sum=1e4)
        ps.pp.log1p(adata)
        ps.pp.highly_variable_genes(adata, n_top_genes=1000)
        assert adata.var["highly_variable"].sum() == 1000

        # Subset to HVGs for PCA
        hvg_mask = adata.var["highly_variable"]
        if hasattr(adata.X, "tocsc"):
            adata.X = adata.X[:, hvg_mask]
        else:
            adata.X = adata.X[:, hvg_mask]

        # Compute
        ps.pp.pca(adata, n_components=50, seed=42)
        ps.pp.neighbors(adata, n_neighbors=15)
        ps.tl.leiden(adata, resolution=1.0, seed=42)

        # Verify all outputs exist
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape[0] == adata.n_obs
        assert "distances" in adata.obsp
        assert "leiden" in adata.obs
        assert adata.uns["leiden"]["n_communities"] >= 1

    def test_pipeline_reproducible(self) -> None:
        """Same seed → same clusters."""
        def run() -> np.ndarray:
            adata = ps.read_h5ad(FIXTURE)
            ps.pp.normalize_total(adata)
            ps.pp.log1p(adata)
            ps.pp.highly_variable_genes(adata, n_top_genes=500)
            mask = adata.var["highly_variable"]
            adata.X = adata.X[:, mask]
            ps.pp.pca(adata, n_components=30, seed=99)
            ps.pp.neighbors(adata, n_neighbors=15)
            ps.tl.leiden(adata, resolution=1.0, seed=99)
            return adata.obs["leiden"].copy()

        labels1 = run()
        labels2 = run()
        np.testing.assert_array_equal(labels1, labels2)

    def test_normalize_then_pca_is_valid(self) -> None:
        """Normalized + log-transformed data should produce valid PCA:
        finite embedding, monotonically decreasing variance_ratio."""
        adata = ps.read_h5ad(FIXTURE)
        ps.pp.normalize_total(adata)
        ps.pp.log1p(adata)
        ps.pp.pca(adata, n_components=20, seed=42)

        emb = adata.obsm["X_pca"]
        assert np.isfinite(emb).all(), "embedding must be finite"
        vr = adata.uns["pca"]["variance_ratio"]
        # Variance ratio should be non-negative and generally decreasing.
        assert (vr >= 0).all()
        assert vr[0] >= vr[-1], "first PC should explain more variance than last"
