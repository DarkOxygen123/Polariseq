"""PCA on the Rust-owned sparse matrix: real (centered) PCA, reproducible.

The reference is an exact SVD (numpy, float64) of the explicitly centered
matrix. Randomized SVD with implicit centering must land in the same subspace
(principal angles < 1 degree on well-separated leading components) with the
same variance ratios — and must be bit-identical from run to run.

The structured synthetic dataset matters: the ``medium_10kx2k`` fixture is
unstructured noise with a flat spectrum, on which *no* truncated solver has a
well-defined answer.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest
import scipy.sparse as sp

FIXTURE_10K = "tests/data/medium_10kx2k.h5ad"


def structured_counts(n: int = 4000, m: int = 1500, rank: int = 6, seed: int = 0) -> sp.csr_matrix:
    """Sparse non-negative counts with ``rank`` well-separated programs."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, rank))
    v = rng.normal(size=(m, rank))
    # Program weights sit well above the noise floor (sigma ~ 28 for this
    # shape/noise level), so all `rank` components are genuinely separated.
    s = np.array([600, 400, 250, 150, 100, 60.0])[:rank]
    x = (u * s) @ v.T + rng.normal(size=(n, m)) * 0.5
    x = np.where(rng.random((n, m)) < 0.3, np.abs(x), 0.0).astype(np.float32)
    return sp.csr_matrix(x)


def principal_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


class TestSparsePca:
    def test_matches_exact_svd(self) -> None:
        x = structured_counts()
        adata = ps.AnnData(X=x)
        ps.pp.pca(adata, n_components=10, seed=0)
        emb = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        vr = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float64)

        xc = x.toarray().astype(np.float64)
        xc -= xc.mean(axis=0)
        u, s, _ = np.linalg.svd(xc, full_matrices=False)
        emb_ref = u[:, :10] * s[:10]
        vr_ref = s[:10] ** 2 / (s**2).sum()

        # Centered: column means of the embedding are ~0 relative to spread.
        assert (np.abs(emb.mean(axis=0)) / emb.std(axis=0)).max() < 1e-3
        # Singular values / variance ratios agree.
        np.testing.assert_allclose(vr[:6], vr_ref[:6], rtol=5e-3)
        np.testing.assert_allclose(np.linalg.norm(emb, axis=0)[:6], s[:6], rtol=5e-3)
        # Same subspace, component by component, on the well-separated part.
        for k in (1, 2, 4, 6):
            angles = principal_angles_deg(emb[:, :k], emb_ref[:, :k])
            assert angles.max() < 1.0, f"top-{k} principal angles (deg): {np.round(angles, 3)}"

    def test_scanpy_arpack_agrees(self) -> None:
        sc = pytest.importorskip("scanpy")
        ad = pytest.importorskip("anndata")
        x = structured_counts(seed=1)
        ours = ps.AnnData(X=x)
        ps.pp.normalize_total(ours, target_sum=1e4)
        ps.pp.log1p(ours)
        ps.pp.pca(ours, n_components=10, seed=0)
        ref = ad.AnnData(X=x.copy())
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        sc.pp.pca(ref, n_comps=10, svd_solver="arpack", zero_center=True)
        emb = np.asarray(ours.obsm["X_pca"], dtype=np.float64)
        emb_ref = np.asarray(ref.obsm["X_pca"], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(ours.uns["pca"]["variance_ratio"])[:6],
            np.asarray(ref.uns["pca"]["variance_ratio"])[:6],
            rtol=1e-2,
        )
        assert principal_angles_deg(emb[:, :6], emb_ref[:, :6]).max() < 1.0

    def test_is_bit_identical_across_runs(self) -> None:
        def run():
            adata = ps.read_h5ad(FIXTURE_10K)
            ps.pp.normalize_total(adata, target_sum=1e4)
            ps.pp.log1p(adata)
            ps.pp.pca(adata, n_components=30, seed=7)
            return adata

        a, b = run(), run()
        np.testing.assert_array_equal(a.obsm["X_pca"], b.obsm["X_pca"])
        np.testing.assert_array_equal(a.uns["pca"]["variance_ratio"], b.uns["pca"]["variance_ratio"])

    def test_uncentered_is_available(self) -> None:
        adata = ps.AnnData(X=structured_counts(n=2500, m=1200, seed=2))
        ps.pp.pca(adata, n_components=5, center=False)
        pc1 = np.asarray(adata.obsm["X_pca"], dtype=np.float64)[:, 0]
        # Without centering PC1 captures the mean direction: its mean is far
        # from zero relative to its spread (a centered PC has mean ~0).
        assert abs(pc1.mean()) > pc1.std()


class TestScaledPca:
    """``scale=True`` must equal scanpy's ``pp.scale`` followed by ``tl.pca``."""

    @pytest.mark.parametrize("max_value", [None, 10.0, 2.0])
    def test_matches_scanpy_scale_then_pca(self, max_value: float | None) -> None:
        sc = pytest.importorskip("scanpy")
        ad = pytest.importorskip("anndata")
        x = structured_counts(n=3000, m=1200, seed=3)
        ours = ps.AnnData(X=x)
        ps.pp.normalize_total(ours, target_sum=1e4)
        ps.pp.log1p(ours)
        ps.pp.pca(ours, n_components=10, scale=True, max_value=max_value, seed=0)
        ref = ad.AnnData(X=x.copy())
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        sc.pp.scale(ref, max_value=max_value)
        sc.pp.pca(ref, n_comps=10, svd_solver="full")
        emb = np.asarray(ours.obsm["X_pca"], dtype=np.float64)
        emb_ref = np.asarray(ref.obsm["X_pca"], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(ours.uns["pca"]["variance_ratio"])[:6],
            np.asarray(ref.uns["pca"]["variance_ratio"])[:6],
            rtol=1e-3,
        )
        assert principal_angles_deg(emb[:, :6], emb_ref[:, :6]).max() < 0.1

    def test_leaves_x_sparse_and_unchanged(self) -> None:
        adata = ps.AnnData(X=structured_counts(n=1500, m=800, seed=4))
        before = adata.X.toarray()
        ps.pp.pca(adata, n_components=5, scale=True, max_value=10.0)
        assert isinstance(adata.X, ps.SparseMatrix)
        np.testing.assert_array_equal(adata.X.toarray(), before)

    def test_rejects_max_value_without_scale(self) -> None:
        adata = ps.AnnData(X=structured_counts(n=500, m=300, seed=5))
        with pytest.raises(ValueError, match="scale=True"):
            ps.pp.pca(adata, n_components=5, max_value=10.0)
