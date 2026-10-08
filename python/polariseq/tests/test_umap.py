"""UMAP layout quality: cluster purity, trustworthiness, determinism.

The layout is judged the way the handbook prescribes: a 20k Gaussian
mixture with known clusters must stay separated (leave-self-out kNN purity
of the 2-D layout) and preserve input neighborhoods (standard
trustworthiness, within 0.01 of umap-learn's layout on the same data).
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

K = 15


def mixture(n: int, d: int = 50, n_clusters: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d)) * 12.0
    labels = rng.integers(0, n_clusters, size=n)
    X = (centers[labels] + rng.normal(size=(n, d))).astype(np.float32)
    return X, labels


def knn_purity(embedding: np.ndarray, labels: np.ndarray, k: int = K) -> float:
    """Fraction of points whose leave-self-out k-NN majority label in the
    2-D layout matches their own label (chunked pairwise distances)."""
    E = np.asarray(embedding, dtype=np.float64)
    n = len(E)
    norms = (E**2).sum(1)
    preds = np.empty(n, dtype=np.int64)
    for start in range(0, n, 2048):
        block = E[start : start + 2048]
        d2 = norms[None, :] + (block**2).sum(1)[:, None] - 2.0 * block @ E.T
        rows = np.arange(start, min(start + 2048, n))
        d2[np.arange(len(rows)), rows] = np.inf  # leave self out
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        for r, nn_row in zip(rows, nn, strict=True):
            vals, counts = np.unique(labels[nn_row], return_counts=True)
            preds[r] = vals[np.argmax(counts)]
    return float((preds == labels).mean())


def trustworthiness(X: np.ndarray, embedding: np.ndarray, k: int = K) -> float:
    """Standard trustworthiness (Venna & Kaski / sklearn formulation):
    penalizes input-far points appearing among the output k nearest
    neighbours. Input-space ranks are computed from the full pairwise
    distances, chunked over rows."""
    X = np.asarray(X, dtype=np.float64)
    E = np.asarray(embedding, dtype=np.float64)
    n = len(X)
    xnorms = (X**2).sum(1)
    enorms = (E**2).sum(1)
    total = 0.0
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        xb, eb = X[start:stop], E[start:stop]
        xd = xnorms[None, :] + (xb**2).sum(1)[:, None] - 2.0 * xb @ X.T
        ed = enorms[None, :] + (eb**2).sum(1)[:, None] - 2.0 * eb @ E.T
        for r in range(stop - start):
            xi_d = xd[r].copy()
            xi_d[start + r] = np.inf
            order = np.argsort(xi_d)  # rank 0..n-2 of every other point
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            eo_d = ed[r].copy()
            eo_d[start + r] = np.inf
            out_nn = np.argpartition(eo_d, k)[:k]
            s = 0.0
            for j in out_nn:
                if rank[j] >= k:
                    s += rank[j] + 1 - k
            total += s
    denom = n * k * (2.0 * n - 3.0 * k - 1.0)
    return float(1.0 - 2.0 * total / denom)


def run_umap(X: np.ndarray, *, deterministic: bool = False, seed: int = 42):
    adata = ps.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
    adata.obsm["X_pca"] = X
    ps.pp.neighbors(adata, n_neighbors=K, seed=0)
    ps.tl.umap(adata, seed=seed, deterministic=deterministic)
    return np.asarray(adata.obsm["X_umap"], dtype=np.float64)


class TestUmap:
    def test_cluster_purity(self) -> None:
        X, labels = mixture(20_000)
        emb = run_umap(X)
        purity = knn_purity(emb, labels)
        assert purity >= 0.95, f"2-D layout kNN purity {purity:.3f}"

    def test_trustworthiness_matches_umap_learn(self) -> None:
        umap = pytest.importorskip("umap")
        X, _labels = mixture(10_000, seed=1)
        ours = run_umap(X)
        ref = umap.UMAP(
            n_neighbors=K, min_dist=0.1, spread=1.0, random_state=0
        ).fit_transform(X)
        t_ours = trustworthiness(X, ours)
        t_ref = trustworthiness(X, ref)
        assert abs(t_ours - t_ref) <= 0.01, (
            f"trustworthiness ours {t_ours:.4f} vs umap-learn {t_ref:.4f}"
        )

    def test_deterministic_mode_is_bit_identical(self) -> None:
        X, _labels = mixture(3_000, seed=2)
        a = run_umap(X, deterministic=True)
        b = run_umap(X, deterministic=True)
        np.testing.assert_array_equal(a, b)

    def test_parallel_mode_runs(self) -> None:
        X, _labels = mixture(3_000, seed=3)
        emb = run_umap(X)
        assert emb.shape == (3_000, 2)
        assert np.isfinite(emb).all()
        span = emb.max(0) - emb.min(0)
        assert (span > 1).all() and (span < 1_000).all(), f"span {span}"


class TestUmapInit:
    """Where the cells start: the default spectral start, a random one, the first two principal
    components, or the caller's array (the last two scaled to [0, 10], umap-learn's convention)."""

    def _adata(self, n: int = 3_000, seed: int = 4):
        X, labels = mixture(n, seed=seed)
        adata = ps.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["X_pca"] = X
        ps.pp.neighbors(adata, n_neighbors=K, seed=0)
        return adata, labels

    @pytest.mark.parametrize("init", ["spectral", "random", "pca", "kmeans", "multilevel", "leiden"])
    def test_named_starts_separate_clusters(self, init) -> None:
        adata, labels = self._adata()
        if init == "leiden":
            ps.tl.leiden(adata, seed=0)
        ps.tl.umap(adata, seed=0, deterministic=True, init=init)
        emb = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
        assert np.isfinite(emb).all()
        assert knn_purity(emb, labels) >= 0.95

    def test_given_start_is_used(self) -> None:
        # with a single epoch the layout stays close to where it started
        adata, _ = self._adata()
        rng = np.random.default_rng(0)
        start = rng.uniform(size=(adata.n_obs, 2))
        ps.tl.umap(adata, seed=0, deterministic=True, init=start, n_epochs=1)
        emb = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
        scaled = 10.0 * (start - start.min(0)) / (start.max(0) - start.min(0))
        assert np.corrcoef(emb[:, 0], scaled[:, 0])[0, 1] > 0.9

    def test_bad_init(self) -> None:
        adata, _ = self._adata(n=500)
        with pytest.raises(ValueError):
            ps.tl.umap(adata, init="kmeans-typo")
        with pytest.raises(ValueError):
            ps.tl.umap(adata, init=np.zeros((10, 2)))


def test_kmeans_start_places_each_cell_at_its_centroid() -> None:
    from polariseq._polariseq import umap_kmeans_start
    X, labels = mixture(5_000, seed=6)
    start = umap_kmeans_start(np.ascontiguousarray(X), 50, 0)
    assert start.shape == (5_000, 2) and np.isfinite(start).all()
    # at most 50 distinct positions, and cells of one mixture component share few of them
    assert len(np.unique(start, axis=0)) <= 50
    same = [len(np.unique(start[labels == c], axis=0)) for c in np.unique(labels)]
    assert max(same) < 50
    # deterministic for a seed
    np.testing.assert_array_equal(start, umap_kmeans_start(np.ascontiguousarray(X), 50, 0))


def _rolled_sheet(n: int = 6_000, seed: int = 0) -> np.ndarray:
    """A connected curved manifold (a noisy rolled sheet in 50 dimensions), whose graph has
    well-separated leading eigenvalues, so that its eigenvectors are defined. (A mixture of
    separated clusters is not: its graph has several eigenvalues equal to one.)"""
    rng = np.random.default_rng(seed)
    t, h = rng.uniform(0, 3 * np.pi, n), rng.uniform(0, 10, n)
    x = np.zeros((n, 50), dtype=np.float32)
    x[:, 0], x[:, 1], x[:, 2] = t * np.cos(t), h, t * np.sin(t)
    return x + rng.normal(0, 0.3, x.shape).astype(np.float32)


def test_multilevel_start_approaches_the_graphs_eigenvectors() -> None:
    """From the group graph's exact solution, smoothing moves the start towards the leading
    eigenvectors of the cell graph, and it starts nearer to them than the default start ends."""
    import scipy.sparse as sp
    from polariseq._polariseq import (
        umap_kmeans_labels,
        umap_multilevel_layout,
        umap_spectral_layout,
    )
    from scipy.sparse.linalg import eigsh

    X = _rolled_sheet()
    adata = ps.AnnData(X=np.zeros((len(X), 1), dtype=np.float32))
    adata.obsm["X_pca"] = X
    ps.pp.neighbors(adata, n_neighbors=K, seed=0)
    ps.tl.leiden(adata, seed=0)   # builds the fuzzy connectivities
    g = adata.obsp["connectivities"].tocsr()
    g.sort_indices()
    deg = np.asarray(g.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(deg))
    _, vec = eigsh((dinv @ g @ dinv).astype(np.float64), k=3, which="LA", tol=1e-10)
    exact = vec[:, :2]   # the two after the trivial one (eigsh: ascending)
    t = np.sqrt(deg)

    def angle(layout):
        # undo each axis's shift along the trivial eigenvector only (an exact eigenvector is unchanged)
        m = np.asarray(layout, np.float64)
        m = m - ((m.T @ t) / t.sum())[None, :]
        sv = np.linalg.svd(np.linalg.qr(m)[0].T @ np.linalg.qr(exact)[0], compute_uv=False)
        return float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))

    csr = (g.indptr.astype(np.uint32), g.indices.astype(np.uint32), g.data.astype(np.float32), len(X))
    labels = np.asarray(umap_kmeans_labels(np.ascontiguousarray(X), 100, 0), dtype=np.uint32)
    a0 = angle(umap_multilevel_layout(*csr, labels, 100, 0))
    a40 = angle(umap_multilevel_layout(*csr, labels, 100, 40))
    default = angle(umap_spectral_layout(*csr, 0))
    assert a40 <= a0 + 1e-6, (a0, a40)
    assert a40 < 10.0, f"after 40 steps the start is {a40:.1f} degrees from the eigenvectors"
    assert a40 <= default + 1.0, f"multilevel {a40:.1f} vs default spectral {default:.1f} degrees"
