"""kNN graph construction: exact brute force and NN-Descent.

NN-Descent is checked for recall against the exact graph on a Gaussian
mixture large enough that the auto-switch in ``pp.neighbors`` picks it.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
from polariseq._polariseq import neighbors_dense, nndescent_dense

K = 15


def mixture(n: int, d: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(25, d)) * 5
    lab = rng.integers(0, 25, size=n)
    return (centers[lab] + rng.normal(size=(n, d))).astype(np.float32)


def recall(approx_indptr, approx_indices, exact_indices, n: int) -> np.ndarray:
    a = np.asarray(approx_indices).reshape(n, K)
    e = np.asarray(exact_indices).reshape(n, K)
    return np.array([len(set(a[i]) & set(e[i])) for i in range(n)]) / K


class TestNnDescent:
    def test_recall_vs_exact(self) -> None:
        emb = mixture(20_000)
        exact = neighbors_dense(emb, K + 1, "euclidean")  # +1: self is dropped
        approx = nndescent_dense(emb, K, 0, 42)
        r = recall(approx["indptr"], approx["indices"], exact["indices"], emb.shape[0])
        assert r.mean() >= 0.97, f"mean recall {r.mean():.3f}"
        assert np.percentile(r, 5) >= 0.85, f"p05 recall {np.percentile(r, 5):.3f}"

    def test_graph_is_valid_and_deterministic(self) -> None:
        emb = mixture(5000, seed=3)
        a = nndescent_dense(emb, K, 0, 7)
        b = nndescent_dense(emb, K, 0, 7)
        np.testing.assert_array_equal(np.asarray(a["indices"]), np.asarray(b["indices"]))
        idx = np.asarray(a["indices"]).reshape(-1, K)
        assert idx.shape[0] == 5000
        assert not (idx == np.arange(5000)[:, None]).any(), "self-loop"
        assert all(len(set(row)) == K for row in idx), "duplicate neighbour"
        dist = np.asarray(a["data"]).reshape(-1, K)
        assert (np.diff(dist, axis=1) >= 0).all(), "distances not sorted"

    def test_pp_neighbors_auto_switch(self) -> None:
        adata = ps.AnnData(X=np.zeros((12_000, 1), dtype=np.float32))
        adata.obsm["X_pca"] = mixture(12_000, seed=5)
        ps.pp.neighbors(adata, n_neighbors=K)
        assert adata.uns["neighbors"]["method"] == "nndescent"
        assert adata.obsp["distances"].shape == (12_000, 12_000)
        # n_neighbors counts the cell itself (scanpy's convention): K - 1
        # stored neighbours per cell, as on the exact path.
        assert adata.obsp["distances"].nnz == 12_000 * (K - 1)
        small = ps.AnnData(X=np.zeros((2_000, 1), dtype=np.float32))
        small.obsm["X_pca"] = mixture(2_000, seed=5)
        ps.pp.neighbors(small, n_neighbors=K)
        assert small.uns["neighbors"]["method"] == "exact"
        assert small.obsp["distances"].nnz == 2_000 * (K - 1)


class TestExact:
    def test_exact_matches_numpy(self) -> None:
        emb = mixture(800, d=8, seed=1)
        g = neighbors_dense(emb, 6, "euclidean")
        idx = np.asarray(g["indices"]).reshape(800, 5)
        d2 = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        ref = np.argsort(d2, axis=1)[:, :5]
        # Sets agree (ties may reorder within a row).
        agree = np.mean([set(idx[i]) == set(ref[i]) for i in range(800)])
        assert agree > 0.99


def test_cosine_is_honoured_on_the_approximate_path() -> None:
    """Above 10,000 cells the search is approximate; metric='cosine' must
    still give cosine neighbours and 1 - cos distances, as the exact path
    does, rather than silently searching in Euclidean distance."""
    import numpy as np
    import polariseq as ps

    rng = np.random.default_rng(0)
    # Directions carry the structure, lengths are noise: cosine and
    # Euclidean neighbours disagree on data like this.
    base = rng.normal(size=(12_000, 10)).astype(np.float32)
    x = base * rng.uniform(0.2, 5.0, size=(12_000, 1)).astype(np.float32)
    exact, approx = ps.AnnData(X=x.copy()), ps.AnnData(X=x.copy())
    for a in (exact, approx):
        a.obsm["X_pca"] = x
    ps.pp.neighbors(exact, n_neighbors=10, metric="cosine", approximate=False)
    ps.pp.neighbors(approx, n_neighbors=10, metric="cosine", approximate=True)
    e, a = exact.obsp["distances"], approx.obsp["distances"]
    overlap = np.mean([len(set(e.indices[e.indptr[i]:e.indptr[i + 1]])
                           & set(a.indices[a.indptr[i]:a.indptr[i + 1]])) / 9
                       for i in range(0, 12_000, 97)])
    assert overlap > 0.9, overlap
    assert float(np.max(a.data)) <= 2.0 + 1e-5      # cosine distances, not Euclidean
