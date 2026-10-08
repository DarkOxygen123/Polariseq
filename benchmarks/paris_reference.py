"""Reference outputs for the Paris port, from scikit-network and Scarf.

Run in the Scarf environment (scikit-network 0.33.2, Scarf 0.32.3, NumPy
1.26.4), which is where Scarf's clustering runs:

    .venv-scarf/bin/python benchmarks/paris_reference.py

Writes ``tests/data/paris_reference.npz``: a set of graphs chosen to stress
the parts of the algorithm where a port can go wrong (ties in unweighted
graphs, disconnected components, isolated nodes, directed input, a UMAP
fuzzy kNN graph like the ones Polariseq clusters), and for each graph the
reference dendrogram (``reorder=False``, Scarf's setting, and ``True``,
scikit-network's default), straight cuts, and Scarf's balanced cut on the
dendrogram as Scarf passes it (infinite heights set to 0).
``python/polariseq/tests/test_paris.py`` requires the port to reproduce the
dendrograms bit for bit and the cuts exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/data/paris_reference.npz"
KS = (2, 3, 5, 10, 20, 50)  # k=1 raises IndexError in scikit-network 0.33.2
BALANCED = ((20, 5, 2.0), (50, 10, 2.0), (100, 20, 1.5))


def graphs() -> dict[str, sp.csr_matrix]:
    from sknetwork.data import house, karate_club

    rng = np.random.default_rng(20260928)
    g: dict[str, sp.csr_matrix] = {
        "house": sp.csr_matrix(house(), dtype=np.float64),
        "karate": sp.csr_matrix(karate_club(), dtype=np.float64),
    }
    # Grid: unweighted, every similarity ties with several others.
    side = 20
    idx = np.arange(side * side).reshape(side, side)
    e = np.concatenate([np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()],
                        np.c_[idx[:-1, :].ravel(), idx[1:, :].ravel()]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(side**2, side**2))
    g["grid"] = (a + a.T).tocsr()
    # Sparse weighted graphs: several components, isolated nodes.
    for s in range(3):
        n = 300
        a = sp.random(n, n, density=0.012, random_state=rng, data_rvs=lambda k: rng.uniform(0.05, 1.0, k))
        a = sp.triu(a, 1)
        g[f"er{s}"] = (a + a.T).tocsr()
    # Directed weighted graphs (Scarf clusters its directed kNN graph).
    for s in range(2):
        n, k = 200, 8
        rows = np.repeat(np.arange(n), k)
        cols = np.concatenate([rng.choice(np.delete(np.arange(n), i), k, replace=False) for i in range(n)])
        g[f"directed{s}"] = sp.csr_matrix((rng.uniform(0.05, 1.0, n * k), (rows, cols)), shape=(n, n))
    # A UMAP fuzzy kNN graph of overlapping Gaussian groups (float32
    # weights, as Polariseq's connectivities).
    from sklearn.neighbors import NearestNeighbors
    from umap.umap_ import fuzzy_simplicial_set

    centres = rng.normal(0, 0.9, (12, 10))
    x = (centres[rng.integers(0, 12, 2000)] + rng.normal(0, 1.0, (2000, 10))).astype(np.float32)
    kd, ki = NearestNeighbors(n_neighbors=15).fit(x).kneighbors(x)
    kd[:, 0] = 0.0
    conn, _, _ = fuzzy_simplicial_set(x, 15, 0, "euclidean", knn_indices=ki, knn_dists=kd)
    conn = sp.csr_matrix(conn, dtype=np.float32)
    conn.sort_indices()
    g["knn"] = sp.csr_matrix(conn.astype(np.float64))
    for a in g.values():
        a.sum_duplicates()
        a.sort_indices()
    return g


def main() -> int:
    import numpy
    import scarf
    import sknetwork
    from scarf.dendrogram import BalancedCut
    from sknetwork.hierarchy import Paris, cut_straight

    out: dict[str, np.ndarray] = {}
    names = []
    for name, a in graphs().items():
        n = a.shape[0]
        names.append(name)
        out[f"{name}_indptr"] = a.indptr.astype(np.int64)
        out[f"{name}_indices"] = a.indices.astype(np.int64)
        out[f"{name}_data"] = a.data.astype(np.float64)
        out[f"{name}_n"] = np.array(n)
        d = Paris(reorder=False).fit_transform(a)
        out[f"{name}_paris"] = d
        out[f"{name}_paris_reordered"] = Paris(reorder=True).fit_transform(a)
        ks = [k for k in KS if k <= n]
        out[f"{name}_ks"] = np.array(ks)
        for k in ks:
            out[f"{name}_straight_{k}"] = cut_straight(d, n_clusters=k)
        scarf_d = d.copy()
        scarf_d[scarf_d == np.inf] = 0  # as Scarf's run_clustering does
        done = []
        for i, (mx, mn, fc) in enumerate(BALANCED):
            if mx >= n:
                continue
            out[f"{name}_balanced_{i}"] = BalancedCut(scarf_d, mx, mn, fc).get_clusters()
            done.append(i)
        out[f"{name}_balanced"] = np.array(done)
        print(f"{name:10s} n={n:5d} nnz={a.nnz:6d} inf heights={int(np.isinf(d[:, 2]).sum())}")
    out["graphs"] = np.array(names)
    out["balanced_params"] = np.array(BALANCED)
    out["versions"] = np.array([f"scikit-network {sknetwork.__version__}", f"scarf {scarf.__version__}",
                                f"numpy {numpy.__version__}"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
