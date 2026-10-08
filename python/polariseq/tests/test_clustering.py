"""Leiden clustering quality: recovery and igraph agreement.

The handbook's acceptance: on a 20k Gaussian-mixture kNN graph the labels
are recovered (ARI ≥ 0.9), and against igraph's C Leiden on the SAME
weighted (fuzzy connectivity) graph the modularity agrees within 0.5 % and
ARI ≥ 0.9.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import polariseq as ps
import pytest

K = 15


def mixture(n: int, d: int = 30, n_clusters: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d)) * 5.0
    labels = rng.integers(0, n_clusters, size=n)
    X = (centers[labels] + rng.normal(size=(n, d))).astype(np.float32)
    return X, labels


def ari(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ct = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(ct, (a, b), 1)

    def c2(x):
        return (x * (x - 1) // 2).sum()

    n = len(a)
    total = n * (n - 1) // 2
    si, sj = c2(ct.sum(1)), c2(ct.sum(0))
    expected = si * sj / total
    actual = c2(ct)
    max_idx = si + sj
    return float((actual - expected) / (0.5 * max_idx - expected))


def igraph_partition(adata):
    ig = pytest.importorskip("igraph")
    conn = adata.obsp["connectivities"]
    n = adata.n_obs
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(conn.indptr))
    cols = conn.indices.astype(np.int64)
    mask = cols > rows
    g = ig.Graph(
        n=n,
        edges=list(zip(rows[mask].tolist(), cols[mask].tolist(), strict=True)),
        directed=False,
    )
    g.es["weight"] = conn.data[mask].astype(float).tolist()
    part = g.community_leiden(
        weights="weight", resolution=1.0, n_iterations=2,
        objective_function="modularity",
    )
    return np.asarray(part.membership, dtype=np.int64), g.modularity(
        part.membership, weights="weight"
    )


class TestLeiden:
    @pytest.fixture(scope="class")
    def clustered(self):
        X, labels = mixture(20_000)
        adata = ps.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
        adata.obsm["X_pca"] = X
        ps.pp.neighbors(adata, n_neighbors=K, seed=0)
        ps.tl.leiden(adata, resolution=1.0, seed=0)
        return adata, labels

    def test_recover_mixture_labels(self, clustered) -> None:
        adata, labels = clustered
        score = ari(adata.obs["leiden"], labels)
        assert score >= 0.9, f"ARI vs mixture labels {score:.3f}"

    def test_connectivities_stored_by_neighbors(self, clustered) -> None:
        adata, _labels = clustered
        conn = adata.obsp["connectivities"]
        assert conn.shape == (adata.n_obs, adata.n_obs)
        # symmetric storage, both directions
        assert (conn != conn.T).nnz == 0

    def test_matches_igraph(self, clustered) -> None:
        adata, _labels = clustered
        part, q_ig = igraph_partition(adata)
        q_ours = adata.uns["leiden"]["quality"]
        assert abs(q_ours - q_ig) / abs(q_ig) < 0.005, (
            f"modularity ours {q_ours:.4f} vs igraph {q_ig:.4f}"
        )
        score = ari(adata.obs["leiden"], part)
        assert score >= 0.9, f"ARI vs igraph {score:.3f}"

    def test_deterministic(self) -> None:
        X, _labels = mixture(3_000, seed=3)
        outs = []
        for _ in range(2):
            adata = ps.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
            adata.obsm["X_pca"] = X
            ps.pp.neighbors(adata, n_neighbors=K, seed=0)
            ps.tl.leiden(adata, resolution=1.0, seed=5)
            outs.append(adata.obs["leiden"].copy())
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_fast_agrees_on_big_structure(self) -> None:
        X, labels = mixture(5_000, seed=4)
        adata = ps.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
        adata.obsm["X_pca"] = X
        ps.pp.neighbors(adata, n_neighbors=K, seed=0)
        ps.tl.leiden(adata, resolution=1.0, seed=0, method="leiden")
        leiden_labels = adata.obs["leiden"].copy()
        q_leiden = adata.uns["leiden"]["quality"]
        ps.tl.leiden(adata, resolution=1.0, seed=0, method="fast")
        q_fast = adata.uns["leiden"]["quality"]
        # Louvain stays within 2% of Leiden's modularity on the same graph.
        assert (q_fast - q_leiden) / q_leiden > -0.02, (
            f"fast {q_fast:.4f} vs leiden {q_leiden:.4f}"
        )
        assert ari(adata.obs["leiden"], leiden_labels) >= 0.8

BENCH = pathlib.Path(__file__).resolve().parents[3] / "data" / "bench"


class TestTermination:
    """Leiden must finish, on the graph it actually clusters.

    `tl.leiden` clusters UMAP fuzzy connectivities, whose weights live in
    [0, 1]. Aggregated levels of such a graph carry large self-loops, so the
    move score `w_to − res·k_i·K_c` is a difference of large numbers and its
    f64 rounding error is `≈ 2.2e-16 · two_m`. An *absolute* move threshold
    below that noise floor lets two communities trade a vertex forever.

    Measured before the fix, on heart_50k: a 50-node aggregate level made
    1,000,000+ moves with a 17.2-million-entry queue, pinned one core, and
    never returned. Nothing in the Rust unit tests caught it — a hand-built
    graph does not reproduce the self-loop structure or the drift that
    accumulates over a million incremental updates (see the note in
    `leiden.rs::tests`). Only the real graph does, so this test uses it.

    The bound is deliberately loose. Measured on an 8 GB M2: 0.10 s at 10k,
    0.38 s at 50k, 0.86 s at 100k, versus igraph's 1.0 / 6.4 / 15.5 s. A
    60-second ceiling cannot fail on timing noise; it fails only if the
    termination bug is back.
    """

    @pytest.mark.parametrize("name,expect_cells", [("heart_10k", 9_863), ("heart_50k", 49_313)])
    def test_leiden_terminates_on_a_real_graph(self, name: str, expect_cells: int) -> None:
        path = BENCH / f"{name}.h5ad"
        if not path.exists():
            pytest.skip(f"{path} not present (real benchmark data is gitignored)")

        adata = ps.pp._ensure_materialized(ps.read_h5ad(str(path), lazy=True))
        ps.pp.calculate_qc_metrics(adata)
        ps.pp.filter_cells(adata, min_genes=200)
        ps.pp.normalize_total(adata, target_sum=1e4)
        ps.pp.log1p(adata)
        ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        ps.pp.pca(adata, 50, seed=0)
        ps.pp.neighbors(adata, n_neighbors=15, seed=0)
        assert adata.n_obs == expect_cells

        t0 = time.perf_counter()
        ps.tl.leiden(adata, resolution=1.0, seed=0)
        elapsed = time.perf_counter() - t0

        assert elapsed < 60.0, (
            f"leiden took {elapsed:.1f}s on {name}; it should be well under a "
            f"second. This is the non-termination regression, not slowness."
        )
        labels = np.asarray(adata.obs["leiden"])
        n_clusters = len(set(labels.tolist()))
        # Sanity: a real tissue at resolution 1.0 gives tens of clusters, not
        # one blob and not one cluster per cell. A capped local_move would
        # still land in this range, so this checks the answer is usable, not
        # that the cap stayed idle.
        assert 5 <= n_clusters <= 100, f"{name}: {n_clusters} clusters"



def test_default_leiden_clusters_the_connectivity_graph_as_given() -> None:
    """Polariseq's Leiden and igraph's, on one connectivity graph, agree as
    closely as igraph agrees with itself across seeds. From 2026-09-06 to
    2026-09-28 the default path re-weighted the connectivities as if they
    were distances (building the fuzzy set a second time) and so clustered a
    different graph from igraph, UMAP and Scanpy: on this data it scored
    0.81 against igraph where igraph scores 0.90 against itself. Twelve
    overlapping groups, because well-separated ones come out perfect under
    either graph and test nothing."""
    pytest.importorskip("igraph")
    from sklearn.metrics import adjusted_rand_score as ari

    rng = np.random.default_rng(1)
    centers = rng.normal(size=(12, 20)) * 0.9
    groups = rng.integers(0, 12, 5000)
    a = ps.AnnData(X=np.zeros((5000, 1), dtype=np.float32))
    a.obsm["X_pca"] = (centers[groups] + rng.normal(size=(5000, 20))).astype(np.float32)
    ps.pp.neighbors(a, n_neighbors=15, seed=0)
    ours, ref = [], []
    for s in range(4):
        ps.tl.leiden(a, seed=s, key_added="o")
        ps.tl.leiden(a, seed=s, key_added="g", method="igraph")
        ours.append(np.asarray(a.obs["o"]))
        ref.append(np.asarray(a.obs["g"]))
    floor = np.mean([ari(ref[i], ref[j]) for i in range(4) for j in range(i + 1, 4)])
    between = np.mean([ari(o, r) for o in ours for r in ref])
    assert between > floor - 0.03, f"ours vs igraph {between:.3f}, igraph vs itself {floor:.3f}"


def test_igraph_leiden_follows_its_seed() -> None:
    """The igraph backend draws from the seed it is given, not from the state
    Python's `random` module was left in by whatever ran before."""
    pytest.importorskip("igraph")
    import random

    rng = np.random.default_rng(2)
    a = ps.AnnData(X=np.zeros((1500, 1), dtype=np.float32))
    a.obsm["X_pca"] = (rng.normal(size=(6, 10))[rng.integers(0, 6, 1500)]
                       + rng.normal(size=(1500, 10))).astype(np.float32)
    ps.pp.neighbors(a, n_neighbors=15, seed=0)
    runs = []
    for state in (1, 2):
        random.seed(state)
        ps.tl.leiden(a, seed=7, method="igraph", key_added="g")
        runs.append(np.asarray(a.obs["g"]).copy())
    np.testing.assert_array_equal(runs[0], runs[1])
