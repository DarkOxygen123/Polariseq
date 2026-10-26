"""What makes Polariseq's UMAP of 4 million cells break into islands? One variable at a time.

    python benchmarks/r3_umap_cause_4m.py FETAL_H5AD FETAL_CELLS_CSV OUT [--threads 8]

Runs the benchmark's Polariseq pipeline on the human fetal atlas (quality control, cells with at
least 200 genes, normalization to 10,000, log, 2,000 'seurat' genes, PCA with 50 components, 15
neighbors, Leiden at resolution 1, seed 0), then lays out that one graph many ways and measures each
layout against the authors' cell types (islands; share of each type in its largest island; the
cell-type purity of the layout's pixels; correlation of type-to-type distances with the principal-
component space). Variables:

  start     the default spectral start (each axis in [-10, 10]); the same start rescaled to [0, 10]
            (umap-learn's convention); that start refined by further subspace iterations; the first
            two principal components; a random start; Scarf's start (k-means centroids of the
            principal components, laid out by their own PCA, each cell at its centroid)
  min_dist  0.1 (Polariseq's default, used in the runs) and 0.5 (Scanpy's default)
  code      umap-learn on Polariseq's graph; Polariseq's UMAP on umap-learn's graph of the same
            principal components

and, for the graph: the recall of the 14 nearest neighbors against exact neighbors (2,000 cells), and
how far the default start is from converged eigenvectors (relative residual of S v = lambda v).
Writes OUT/summary.json (filled in as it goes), the layouts and the graph.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def centroid_spearman(pcs, emb, groups) -> float:
    from scipy.stats import spearmanr
    labs = np.unique(groups[groups >= 0])
    c_hi = np.array([pcs[groups == g].mean(0) for g in labs])
    c_lo = np.array([emb[groups == g].mean(0) for g in labs])
    iu = np.triu_indices(len(labs), 1)
    d_hi = np.linalg.norm(c_hi[:, None] - c_hi[None], axis=-1)[iu]
    d_lo = np.linalg.norm(c_lo[:, None] - c_lo[None], axis=-1)[iu]
    return float(spearmanr(d_hi, d_lo).statistic)


def normalized_adjacency(conn):
    import scipy.sparse as sp
    deg = np.asarray(conn.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return sp.diags(dinv) @ conn @ sp.diags(dinv), np.sqrt(deg)


def residuals(S, triv, v2: np.ndarray) -> list[float]:
    """Relative residual ||S v - (v'Sv) v|| / ||v|| of each start axis, after removing the trivial
    eigenvector (the start's axes are rescaled eigenvector estimates)."""
    out = []
    t = triv / np.linalg.norm(triv)
    for j in range(v2.shape[1]):
        v = v2[:, j].astype(np.float64)
        v = v - v.mean()
        v = v - (v @ t) * t
        v /= np.linalg.norm(v)
        sv = S @ v
        lam = v @ sv
        out.append(float(np.linalg.norm(sv - lam * v)))
    return out


def refine(S, triv, v2: np.ndarray, iters: int) -> np.ndarray:
    """More subspace iterations from the given start (3 vectors: the trivial one and two more)."""
    t = triv / np.linalg.norm(triv)
    x = np.column_stack([t, v2[:, 0] - v2[:, 0].mean(), v2[:, 1] - v2[:, 1].mean()]).astype(np.float64)
    for _ in range(iters):
        x, _ = np.linalg.qr(S @ x)
    # Rayleigh-Ritz in the 3-D span; drop the trivial direction
    b = x.T @ (S @ x)
    w, u = np.linalg.eigh(b)
    y = x @ u[:, ::-1]
    return y[:, 1:3]


def span_angle(a: np.ndarray, b: np.ndarray, triv: np.ndarray) -> float:
    """Largest principal angle (degrees) between two 2-D layouts' spans, each axis's shift undone
    along the trivial eigenvector (benchmarks/r4_spectral.py)."""
    import r4_spectral as r4s
    return r4s.angle(a, r4s.span(b, triv), triv)


def scarf_start(pcs: np.ndarray, n_centroids: int = 1000, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    km = MiniBatchKMeans(n_clusters=n_centroids, random_state=seed, batch_size=20_000, n_init=1).fit(pcs)
    pc = PCA(n_components=2, random_state=seed).fit_transform(km.cluster_centers_)
    return pc[km.labels_]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("cells_csv")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-umap-learn", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    import polariseq as ps
    import scipy.sparse as sp
    from polariseq._polariseq import umap_spectral_layout
    from r3_layout_purity import purity
    from r3_umap_fragmentation import islands, recall

    out = Path(a.out)
    (out / "layouts").mkdir(parents=True, exist_ok=True)
    summary = {"layouts": {}}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    ps.set_budget(ram_gb=150, threads=a.threads)
    log("pipeline")
    adata = ps.read_h5ad(a.h5ad, lazy=True)
    adata = ps.pp._ensure_materialized(adata)
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=200)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=True)
    ps.pp.pca(adata, 50, seed=0)
    ps.pp.neighbors(adata, n_neighbors=15, seed=0)
    ps.tl.leiden(adata, resolution=1.0, seed=0)
    pcs = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    names = np.asarray(adata.obs_names).astype(str)
    cells = pd.read_csv(a.cells_csv, usecols=["cell", "Main_cluster_name"],
                        dtype={"Main_cluster_name": "category"}).set_index("cell")
    groups = cells.Main_cluster_name.reindex(names).cat.codes.to_numpy().astype(np.int64)
    log("cells", len(names), "with a published type", int((groups >= 0).sum()))
    assert (groups >= 0).all(), "every kept cell should have a published type"
    conn = adata.obsp["connectivities"].tocsr()
    conn.sort_indices()
    dist = adata.obsp["distances"].tocsr()
    np.save(out / "pcs.npy", pcs)
    np.save(out / "leiden.npy", np.asarray(adata.obs["leiden"]).astype(np.int32))
    np.save(out / "groups.npy", groups)
    sp.save_npz(out / "connectivities.npz", conn)
    summary["cells"] = int(len(names))
    summary["clusters"] = int(len(np.unique(np.asarray(adata.obs["leiden"]))))

    k = int(np.diff(dist.indptr).max())
    summary["graph"] = {"neighbors_listed": k,
                        "recall_polariseq": round(recall(pcs, dist.indptr, dist.indices, k, metric="euclidean"), 4),
                        "components": int(sp.csgraph.connected_components(conn, directed=False)[0])}
    log("graph", summary["graph"])
    save()

    def measure(name, emb, seconds):
        emb = np.asarray(emb, dtype=np.float32)
        m = islands(emb, groups)
        m.update({"pixel_type_purity": round(purity(emb, groups), 4),
                  "centroid_spearman": round(centroid_spearman(pcs, emb, groups), 4),
                  "seconds": round(seconds, 1)})
        summary["layouts"][name] = m
        np.save(out / "layouts" / (name.replace(" ", "_").replace(",", "") + ".npy"), emb)
        log(name, m)
        save()

    def run(name, init, min_dist=0.1, graph=None):
        b = adata if graph is None else graph
        t0 = time.perf_counter()
        ps.tl.umap(b, seed=0, init=init, min_dist=min_dist)
        measure(name, b.obsm["X_umap"], time.perf_counter() - t0)

    # the default start, by itself
    t0 = time.perf_counter()
    start = umap_spectral_layout(conn.indptr.astype(np.uint32), conn.indices.astype(np.uint32),
                                 conn.data.astype(np.float32), conn.shape[0], 0)
    t_start = time.perf_counter() - t0
    S, triv = normalized_adjacency(conn.astype(np.float64))
    t0 = time.perf_counter()
    refined = refine(S, triv, start, 300)
    summary["spectral_start"] = {"seconds": round(t_start, 1),
                                 "relative_residual_default": [round(x, 5) for x in residuals(S, triv, start)],
                                 "relative_residual_after_300_more_iterations":
                                     [round(x, 5) for x in residuals(S, triv, refined)],
                                 "refine_seconds": round(time.perf_counter() - t0, 1),
                                 "angle_default_vs_refined_degrees": round(span_angle(start, refined, triv), 2)}
    log("spectral start", summary["spectral_start"])
    save()

    run("spectral, default scale, min_dist 0.1 (as in the runs)", "spectral")
    run("spectral rescaled to [0,10], min_dist 0.1", start)
    run("spectral refined, [0,10], min_dist 0.1", refined)
    run("pca, [0,10], min_dist 0.1", "pca")
    run("random, min_dist 0.1", "random")
    t0 = time.perf_counter()
    scarf = scarf_start(pcs)
    log("scarf start", round(time.perf_counter() - t0, 1), "s")
    run("scarf start (k-means centroids), [0,10], min_dist 0.1", scarf)
    run("spectral, default scale, min_dist 0.5", "spectral", 0.5)
    run("spectral rescaled to [0,10], min_dist 0.5", start, 0.5)
    run("pca, [0,10], min_dist 0.5", "pca", 0.5)
    run("scarf start (k-means centroids), [0,10], min_dist 0.5", scarf, 0.5)

    # the other implementation and the other graph
    import anndata as ad
    import scanpy as sc
    sc.settings.n_jobs = a.threads
    b = ad.AnnData(obs=pd.DataFrame(index=names))
    b.obsm["X_pca"] = pcs
    t0 = time.perf_counter()
    sc.pp.neighbors(b, n_neighbors=15, use_rep="X_pca", random_state=0)
    t_nb = time.perf_counter() - t0
    d2 = b.obsp["distances"].tocsr()
    summary["graph"]["recall_umap_learn"] = round(recall(pcs, d2.indptr, d2.indices, k, metric="euclidean"), 4)
    summary["graph"]["umap_learn_neighbors_seconds"] = round(t_nb, 1)
    log("umap-learn graph", summary["graph"])
    g = ps.AnnData(X=sp.csr_matrix((len(names), 1), dtype=np.float32))
    g.obsp["connectivities"] = b.obsp["connectivities"].tocsr()
    g.obsm["X_pca"] = pcs
    run("umap-learn graph, Polariseq UMAP, spectral default, min_dist 0.1", "spectral", graph=g)
    run("umap-learn graph, Polariseq UMAP, spectral rescaled, min_dist 0.5",
        umap_spectral_layout(g.obsp["connectivities"].indptr.astype(np.uint32),
                             g.obsp["connectivities"].indices.astype(np.uint32),
                             g.obsp["connectivities"].data.astype(np.float32), len(names), 0), 0.5, graph=g)
    if not a.skip_umap_learn:
        c = ad.AnnData(obs=pd.DataFrame(index=names))
        c.obsm["X_pca"] = pcs
        c.obsp["distances"] = dist
        c.obsp["connectivities"] = conn
        c.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                              "params": {"n_neighbors": 15, "method": "umap", "metric": "euclidean"}}
        t0 = time.perf_counter()
        sc.tl.umap(c, min_dist=0.5, random_state=0)
        measure("Polariseq graph, umap-learn UMAP (Scanpy's call), min_dist 0.5", c.obsm["X_umap"],
                time.perf_counter() - t0)
    log("done")


if __name__ == "__main__":
    main()
