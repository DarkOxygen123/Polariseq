"""Why Polariseq's UMAP of the fetal atlas breaks into many small islands: the graph or the layout?

    .venv/bin/python benchmarks/r3_umap_fragmentation.py

On the authors' balanced sample of the fetal atlas (377,456 cells; 50 principal components, 50
neighbors, cosine, as in the authors' procedure), two neighbor graphs of the same principal
components are built: Polariseq's (ps.pp.neighbors) and umap-learn's (scanpy's sc.pp.neighbors,
pynndescent). Each is laid out by both UMAP implementations, so that a difference that follows the
graph and not the layout code points to the graph. Measured:

  graph   recall of the 50 nearest neighbors against exact neighbors (2,000 random cells, brute force);
          connected components
  layout  islands (connected groups of occupied pixels on a 500 x 500 density grid,
          holding at least 0.05% of the cells); for each published type, the share of its cells in its
          largest island (1.0: the type lies in one piece)

The layouts on Polariseq's graph are those of r3_analyses.py umap (benchmarks/results/r3/umap/
fetal_matched, r3_analyses.py umap); the two on umap-learn's graph are computed here. Writes
benchmarks/results/r3/umap_fragmentation/ (summary.json and the layouts).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
OUT = ROOT / "benchmarks/results/r3/umap_fragmentation"
PREV = ROOT / "benchmarks/results/r3/umap/fetal_matched"


def recall(pcs: np.ndarray, indptr, indices, k: int, n_query: int = 2000, seed: int = 0,
           metric: str = "cosine") -> float:
    """Share of each query cell's exact k nearest neighbors (cosine or euclidean) that the graph lists."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(pcs), n_query, replace=False)
    if metric == "cosine":
        x = pcs / np.linalg.norm(pcs, axis=1, keepdims=True)
    else:
        x = np.asarray(pcs, dtype=np.float32)
        sq = (x.astype(np.float64) ** 2).sum(1)
    hits = []
    for i in q:
        # similarity: larger is nearer (for euclidean, minus the squared distance up to a constant)
        sim = x @ x[i] if metric == "cosine" else 2.0 * (x @ x[i]) - sq
        sim[i] = -np.inf
        exact = set(np.argpartition(-sim, k)[:k].tolist())
        listed = set(indices[indptr[i]:indptr[i + 1]].tolist()) - {i}
        hits.append(len(exact & listed) / k)
    return float(np.mean(hits))


def components(g) -> dict:
    from scipy.sparse.csgraph import connected_components
    n, lab = connected_components(g, directed=False)
    sizes = np.sort(np.bincount(lab))[::-1]
    return {"components": int(n), "largest_share": round(float(sizes[0] / sizes.sum()), 4),
            "components_over_100_cells": int((sizes >= 100).sum())}


def islands(emb: np.ndarray, groups: np.ndarray, bins: int | None = None) -> dict:
    """Islands of a layout: connected groups of occupied pixels holding at least 0.05% of the cells.
    The grid has about sqrt(n)/4 pixels on its longer side (100 to 600), so that a layout's density,
    not the number of cells, decides what is empty, and one-pixel gaps are closed first, so that
    sparse fringes do not count as separate islands."""
    from scipy import ndimage

    from polariseq.plotting import _raster_grid
    if bins is None:
        bins = int(np.clip(np.sqrt(len(emb)) / 4, 100, 600))
    pix, (nx, ny), _ = _raster_grid(np.asarray(emb, dtype=float), bins)
    occ = np.bincount(pix, minlength=nx * ny).reshape(ny, nx) > 0
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)))
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    cell_island = lab.ravel()[pix]
    counts = np.bincount(cell_island, minlength=n + 1)[1:]
    big = int((counts >= 0.0005 * len(emb)).sum())
    whole = []
    for g in np.unique(groups):
        isl = cell_island[groups == g]
        whole.append(np.bincount(isl).max() / len(isl))
    whole = np.array(whole)
    return {"islands": big, "type_share_in_largest_island_median": round(float(np.median(whole)), 3),
            "types_mostly_in_one_island": int((whole >= 0.8).sum()), "types": int(len(whole)), "grid": bins}


def main() -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    import fetal_matched as fm
    import polariseq as ps

    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_budget(ram_gb=6, threads=4)
    x, meta, *_ = fm.matched_sample()
    groups = meta["Main_cluster_name"].to_numpy().astype(str)
    assert np.array_equal(groups, np.load(PREV / "groups.npy", allow_pickle=True).astype(str))
    a = ps.AnnData(X=x)
    ps.pp.pca(a, fm.N_PCS, seed=fm.SEED)
    ps.pp.neighbors(a, n_neighbors=fm.N_NEIGHBORS, metric="cosine", seed=fm.SEED)
    ps.tl.leiden(a, seed=0)  # builds obsp['connectivities']
    pcs = np.asarray(a.obsm["X_pca"], dtype=np.float32)
    k = fm.N_NEIGHBORS - 1   # each row lists k others besides the cell (scanpy's convention)
    res = {"cells": int(len(pcs)), "graphs": {}, "layouts": {}}
    d_ps = a.obsp["distances"].tocsr()
    res["graphs"]["polariseq"] = {"recall": round(recall(pcs, d_ps.indptr, d_ps.indices, k), 4),
                                  **components(a.obsp["connectivities"])}
    b = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(pcs))]))
    b.obsm["X_pca"] = pcs
    t0 = time.perf_counter()
    sc.pp.neighbors(b, n_neighbors=fm.N_NEIGHBORS, use_rep="X_pca", metric="cosine", random_state=0)
    res["graphs"]["umap-learn"] = {"seconds": round(time.perf_counter() - t0, 1)}
    d_un = b.obsp["distances"].tocsr()
    res["graphs"]["umap-learn"].update({"recall": round(recall(pcs, d_un.indptr, d_un.indices, k), 4),
                                        **components(b.obsp["connectivities"])})
    print(json.dumps(res["graphs"], indent=1), flush=True)
    # layouts: the two on Polariseq's graph from r3_analyses.py umap, the two on umap-learn's graph here
    lay = {"polariseq graph, Polariseq UMAP": np.load(PREV / "polariseq_seed0.npy"),
           "polariseq graph, umap-learn UMAP": np.load(PREV / "umap_learn_seed0.npy")}
    sc.tl.umap(b, min_dist=fm.MIN_DIST, random_state=0)
    lay["umap-learn graph, umap-learn UMAP"] = np.asarray(b.obsm["X_umap"])
    c = ps.AnnData(X=x)
    c.obsp["connectivities"] = b.obsp["connectivities"].tocsr()
    ps.tl.umap(c, min_dist=fm.MIN_DIST, seed=0)
    lay["umap-learn graph, Polariseq UMAP"] = np.asarray(c.obsm["X_umap"])
    for name, emb in lay.items():
        res["layouts"][name] = islands(emb, groups)
        np.save(OUT / (name.replace(", ", "__").replace(" ", "_") + ".npy"), np.asarray(emb, dtype=np.float32))
        print(name, res["layouts"][name], flush=True)
    np.save(OUT / "groups.npy", groups)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
