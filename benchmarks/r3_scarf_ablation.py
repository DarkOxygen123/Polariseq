"""Why do Scarf's clusters match the fetal atlas's published cell types better? One step at a time.

    python benchmarks/r3_scarf_ablation.py FETAL_H5AD FETAL_CELLS_CSV SCARF_HVG_TXT OUT [--threads 8]

On the human fetal atlas (cells with at least 200 genes), Polariseq's pipeline is run with each
combination of a gene set and a normalization, and its Leiden clusters are scored against the
authors' 77 cell types (adjusted Rand index, normalized mutual information, purity, types recovered),
at resolution 1 and over a range of resolutions (so that a difference in the number of clusters is
not mistaken for a difference in quality).

Gene sets (2,000 each):
  seurat          Polariseq's (Scanpy's 'seurat' flavor, as in the benchmark)
  seurat>=1%      the same ranking among genes detected in at least 1% of cells (Scarf's default floor)
  scarf           the genes Scarf selected in its run on the cluster node
  scarf-port      Scarf's method reimplemented here (library-size normalization to 1,000 without log;
                  genes binned by log mean into 200 bins; a LOWESS curve through each bin's least
                  variable gene; variance divided by that curve; genes detected in more than 1% of
                  cells; the top 2,000), to check a port against Scarf's own selection

Normalizations:
  log1e4          Polariseq's: counts to 10,000 per cell over all genes, log, no scaling (benchmark)
  scarf           Scarf's make_graph defaults: counts to 1,000 per cell over the selected genes only,
                  log, each gene scaled to unit variance before PCA

Writes OUT/summary.json (filled in as it goes) and OUT/genes_<set>.txt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESOLUTIONS = (0.2, 0.3, 0.5, 0.7, 1.0)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def scores(clusters: np.ndarray, types: np.ndarray) -> dict:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    t = pd.crosstab(clusters, types)
    share_of_type = t / t.sum(axis=0)
    share_of_cluster = t.div(t.sum(axis=1), axis=0)
    recovered = int(sum(1 for c in t.columns
                        if share_of_type[c].max() >= 0.5
                        and share_of_cluster.loc[share_of_type[c].idxmax(), c] >= 0.5))
    return {"clusters": int(t.shape[0]), "ari": round(adjusted_rand_score(types, clusters), 4),
            "nmi": round(normalized_mutual_info_score(types, clusters), 4),
            "purity": round(float(t.max(axis=1).sum() / len(clusters)), 4), "types_recovered": recovered}


def gene_stats(h5ad: str, n_genes: int, min_genes: int = 200, block: int = 100_000) -> dict:
    """Scarf's per-gene statistics, streamed from the file: on counts scaled to 1,000 per cell (no
    log), over the cells with at least `min_genes` genes."""
    import h5py
    with h5py.File(h5ad, "r") as f:
        indptr = f["X/indptr"][:]
        n_total = len(indptr) - 1
        ncells = np.zeros(n_genes)
        s1 = np.zeros(n_genes)
        s2 = np.zeros(n_genes)
        kept = 0
        for r0 in range(0, n_total, block):
            r1 = min(r0 + block, n_total)
            a, b = indptr[r0], indptr[r1]
            data = f["X/data"][a:b].astype(np.float64)
            idx = f["X/indices"][a:b]
            lens = np.diff(indptr[r0:r1 + 1])
            rows = np.repeat(np.arange(r1 - r0), lens)
            lib = np.bincount(rows, weights=data, minlength=r1 - r0)
            keep_row = lens >= min_genes
            m = keep_row[rows]
            normed = data[m] / lib[rows[m]] * 1000.0
            ncells += np.bincount(idx[m], minlength=n_genes)
            s1 += np.bincount(idx[m], weights=normed, minlength=n_genes)
            s2 += np.bincount(idx[m], weights=normed * normed, minlength=n_genes)
            kept += int(keep_row.sum())
    mean = s1 / kept
    return {"n_cells_total": n_total, "n_cells_kept": kept, "detected": ncells,
            "avg": s1 / n_total,                      # Scarf divides by all cells in its table
            "var": s2 / kept - mean * mean, "nz_mean": np.divide(s1, ncells, out=np.zeros(n_genes), where=ncells > 0)}


def scarf_port_genes(st: dict, top_n: int = 2000, n_bins: int = 200, frac: float = 0.1) -> np.ndarray:
    """Scarf's mark_hvgs (BSD-3-Clause, Copyright (c) Parashar Dhapola and contributors), reimplemented."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    a, b = st["avg"], st["var"]
    ok = (a > 0) & (b > 0)
    la, lb = np.log(a[ok]), np.log(b[ok])
    edges = np.histogram(la, bins=n_bins)[1]
    edges[-1] += 0.1
    which = np.digitize(la, edges) - 1
    bins = [np.flatnonzero(which == i) for i in range(n_bins)]
    bins = [x for x in bins if len(x)]
    floor = np.array([[lb[x][np.argmin(lb[x])], la[x][np.argmin(lb[x])]] for x in bins]).T
    fit = lowess(floor[0], floor[1], return_sorted=False, frac=frac, it=100)
    c = np.zeros(ok.sum())
    for f_, x in zip(fit, bins, strict=True):
        c[x] = np.exp(lb[x] - f_)
    c_var = np.zeros(len(a))
    c_var[ok] = c
    min_cells = int(0.01 * st["n_cells_total"])
    valid = st["detected"] > min_cells
    cutoff = np.sort(c_var[valid])[::-1][top_n]
    return valid & (c_var > cutoff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("cells_csv")
    ap.add_argument("scarf_hvg")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import pandas as pd

    import polariseq as ps
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"runs": {}}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    ps.set_budget(ram_gb=150, threads=a.threads)
    base = ps.read_h5ad(a.h5ad, lazy=True)
    var_names = np.asarray(base.var_names).astype(str)
    log("gene statistics (streamed)")
    st = gene_stats(a.h5ad, len(var_names))
    detected_1pct = st["detected"] > int(0.01 * st["n_cells_total"])
    summary["genes_detected_in_1pct"] = int(detected_1pct.sum())
    port = scarf_port_genes(st)
    scarf = np.isin(var_names, [g.strip() for g in open(a.scarf_hvg) if g.strip()])
    log("scarf list", int(scarf.sum()), "port", int(port.sum()))

    def prep():
        x = ps.read_h5ad(a.h5ad, lazy=True)
        x = ps.pp._ensure_materialized(x)
        ps.pp.calculate_qc_metrics(x)
        ps.pp.filter_cells(x, min_genes=200)
        return x

    # log-normalized over all genes (Polariseq's benchmark pipeline)
    x = prep()
    names = np.asarray(x.obs_names).astype(str)
    cells = pd.read_csv(a.cells_csv, usecols=["cell", "Main_cluster_name"],
                        dtype={"Main_cluster_name": "category"}).set_index("cell")
    types = cells.Main_cluster_name.reindex(names).cat.codes.to_numpy()
    assert (types >= 0).all()
    ps.pp.normalize_total(x, target_sum=1e4)
    ps.pp.log1p(x)
    ps.pp.highly_variable_genes(x, n_top_genes=2000, flavor="seurat", subset=False)
    seurat = np.asarray(x.var["highly_variable"]).astype(bool)
    sub = x[:, detected_1pct]
    ps.pp.highly_variable_genes(sub, n_top_genes=2000, flavor="seurat", subset=False)
    seurat_1pct = np.isin(var_names, np.asarray(sub.var_names).astype(str)[np.asarray(sub.var["highly_variable"]).astype(bool)])
    del sub
    sets = {"seurat": seurat, "seurat>=1%": seurat_1pct, "scarf": scarf, "scarf-port": port}
    for k, m in sets.items():
        (out / f"genes_{k.replace('>=', '_ge').replace('%', 'pct')}.txt").write_text("\n".join(var_names[m]))
    j = lambda p, q: round(float((p & q).sum() / (p | q).sum()), 4)  # noqa: E731
    summary["gene_sets"] = {k: {"genes": int(m.sum()), "jaccard_with_scarf": j(m, scarf),
                                "jaccard_with_seurat": j(m, seurat),
                                "median_cells_detected": int(np.median(st["detected"][m]))} for k, m in sets.items()}
    log("gene sets", summary["gene_sets"])
    save()

    def cluster(adata, name, pca_scale):
        t0 = time.perf_counter()
        ps.pp.pca(adata, 50, scale=pca_scale, seed=0)
        ps.pp.neighbors(adata, n_neighbors=15, seed=0)
        res = {}
        for r in RESOLUTIONS:
            ps.tl.leiden(adata, resolution=r, seed=0)
            res[str(r)] = scores(np.asarray(adata.obs["leiden"]).astype(np.int64), types)
        summary["runs"][name] = {"by_resolution": res, "seconds": round(time.perf_counter() - t0, 1)}
        log(name, res["1.0"])
        save()

    for k in ("seurat", "seurat>=1%", "scarf", "scarf-port"):
        cluster(x[:, sets[k]], f"{k} genes, log1e4", False)
    del x
    # Scarf's normalization: counts of the selected genes only, to 1,000 per cell, log, scaled
    raw = prep()
    for k in ("scarf", "seurat", "seurat>=1%", "scarf-port"):
        y = raw[:, sets[k]]
        ps.pp.normalize_total(y, target_sum=1000)
        ps.pp.log1p(y)
        cluster(y, f"{k} genes, scarf normalization", True)
        del y
    log("done")


if __name__ == "__main__":
    main()
