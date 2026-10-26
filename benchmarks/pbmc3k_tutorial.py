"""Scanpy's PBMC 3k tutorial, reproduced in scanpy and repeated in Polariseq.

The tutorial ("Preprocessing and clustering 3k PBMCs (legacy workflow)",
docs/tutorials/basics/clustering-2017.ipynb in the scanpy repository) is the
most widely read scanpy analysis: its code, figures and cell-type calls are
public. Three arms run on the same 10x file:

- ``scanpy-tutorial``: the tutorial's code, step for step. Checked against
  its published outputs (cells after QC, cluster count, top markers).
- ``polariseq``: the same analysis in Polariseq, including scaling each
  gene to unit variance (clipped at 10) before PCA. Two tutorial steps have
  no Polariseq counterpart and are left out rather than imitated: regressing
  out total counts and mitochondrial fraction, and ``seurat_v3`` gene
  selection on raw counts (Polariseq selects with the ``seurat`` flavor on
  the log-normalised data).
- ``scanpy-same-steps``: scanpy running Polariseq's steps, scaling included.
  Its distance from the tutorial arm is what the two missing steps change;
  its distance from the Polariseq arm is what the implementation changes.

Every arm is labelled by the same rule, the tutorial's own marker table
scored per cluster (z-score across clusters, summed over each type's
markers), so the cell types being compared come from the data, not from a
person reading each run's plots.

    python benchmarks/pbmc3k_tutorial.py            # all arms, then compare
    python benchmarks/pbmc3k_tutorial.py compare    # recompute the comparison

Outputs land in ``benchmarks/results/pbmc3k/<arm>/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MTX = ROOT / "benchmarks/results/validation_data/pbmc3k_mtx/filtered_gene_bc_matrices/hg19"
OUT = ROOT / "benchmarks/results/pbmc3k"
#: With --scaled, Polariseq and the same-steps control also scale genes
#: before PCA (the tutorial does, together with regression and seurat_v3
#: selection); results then go to pbmc3k_scaled/. The main comparison is
#: unscaled: scaling alone changes the clustering at the tutorial resolution.
SCALED = False

#: The tutorial's parameters.
MIN_GENES, MAX_GENES, MAX_PCT_MT = 200, 2500, 5.0
N_HVG, N_NEIGHBORS, N_PCS, RESOLUTION = 2000, 10, 40, 0.7
#: scanpy's UMAP default; Polariseq's own default is 0.1.
UMAP_MIN_DIST = 0.5

#: The tutorial's marker list, in its dot-plot order.
MARKER_GENES = [
    "IL7R", "CD79A", "MS4A1", "CD8A", "CD8B", "LYZ", "CD14",
    "LGALS3", "S100A8", "GNLY", "NKG7", "KLRB1",
    "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP",
]

#: The tutorial's table of markers per cell type ("Leiden Group | Markers |
#: Cell Type"), with its names from `new_cluster_names`.
CELL_TYPES: dict[str, list[str]] = {
    "CD4 T": ["IL7R"],
    "CD14+ Monocytes": ["CD14", "LYZ"],
    "B": ["MS4A1"],
    "CD8 T": ["CD8A"],
    "NK": ["GNLY", "NKG7"],
    "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
    "Dendritic": ["FCER1A", "CST3"],
    "Megakaryocytes": ["PPBP"],
}

#: What the published notebook reports, for the reproduction check.
PUBLISHED = {
    "n_cells_after_qc": 2638,
    "n_genes_after_filter": 13714,
    "n_clusters": 8,
    # sc.tl.rank_genes_groups(..., method="t-test"), top 5 per cluster
    "top5_ttest": {
        "0": ["LTB", "IL32", "IL7R", "MALAT1", "CD2"],
        "1": ["CD74", "CD79A", "HLA-DRA", "CD79B", "HLA-DPB1"],
        "2": ["LYZ", "S100A9", "S100A8", "TYROBP", "FTL"],
        "3": ["NKG7", "GNLY", "GZMB", "CTSW", "PRF1"],
        "4": ["CCL5", "NKG7", "CST7", "GZMA", "B2M"],
        "5": ["LST1", "FCER1G", "AIF1", "FCGR3A", "COTL1"],
        "6": ["HLA-DPA1", "HLA-DPB1", "HLA-DRB1", "HLA-DRA", "CD74"],
        "7": ["PF4", "SDPR", "GNG11", "PPBP", "NRGN"],
    },
    # `new_cluster_names`, in cluster order
    "cluster_names": ["CD4 T", "B", "CD14+ Monocytes", "NK", "CD8 T",
                      "FCGR3A+ Monocytes", "Dendritic", "Megakaryocytes"],
}

N_TOP_MARKERS = 25


# --------------------------------------------------------------------------
# shared: annotation and per-cluster summaries, identical for every arm
# --------------------------------------------------------------------------

def cluster_means(x, labels: np.ndarray, genes: list[str],
                  var_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean log-normalised expression and fraction of expressing cells per
    cluster, for `genes`, from a cells x genes scipy/numpy matrix."""
    import scipy.sparse as sp

    idx = {g: i for i, g in enumerate(var_names)}
    cols = [idx[g] for g in genes]
    sub = x[:, cols]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    groups = sorted(set(labels.tolist()))
    mean = pd.DataFrame([sub[labels == g].mean(axis=0) for g in groups],
                        index=groups, columns=genes)
    frac = pd.DataFrame([(sub[labels == g] > 0).mean(axis=0) for g in groups],
                        index=groups, columns=genes)
    return mean, frac


def annotate(mean: pd.DataFrame) -> dict[int, str]:
    """Cell type per cluster from the tutorial's marker table.

    Per-gene z-scores across clusters, summed over each type's markers; the
    best type wins. A type is scored on relative specificity, so a gene every
    cluster expresses a little of cannot carry it."""
    z = (mean - mean.mean(axis=0)) / mean.std(axis=0).replace(0, 1)
    scores = pd.DataFrame({t: z[g].sum(axis=1) for t, g in CELL_TYPES.items()})
    return {int(c): str(scores.loc[c].idxmax()) for c in scores.index}


def write_arm(name: str, *, obs_names, labels, umap, pca, hvg, markers: pd.DataFrame,
              x_norm, var_names, timing: dict, extra: dict | None = None) -> None:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int32)
    mean, frac = cluster_means(x_norm, labels, MARKER_GENES, var_names)
    ann = annotate(mean)
    pd.DataFrame({"cell": obs_names, "cluster": labels,
                  "cell_type": [ann[int(c)] for c in labels]}).to_csv(
        out / "labels.csv", index=False)
    np.save(out / "umap.npy", np.asarray(umap, dtype=np.float32))
    np.save(out / "pca.npy", np.asarray(pca, dtype=np.float32))
    (out / "hvg.txt").write_text("\n".join(sorted(map(str, hvg))))
    markers.to_csv(out / "markers.csv", index=False)
    mean.to_csv(out / "dot_mean.csv")
    frac.to_csv(out / "dot_frac.csv")
    # The same tables by cell type, for the dot plots the figure compares.
    types = np.array([ann[int(c)] for c in labels])
    order = [t for t in CELL_TYPES if t in set(types)]
    codes = np.array([order.index(t) for t in types])
    tmean, tfrac = cluster_means(x_norm, codes, MARKER_GENES, var_names)
    tmean.index = tfrac.index = order
    tmean.to_csv(out / "dot_mean_by_type.csv")
    tfrac.to_csv(out / "dot_frac_by_type.csv")
    sizes = pd.Series(labels).value_counts().sort_index()
    (out / "summary.json").write_text(json.dumps({
        "arm": name,
        "n_cells": int(len(labels)),
        "n_genes_after_filter": int(len(var_names)),
        "n_clusters": int(len(sizes)),
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "annotation": {str(k): v for k, v in ann.items()},
        "timing_s": {k: round(v, 3) for k, v in timing.items()},
        "platform": platform.platform(),
        **(extra or {}),
    }, indent=1))
    print(f"[{name}] {len(labels):,} cells, {len(sizes)} clusters: "
          + ", ".join(f"{c}={ann[int(c)]}" for c in sizes.index))


def scanpy_markers(adata, key: str = "leiden") -> pd.DataFrame:
    r = adata.uns["rank_genes_groups"]
    rows = []
    for g in r["names"].dtype.names:
        for rank in range(N_TOP_MARKERS):
            rows.append({"cluster": int(g), "rank": rank, "gene": str(r["names"][g][rank]),
                         "score": float(r["scores"][g][rank])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def run_scanpy(same_steps: bool) -> None:
    """The tutorial's code (`same_steps=False`) or Polariseq's steps in scanpy."""
    import scanpy as sc

    name = "scanpy-same-steps" if same_steps else "scanpy-tutorial"
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    adata = sc.read_10x_mtx(str(MTX), var_names="gene_symbols")
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    adata = adata[(adata.obs.n_genes_by_counts < MAX_GENES)
                  & (adata.obs.n_genes_by_counts > MIN_GENES)
                  & (adata.obs.pct_counts_mt < MAX_PCT_MT), :].copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    x_norm = adata.X.copy()
    t["qc_normalise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if same_steps:
        sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
        sub = adata[:, adata.var["highly_variable"]].copy()
        if SCALED:
            sc.pp.scale(sub, max_value=10)
        sc.pp.pca(sub, n_comps=50, svd_solver="arpack")
        adata.obsm["X_pca"] = sub.obsm["X_pca"]
        del sub
    else:
        sc.pp.highly_variable_genes(adata, layer="counts", n_top_genes=N_HVG,
                                    min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    flavor="seurat_v3")
        adata.layers["scaled"] = adata.X.toarray()
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"], layer="scaled")
        sc.pp.scale(adata, max_value=10, layer="scaled")
        sc.pp.pca(adata, layer="scaled", svd_solver="arpack")
    t["hvg_pca"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=RESOLUTION, random_state=0, flavor="igraph",
                 n_iterations=2, directed=False)
    t["neighbors_umap_leiden"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sc.tl.rank_genes_groups(adata, "leiden", mask_var="highly_variable", method="t-test")
    t["markers"] = time.perf_counter() - t0

    write_arm(
        name, obs_names=list(adata.obs_names),
        labels=adata.obs["leiden"].astype(int).to_numpy(),
        umap=adata.obsm["X_umap"], pca=adata.obsm["X_pca"],
        hvg=adata.var_names[adata.var["highly_variable"]],
        markers=scanpy_markers(adata), x_norm=x_norm,
        var_names=list(adata.var_names), timing=t,
        extra={"versions": {"scanpy": sc.__version__}},
    )


def run_polariseq() -> None:
    import polariseq as ps

    ps.set_budget(threads=4)
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    adata = ps.read_10x_mtx(str(MTX), lazy=False)
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=MIN_GENES)
    ps.pp.filter_genes(adata, min_cells=3)
    ps.pp.calculate_qc_metrics(adata)
    # the tutorial's cuts are strict inequalities on integer gene counts
    ps.pp.filter_cells(adata, min_genes=MIN_GENES + 1, max_genes=MAX_GENES - 1,
                       max_pct_mt=MAX_PCT_MT)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    x_norm = adata.X.to_scipy()
    var_all = [str(v) for v in adata.var_names]
    t["qc_normalise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ps.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=True)  # 1% floor (default)
    hvg = [str(v) for v in adata.var_names]
    if ps.workflow() == "polariseq":   # the default's second normalization and scaling
        ps.pp.renormalize(adata, target_sum=1000)
        ps.pp.pca(adata, 50, scale=True, max_value=10, seed=0)
    elif SCALED:
        ps.pp.pca(adata, 50, scale=True, max_value=10, seed=0)
    else:
        ps.pp.pca(adata, 50, seed=0)
    t["hvg_pca"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ps.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, seed=0)
    ps.tl.umap(adata, seed=0)   # the default UMAP (multilevel start)
    ps.tl.leiden(adata, resolution=RESOLUTION, seed=0)
    t["neighbors_umap_leiden"] = time.perf_counter() - t0

    # As the tutorial: markers from the log-normalised data of the selected genes.
    t0 = time.perf_counter()
    ps.tl.rank_genes_groups(adata, "leiden", n_genes=N_TOP_MARKERS)
    t["markers"] = time.perf_counter() - t0
    rows = []
    for g, r in adata.uns["rank_genes_groups"].items():
        for rank, (gene, score) in enumerate(zip(r["names"], r["scores"], strict=True)):
            rows.append({"cluster": int(g), "rank": rank, "gene": gene, "score": float(score)})

    write_arm(
        "polariseq", obs_names=[str(o) for o in adata.obs_names],
        labels=np.asarray(adata.obs["leiden"]), umap=adata.obsm["X_umap"],
        pca=adata.obsm["X_pca"], hvg=hvg, markers=pd.DataFrame(rows),
        x_norm=x_norm, var_names=var_all, timing=t,
        extra={"versions": {"polariseq": ps.__version__},
               "leiden": {k: v for k, v in adata.uns["leiden"].items()
                          if isinstance(v, (int, float, str))}},
    )


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def load_arm(name: str) -> dict:
    d = OUT / name
    return {
        "labels": pd.read_csv(d / "labels.csv").set_index("cell"),
        "markers": pd.read_csv(d / "markers.csv"),
        "hvg": set((d / "hvg.txt").read_text().split()),
        "summary": json.loads((d / "summary.json").read_text()),
    }


def reproduction_check(tut: dict) -> dict:
    """Our scanpy run against what the published notebook shows."""
    s = tut["summary"]
    top5 = (tut["markers"][tut["markers"]["rank"] < 5]
            .groupby("cluster")["gene"].apply(list).to_dict())
    per_cluster = {
        c: len(set(top5.get(int(c), [])) & set(genes)) / 5
        for c, genes in PUBLISHED["top5_ttest"].items()
    }
    names_match = [s["annotation"].get(str(i)) == n
                   for i, n in enumerate(PUBLISHED["cluster_names"])]
    return {
        "n_cells_after_qc": [s["n_cells"], PUBLISHED["n_cells_after_qc"]],
        "n_genes_after_filter": [s["n_genes_after_filter"], PUBLISHED["n_genes_after_filter"]],
        "n_clusters": [s["n_clusters"], PUBLISHED["n_clusters"]],
        "top5_overlap_per_cluster": per_cluster,
        "marker_rule_reproduces_published_names": f"{sum(names_match)}/{len(names_match)}",
    }


def compare_pair(a: dict, b: dict) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    common = a["labels"].index.intersection(b["labels"].index)
    la, lb = a["labels"].loc[common], b["labels"].loc[common]
    ta, tb = la["cell_type"], lb["cell_type"]
    per_type = {}
    for t in CELL_TYPES:
        in_a, in_b = ta == t, tb == t
        union = (in_a | in_b).sum()
        per_type[t] = {
            "cells_a": int(in_a.sum()), "cells_b": int(in_b.sum()),
            "jaccard_cells": round(float((in_a & in_b).sum() / union), 3) if union else None,
        }
    # Marker agreement per cell type: the top-N markers of the clusters each
    # tool calls that type (the best-scoring cluster, when several share it).
    def top_by_type(arm: dict) -> dict[str, set]:
        ann = arm["summary"]["annotation"]
        m = arm["markers"]
        out: dict[str, set] = {}
        for c, t in ann.items():
            genes = set(m[(m.cluster == int(c)) & (m["rank"] < N_TOP_MARKERS)].gene)
            out.setdefault(t, set()).update(genes)
        return out
    ma, mb = top_by_type(a), top_by_type(b)
    for t in CELL_TYPES:
        if t in ma and t in mb:
            per_type[t]["marker_jaccard"] = round(len(ma[t] & mb[t]) / len(ma[t] | mb[t]), 3)
    return {
        "n_common_cells": int(len(common)),
        "cells_only_in_a": int(len(a["labels"]) - len(common)),
        "cells_only_in_b": int(len(b["labels"]) - len(common)),
        "ari_clusters": round(float(adjusted_rand_score(la["cluster"], lb["cluster"])), 3),
        "nmi_clusters": round(float(normalized_mutual_info_score(la["cluster"], lb["cluster"])), 3),
        "ari_cell_types": round(float(adjusted_rand_score(ta, tb)), 3),
        "same_cell_type_fraction": round(float((ta == tb).mean()), 4),
        "hvg_jaccard": round(len(a["hvg"] & b["hvg"]) / len(a["hvg"] | b["hvg"]), 3),
        "per_type": per_type,
        "crosstab": pd.crosstab(ta, tb).to_dict(),
    }


def run_compare() -> None:
    arms = {n: load_arm(n) for n in ("scanpy-tutorial", "scanpy-same-steps", "polariseq")}
    report = {
        "reproduction": reproduction_check(arms["scanpy-tutorial"]),
        "polariseq_vs_tutorial": compare_pair(arms["scanpy-tutorial"], arms["polariseq"]),
        "same_steps_vs_tutorial": compare_pair(arms["scanpy-tutorial"], arms["scanpy-same-steps"]),
        "polariseq_vs_same_steps": compare_pair(arms["scanpy-same-steps"], arms["polariseq"]),
    }
    (OUT / "comparison.json").write_text(json.dumps(report, indent=1))
    r = report["reproduction"]
    print("reproduction:", {k: v for k, v in r.items() if k != "top5_overlap_per_cluster"})
    print("  top-5 overlap per cluster:", r["top5_overlap_per_cluster"])
    for k in ("polariseq_vs_tutorial", "same_steps_vs_tutorial", "polariseq_vs_same_steps"):
        c = report[k]
        print(f"{k}: ARI clusters {c['ari_clusters']}, ARI types {c['ari_cell_types']}, "
              f"same type {c['same_cell_type_fraction']:.1%}, HVG J {c['hvg_jaccard']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", nargs="?", default="all",
                    choices=["all", "scanpy-tutorial", "scanpy-same-steps", "polariseq", "compare"])
    ap.add_argument("--scaled", action="store_true",
                    help="scale genes before PCA in Polariseq and the control; "
                         "write to pbmc3k_scaled/")
    a = ap.parse_args()
    global OUT, SCALED
    if a.scaled:
        SCALED, OUT = True, ROOT / "benchmarks/results/pbmc3k_scaled"
    if a.phase in ("all", "scanpy-tutorial"):
        run_scanpy(same_steps=False)
    if a.phase in ("all", "scanpy-same-steps"):
        run_scanpy(same_steps=True)
    if a.phase in ("all", "polariseq"):
        run_polariseq()
    if a.phase in ("all", "compare"):
        run_compare()


if __name__ == "__main__":
    main()
