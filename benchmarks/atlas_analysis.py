"""Atlas-scale biological analysis, and an interpretation check against scanpy.

Three questions, one script:

1. Does the out-of-core path do *biology* at atlas scale — cell types,
   markers, rare populations — on the full 1,306,127-cell mouse-brain file
   inside the same 6 GB budget the timing campaigns use?
2. On a size both tools complete (800,000 cells), does a Polariseq analysis
   and a scanpy analysis of the same cells lead to the same interpretation —
   same clusters, same marker genes, same cell-type assignments?
3. What does the interpretation look like past the reference tool's limit?

Method parity matters more here than speed. Marker statistics for the
out-of-core runs come from `streaming_cluster_stats_normalized`: the same
normalize-total + log1p transform, the same per-cluster kernel, two passes
over the file, never materialised. Scanpy runs its own
`sc.tl.rank_genes_groups(method="t-test")` on the same 2,000 highly-variable
genes both tools select (their HVG overlap is 1.0 by Jaccard, measured in the
equivalence campaign). Annotation uses PanglaoDB's brain markers with the
same z-score-across-clusters scoring for both tools, so a label disagreement
is a data disagreement, not a scoring difference.

Outputs land in ``benchmarks/results/atlas/<phase>/`` as .npy/.csv/.json —
the figure script reads those, never this module's internals.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/results/validation_data"
OUT = ROOT / "benchmarks/results/atlas"

# The parameters the timing campaign uses (telemetry `params`), so the
# biology here is the biology of the timed benchmark runs.
PARAMS = dict(n_hvg=2000, n_pcs=50, n_neighbors=15, target_sum=1e4,
              resolution=1.0, seed=0, min_genes=200)
BUDGET_GB = 6.0
THREADS = 4

#: Clusters at or below this share of all cells count as rare populations.
RARE_FRACTION = 0.001

#: Curated mouse-brain marker panel, the annotation used in the benchmark.
#:
#: PanglaoDB's best-match scored two obvious misses on this data (an
#: endothelial cluster on Cldn5/Sparc/Igfbp7 as "Pinealocytes", an erythroid
#: cluster on Hbb-bs/Hba-a1 as "Adrenergic neurons") — best-of-N marker
#: scoring will do that when two types share a low-weight marker. The panel
#: below is the standard E18 cortex taxonomy a reviewer can check against any
#: atlas, and the SAME panel scores both tools in the comparison, so label
#: agreement measures the data, not the scoring. Gene symbols are mouse
#: style, as the 10x matrix names them.
CANONICAL_PANEL: dict[str, list[str]] = {
    "Newborn neurons": ["Dcx", "Sox11", "Igfbpl1", "Stmn2", "Tubb3"],
    "Excitatory neurons": ["Slc17a7", "Satb2", "Neurod6", "Camk2a"],
    "Interneurons": ["Gad1", "Gad2", "Dlx1", "Dlx2", "Arx"],
    "Intermediate progenitors": ["Eomes", "Insm1", "Neurog2"],
    "Radial glia": ["Sox2", "Hes1", "Fabp7", "Slc1a3", "Vim"],
    "Astrocytes": ["Aqp4", "Gja1", "Slc1a2", "S100b"],
    "OPCs": ["Pdgfra", "Ptprz1", "Cspg4"],
    "Oligodendrocytes": ["Mog", "Mbp", "Plp1"],
    "Microglia": ["P2ry12", "Tmem119", "Cx3cr1", "C1qb", "Tyrobp"],
    "Endothelial cells": ["Cldn5", "Pecam1", "Kdr", "Flt1"],
    "Pericytes": ["Pdgfrb", "Rgs5", "Kcnj8", "Des"],
    "Meningeal fibroblasts": ["Col1a1", "Col3a1", "Dcn", "Lum"],
    "Vascular smooth muscle": ["Acta2", "Tagln", "Myh11"],
    "Erythroid cells": ["Hbb-bs", "Hba-a1", "Hba-a2", "Alas2"],
    "Cajal-Retzius cells": ["Reln", "Lhx1", "Trp73", "Pcp4"],
    "Ependymal cells": ["Foxj1", "Tmem212", "Ccdc153"],
    "Choroid plexus": ["Ttr", "Folr1", "Kl"],
    "Neuroblasts": ["Neurod2", "Mdk", "Ezr", "Isl1"],
}

FILES = {"800k": DATA / "brain800k.h5ad", "1306k": DATA / "brain1306k.h5ad"}


def env_record(phase: str, path: Path) -> dict:
    import platform
    import polariseq as ps
    return {
        "phase": phase,
        "path": str(path),
        "params": PARAMS,
        "budget_gb": BUDGET_GB,
        "threads": THREADS,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "polariseq": ps.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# shared maths: Welch markers and PanglaoDB annotation, used by both tools
# --------------------------------------------------------------------------

def welch_markers(stats: dict, gene_names: list[str], n_top: int = 50) -> pd.DataFrame:
    """Rank genes per cluster by Welch's t (cluster vs rest), from streaming
    per-cluster sums, sums of squares and expressing-cell counts.

    This is the same statistic scanpy's ``rank_genes_groups(method="t-test")``
    computes, evaluated from sufficient statistics instead of the matrix —
    which is the only way to have it at 1.3M cells on an 8 GB machine.
    """
    sums = np.asarray(stats["sums"], dtype=np.float64)       # (genes, clusters)
    sq = np.asarray(stats["sq_sums"], dtype=np.float64)
    det = np.asarray(stats["counts"], dtype=np.float64)      # expressing cells
    sizes = np.asarray(stats["cluster_sizes"], dtype=np.float64)
    g, k = sums.shape

    total_sum = sums.sum(axis=1)
    total_sq = sq.sum(axis=1)
    total_det = det.sum(axis=1)
    n_all = sizes.sum()

    rows = []
    for c in range(k):
        n_c = sizes[c]
        n_r = n_all - n_c
        if n_c < 2 or n_r < 2:
            continue
        mean_c = sums[:, c] / n_c
        var_c = np.maximum(0.0, (sq[:, c] - n_c * mean_c**2) / (n_c - 1))
        sum_r = total_sum - sums[:, c]
        sq_r = total_sq - sq[:, c]
        mean_r = sum_r / n_r
        var_r = np.maximum(0.0, (sq_r - n_r * mean_r**2) / (n_r - 1))
        denom = np.sqrt(var_c / n_c + var_r / n_r)
        t = np.where(denom > 0, (mean_c - mean_r) / np.maximum(denom, 1e-30), 0.0)
        rows.append(pd.DataFrame({
            "cluster": c,
            "gene": gene_names,
            "scores": t,
            "logfoldchanges": mean_c - mean_r,
            "pct_expr": det[:, c] / n_c,
            "pct_expr_rest": (total_det - det[:, c]) / n_r,
        }).nlargest(n_top, "scores"))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def annotate(means: np.ndarray, gene_names: list[str],
             panel: dict[str, list[str]] | None = None,
             organ: str = "Brain") -> dict[int, dict]:
    """Best cell type per cluster, z-scored across clusters.

    `means` is (genes, clusters) normalised mean expression. Scoring follows
    `ps.tl.auto_annotate`: per-gene z-scores across clusters, summed over the
    type's markers, so a type wins on *relative* specificity rather than on
    owning one ubiquitously expressed gene. With `panel`, those curated
    markers are scored (the benchmark's annotation); the default is PanglaoDB.
    """
    from polariseq.panglaodb import get_panglaodb_markers
    markers = panel if panel is not None else get_panglaodb_markers(
        organ=organ, species="mouse")
    by_symbol = {str(n).upper(): i for i, n in enumerate(gene_names)}

    sd = means.std(axis=1, keepdims=True)
    z = np.where(sd > 0, (means - means.mean(axis=1, keepdims=True))
                 / np.maximum(sd, 1e-30), 0.0)

    out: dict[int, dict] = {}
    for ct, genes in markers.items():
        idx = [by_symbol.get(gn.upper()) for gn in genes]
        idx = [i for i in idx if i is not None]
        if panel is None and len(idx) < 3:
            continue
        if len(idx) < 2:
            continue
        score = z[idx, :].sum(axis=0)
        for c in range(means.shape[1]):
            cur = out.get(c)
            if cur is None or score[c] > cur["score"]:
                out[c] = {"cell_type": ct, "score": float(score[c]),
                          "n_markers_found": len(idx)}
    return out


def rare_clusters(sizes: np.ndarray) -> np.ndarray:
    n = sizes.sum()
    lo = max(2, RARE_FRACTION * n)
    return np.flatnonzero((sizes <= lo) & (sizes >= 2))


# --------------------------------------------------------------------------
# phase: polariseq, out-of-core
# --------------------------------------------------------------------------

def run_polariseq(size: str) -> None:
    import polariseq as ps
    from polariseq._polariseq import streaming_cluster_stats_normalized

    path = FILES[size]
    out = OUT / f"polariseq_{size}"
    out.mkdir(parents=True, exist_ok=True)
    env = env_record("polariseq", path)
    (out / "env.json").write_text(json.dumps(env, indent=1))

    ps.set_budget(ram_gb=BUDGET_GB, threads=THREADS)

    t0 = time.perf_counter()
    adata = ps.process_diskbacked(str(path), **PARAMS)
    t_pipe = time.perf_counter() - t0

    t0 = time.perf_counter()
    ps.tl.umap(adata, seed=PARAMS["seed"])
    t_umap = time.perf_counter() - t0

    labels_kept = np.asarray(adata.obs["leiden"]).astype(np.uint32)
    n_clusters = int(labels_kept.max()) + 1
    obs_names = np.asarray(adata.obs_names, dtype=object)

    # Full-file labels with a sentinel for QC-dropped cells: the streaming
    # kernel ignores ids >= n_clusters, so the two label vectors describe the
    # same cells without padding tricks.
    import anndata as ad
    full_names = pd.Index(
        np.asarray(ad.read_h5ad(str(path), backed="r").obs_names, dtype=object))
    pos = full_names.get_indexer(pd.Index(obs_names))
    assert (pos >= 0).all(), "kept cells must be a subset of the file's"
    labels_full = np.full(len(full_names), n_clusters, dtype=np.uint32)
    labels_full[pos] = labels_kept

    t0 = time.perf_counter()
    stats = streaming_cluster_stats_normalized(
        str(path), labels_full, n_clusters, PARAMS["target_sum"])
    t_stats = time.perf_counter() - t0

    var_names = [str(v) for v in ad.read_h5ad(str(path), backed="r").var_names]
    sizes = np.asarray(stats["cluster_sizes"], dtype=np.int64)
    means = (np.asarray(stats["sums"], dtype=np.float64)
             / np.maximum(sizes[None, :], 1))

    # The sufficient statistics are the expensive part; save them so the
    # annotation can be revisited without touching the file again.
    np.savez_compressed(
        out / "stats.npz", sums=stats["sums"], sq_sums=stats["sq_sums"],
        det=stats["counts"], sizes=sizes)
    (out / "var_names.txt").write_text("\n".join(var_names))

    markers = welch_markers(stats, var_names, n_top=50)
    markers.to_csv(out / "markers.csv", index=False)

    ann = annotate(means, var_names, panel=CANONICAL_PANEL)
    hvg = set(map(str, adata.uns.get("hvg_names", [])))
    rare = rare_clusters(sizes)
    clusters = pd.DataFrame([{
        "cluster": c,
        "size": int(sizes[c]),
        "cell_type": ann.get(c, {}).get("cell_type", "?"),
        "top_markers": ", ".join(
            markers[markers.cluster == c].gene.head(5).tolist()),
        "rare": bool(c in rare),
    } for c in range(n_clusters)]).sort_values("size", ascending=False)
    clusters.to_csv(out / "clusters.csv", index=False)

    np.save(out / "labels.npy", labels_kept)
    np.save(out / "umap.npy", np.asarray(adata.obsm["X_umap"]))
    np.save(out / "pca.npy", np.asarray(adata.obsm["X_pca"], dtype=np.float32))
    np.savetxt(out / "obs_names.txt", obs_names, fmt="%s")
    (out / "hvg.txt").write_text("\n".join(sorted(hvg)))
    (out / "summary.json").write_text(json.dumps({
        "n_cells_file": int(len(full_names)),
        "n_cells_kept": int(len(obs_names)),
        "n_clusters": int(n_clusters),
        "rare_clusters": [int(c) for c in rare],
        "timing_s": {"pipeline": round(t_pipe, 1), "umap": round(t_umap, 1),
                     "streaming_stats": round(t_stats, 1)},
        "annotation": {str(c): v for c, v in ann.items()},
    }, indent=1))
    print(f"[polariseq {size}] {len(obs_names):,} cells, {n_clusters} clusters, "
          f"pipeline {t_pipe:.0f}s umap {t_umap:.0f}s stats {t_stats:.0f}s")
    print(clusters.head(12).to_string(index=False))


# --------------------------------------------------------------------------
# phase: scanpy reference
# --------------------------------------------------------------------------

def run_scanpy(size: str) -> None:
    import scanpy as sc

    path = FILES[size]
    out = OUT / f"scanpy_{size}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(env_record("scanpy", path), indent=1))

    # scanpy holds the full CSR of this file (12.9 GB at 800K) and survives
    # on this 8.59 GB machine only through compression — which needs headroom
    # to work with. The telemetry campaign gates every run on available
    # memory for exactly this reason (a scanpy 800K run launched on a busy
    # machine is killed, not slowed); wait for the same quiet here.
    import psutil
    deadline = time.time() + 1800
    while time.time() < deadline:
        avail = psutil.virtual_memory().available
        if avail >= 4.1e9:
            break
        print(f"[scanpy {size}] waiting for memory headroom "
              f"({avail / 1e9:.2f} GB available, need 4.10 GB)", flush=True)
        time.sleep(20)

    t0 = time.perf_counter()
    adata = sc.read_h5ad(str(path))
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=PARAMS["min_genes"])
    sc.pp.normalize_total(adata, target_sum=PARAMS["target_sum"])
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=PARAMS["n_hvg"],
                                flavor="seurat", subset=True)
    sc.pp.pca(adata, n_comps=PARAMS["n_pcs"], svd_solver="arpack",
              random_state=PARAMS["seed"])
    sc.pp.neighbors(adata, n_neighbors=PARAMS["n_neighbors"],
                    n_pcs=PARAMS["n_pcs"], random_state=PARAMS["seed"])
    sc.tl.leiden(adata, resolution=PARAMS["resolution"],
                 random_state=PARAMS["seed"],
                 flavor="igraph", n_iterations=2, directed=False)
    t_pipe = time.perf_counter() - t0

    t0 = time.perf_counter()
    sc.tl.rank_genes_groups(adata, "leiden", method="t-test", n_genes=50)
    t_de = time.perf_counter() - t0

    labels = np.asarray(adata.obs["leiden"]).astype(np.uint32)
    clusters = np.unique(labels)
    n_clusters = len(clusters)

    # Cluster means over the HVG subset, for annotation on the same feature
    # universe the comparison uses. Kept sparse: densifying an 800K × 2,000
    # matrix would cost 6.4 GB for no benefit.
    X = adata.X
    means = np.zeros((X.shape[1], n_clusters))
    for ci, c in enumerate(clusters):
        m = X[labels == c].mean(axis=0)
        means[:, ci] = np.asarray(m).flatten()
    var_names = [str(v) for v in adata.var_names]
    np.savez_compressed(out / "means.npz", means=means)
    (out / "var_names.txt").write_text("\n".join(var_names))
    ann = annotate(means, var_names, panel=CANONICAL_PANEL)

    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)[clusters]
    rows = []
    for ci, c in enumerate(clusters):
        top = adata.uns["rank_genes_groups"]["names"][str(c)]
        rows.append({
            "cluster": int(c),
            "size": int(sizes[ci]),
            "cell_type": ann.get(int(c), {}).get("cell_type", "?"),
            "top_markers": ", ".join(map(str, top[:5])),
            "rare": bool(c in rare_clusters(sizes.astype(np.float64))),
        })
    pdf = pd.DataFrame(rows).sort_values("size", ascending=False)
    pdf.to_csv(out / "clusters.csv", index=False)
    np.save(out / "labels.npy", labels)
    np.savetxt(out / "obs_names.txt", np.asarray(adata.obs_names, dtype=object), fmt="%s")
    (out / "hvg.txt").write_text("\n".join(map(str, adata.var_names)))
    rg = adata.uns["rank_genes_groups"]
    mk = pd.DataFrame({
        "cluster": np.repeat([int(c) for c in clusters], 50),
        "gene": np.concatenate(
            [list(map(str, rg["names"][str(c)][:50])) for c in clusters]),
    })
    mk.to_csv(out / "markers.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "n_cells_kept": int(adata.n_obs),
        "n_clusters": int(n_clusters),
        "timing_s": {"pipeline": round(t_pipe, 1), "de": round(t_de, 1)},
        "annotation": {str(int(c)): v for c, v in ann.items()},
    }, indent=1))
    print(f"[scanpy {size}] {adata.n_obs:,} cells, {n_clusters} clusters, "
          f"pipeline {t_pipe:.0f}s de {t_de:.0f}s")
    print(pdf.head(12).to_string(index=False))


# --------------------------------------------------------------------------
# phases: recompute statistics / re-annotate without re-running a pipeline
# --------------------------------------------------------------------------

def run_stats(size: str) -> None:
    """Recompute (and save) the streaming statistics for a finished
    polariseq run, from its saved labels — for when annotation changes and
    the pipeline must not run again."""
    import anndata as ad
    from polariseq._polariseq import streaming_cluster_stats_normalized

    path = FILES[size]
    out = OUT / f"polariseq_{size}"
    labels_kept = np.load(out / "labels.npy").astype(np.uint32)
    obs_names = np.loadtxt(out / "obs_names.txt", dtype=object)
    n_clusters = int(labels_kept.max()) + 1

    full_names = pd.Index(
        np.asarray(ad.read_h5ad(str(path), backed="r").obs_names, dtype=object))
    pos = full_names.get_indexer(pd.Index(obs_names))
    assert (pos >= 0).all()
    labels_full = np.full(len(full_names), n_clusters, dtype=np.uint32)
    labels_full[pos] = labels_kept

    stats = streaming_cluster_stats_normalized(
        str(path), labels_full, n_clusters, PARAMS["target_sum"])
    np.savez_compressed(
        out / "stats.npz", sums=stats["sums"], sq_sums=stats["sq_sums"],
        det=stats["counts"], sizes=np.asarray(stats["cluster_sizes"]))
    var_names = [str(v) for v in ad.read_h5ad(str(path), backed="r").var_names]
    (out / "var_names.txt").write_text("\n".join(var_names))
    markers = welch_markers(stats, var_names, n_top=50)
    markers.to_csv(out / "markers.csv", index=False)
    print(f"[stats {size}] recomputed for {len(obs_names):,} cells")


def run_reannotate(size: str) -> None:
    """Redo annotation (canonical panel) for one arm from saved statistics,
    rewriting clusters.csv and the annotation block of summary.json."""
    for arm in ("polariseq", "scanpy"):
        out = OUT / f"{arm}_{size}"
        if not (out / "stats.npz").exists() and not (out / "means.npz").exists():
            continue
        var_names = (out / "var_names.txt").read_text().splitlines()
        if (out / "stats.npz").exists():
            st = np.load(out / "stats.npz")
            means = st["sums"] / np.maximum(st["sizes"][None, :], 1)
        else:
            means = np.load(out / "means.npz")["means"]
        ann = annotate(means, var_names, panel=CANONICAL_PANEL)
        clusters = pd.read_csv(out / "clusters.csv")
        clusters["cell_type"] = clusters.cluster.map(
            lambda c: ann.get(int(c), {}).get("cell_type", "?"))
        clusters.to_csv(out / "clusters.csv", index=False)
        summary = json.loads((out / "summary.json").read_text())
        summary["annotation"] = {str(c): v for c, v in ann.items()}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[reannotate {arm} {size}]",
              clusters[["cluster", "size", "cell_type"]].head(8).to_string(index=False))


# --------------------------------------------------------------------------
# phase: interpretation comparison at the shared size
# --------------------------------------------------------------------------

def run_compare(size: str = "800k") -> None:
    from sklearn.metrics import adjusted_rand_score

    p_lab = np.load(OUT / f"polariseq_{size}/labels.npy")
    s_lab = np.load(OUT / f"scanpy_{size}/labels.npy")
    p_obs = np.loadtxt(OUT / f"polariseq_{size}/obs_names.txt", dtype=object)
    s_obs = np.loadtxt(OUT / f"scanpy_{size}/obs_names.txt", dtype=object)
    if not np.array_equal(p_obs, s_obs):
        # Same file, same QC rule; order is preserved by both tools. If the
        # barcodes ever disagree, the comparison below would be meaningless.
        common, pi, si = np.intersect1d(p_obs, s_obs, return_indices=True)
        p_lab, s_lab = p_lab[pi], s_lab[si]
    else:
        common = p_obs

    p_clusters = pd.read_csv(OUT / f"polariseq_{size}/clusters.csv")
    s_clusters = pd.read_csv(OUT / f"scanpy_{size}/clusters.csv")
    p_ann = dict(zip(p_clusters.cluster, p_clusters.cell_type))
    s_ann = dict(zip(s_clusters.cluster, s_clusters.cell_type))

    # Greedy Jaccard matching (clusters are few; exhaustive is clearer than
    # an assignment solve and leaves unmatched clusters unmatched).
    pk = np.unique(p_lab)
    sk = np.unique(s_lab)
    pairs = []
    for pc in pk:
        pm = p_lab == pc
        best, best_j = None, 0.0
        for sc_ in sk:
            j = np.logical_and(pm, s_lab == sc_).sum() / np.logical_or(pm, s_lab == sc_).sum()
            if j > best_j:
                best, best_j = sc_, j
        if best is not None and best_j >= 0.5:
            pairs.append((int(pc), int(best), float(best_j)))
    matched_p = {p for p, _, _ in pairs}
    matched_s = {s for _, s, _ in pairs}
    in_matched = np.isin(p_lab, list(matched_p)) & np.isin(s_lab, list(matched_s))

    # Marker overlap on the shared feature universe (both tools' HVG sets).
    p_hvg = set((OUT / f"polariseq_{size}/hvg.txt").read_text().splitlines())
    s_hvg = set((OUT / f"scanpy_{size}/hvg.txt").read_text().splitlines())
    universe = p_hvg & s_hvg
    p_mk = pd.read_csv(OUT / f"polariseq_{size}/markers.csv")
    s_mk = pd.read_csv(OUT / f"scanpy_{size}/markers.csv")
    overlaps = []
    for pc, sc_, j in pairs:
        pt = [g for g in p_mk[p_mk.cluster == pc].gene.head(20) if g in universe]
        st = [g for g in s_mk[s_mk.cluster == sc_].gene.head(20) if g in universe]
        inter = len(set(pt) & set(st))
        overlaps.append({
            "polariseq_cluster": pc, "scanpy_cluster": sc_, "jaccard_cells": j,
            "polariseq_type": p_ann.get(pc, "?"), "scanpy_type": s_ann.get(sc_, "?"),
            "type_agrees": p_ann.get(pc) == s_ann.get(sc_),
            "marker_overlap_top20": inter,
            "marker_overlap_frac": round(inter / max(len(set(pt) | set(st)), 1), 3),
        })

    ann_agree = sum(o["type_agrees"] for o in overlaps)
    res = {
        "size": size,
        "n_cells_compared": int(len(common)),
        "ari": round(float(adjusted_rand_score(p_lab, s_lab)), 3),
        "polariseq_clusters": int(len(pk)),
        "scanpy_clusters": int(len(sk)),
        "matched_pairs": len(pairs),
        "cells_in_matched_pairs": round(float(in_matched.mean()), 3),
        "matched_pairs_with_same_annotation": ann_agree,
        "mean_marker_overlap_top20": round(
            float(np.mean([o["marker_overlap_top20"] for o in overlaps])), 1),
        "hvg_universe_shared": len(universe),
        "pairs": overlaps,
    }
    (OUT / f"comparison_{size}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))
    print(pd.DataFrame(overlaps).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",
                    choices=["polariseq", "scanpy", "compare", "stats",
                             "reannotate"])
    ap.add_argument("--size", default="800k", choices=list(FILES))
    args = ap.parse_args()
    if args.phase == "polariseq":
        run_polariseq(args.size)
    elif args.phase == "scanpy":
        run_scanpy(args.size)
    elif args.phase == "stats":
        run_stats(args.size)
    elif args.phase == "reannotate":
        run_reannotate(args.size)
    else:
        run_compare(args.size)


if __name__ == "__main__":
    main()
