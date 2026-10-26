"""Million-cell biology: Polariseq out-of-core against a published atlas.

The human fetal atlas (Cao et al. 2020, Science 370:eaba7721; GEO GSE156793)
publishes, for each of its 4,062,980 cells, the cell type the authors
assigned (77 main types across 15 organs) and the cell's position in their
global UMAP; supplement S8 lists each type's differentially expressed genes.
This script analyses the atlas with Polariseq's out-of-core mode on the
8.59 GB laptop, knowing none of that, and then asks whether the result is
the published biology:

1. Do Polariseq's clusters reproduce the authors' cell types? Agreement
   (ARI, NMI), and per published type the share of its cells that land in
   the cluster holding most of them, and that cluster's purity.
2. Are the types recovered, rare ones included? A type counts as recovered
   when one cluster holds at least half its cells and is at least half made
   of it.
3. Are the markers the published markers? Each cluster's top Welch markers
   (streamed from the file, never materialised) against the S8 markers of
   the type the cluster mostly contains, with the overlap against the
   *other* types' markers as the chance baseline.

    python benchmarks/fetal_biology.py polariseq --size 4062k
    python benchmarks/fetal_biology.py compare --size 4062k

Outputs land in ``benchmarks/results/fetal/<size>/``.
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
sys.path.insert(0, str(ROOT / "benchmarks"))
from atlas_analysis import welch_markers  # noqa: E402

DATA = ROOT / "benchmarks/results/fetal_data"
OUT = ROOT / "benchmarks/results/fetal"
CELLS = DATA / "fetal_cells.csv.gz"
DE = DATA / "GSE156793_S8_DE_gene_cells.csv.gz"
#: Every fetal file present, keyed by the size in its name ("4062k"), so a
#: smaller file built for a trial run is selectable without editing this.
FILES = {p.stem.removeprefix("fetal"): p for p in sorted(DATA.glob("fetal*k.h5ad"))}

#: The timing campaign's parameters, so the biology is the biology of the
#: runs the benchmark times.
PARAMS = dict(n_hvg=2000, n_pcs=50, n_neighbors=15, target_sum=1e4,
              resolution=1.0, seed=0, min_genes=200)
BUDGET_GB = 6.0
THREADS = 4

#: Published markers per type: S8 genes whose highest-expressing type is
#: this one, q < 0.05, protein-coding, at least MIN_FOLD times the
#: second-highest type, then the N most expressed in the type. (Ranking by
#: fold change alone promotes barely detected genes: SLC26A4 and IQCF3
#: would head the excitatory-neuron list.)
N_PUBLISHED_MARKERS = 20
MIN_FOLD = 2.0
N_OUR_MARKERS = 20
#: Recovery: one cluster holds >= this share of the type, and is >= this pure.
RECOVERY = 0.5


def published_markers() -> dict[str, list[str]]:
    de = pd.read_csv(DE, encoding="utf-8-sig")
    de["gene"] = de["gene_short_name"].str.strip("'")
    de = de[(de["qval"] < 0.05) & (de["gene_type"] == "protein_coding")
            & (de["fold.change"] >= MIN_FOLD)]
    de = (de.sort_values("max.expr", ascending=False)
            .drop_duplicates(["max.cluster", "gene"]))
    return {t: g["gene"].head(N_PUBLISHED_MARKERS).tolist()
            for t, g in de.groupby("max.cluster")}


def out_dir(size: str, resolution: float) -> Path:
    """One directory per (size, resolution); the campaign's 1.0 keeps the
    plain name so nothing downstream has to know about the sweep."""
    return OUT / (size if resolution == PARAMS["resolution"] else f"{size}_res{resolution:g}")


def run_polariseq(size: str, resolution: float | None = None) -> None:
    import anndata as ad

    import polariseq as ps
    from polariseq._polariseq import streaming_cluster_stats_normalized

    path = FILES[size]
    params = dict(PARAMS)
    if resolution is not None:
        params["resolution"] = resolution
    out = out_dir(size, params["resolution"])
    out.mkdir(parents=True, exist_ok=True)
    ps.set_budget(ram_gb=BUDGET_GB, threads=THREADS)
    env = {"path": str(path.relative_to(ROOT)), "params": params, "budget_gb": BUDGET_GB,
           "threads": THREADS, "polariseq": ps.__version__,
           "threads_used": int(ps._polariseq.thread_count()),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    t0 = time.perf_counter()
    adata = ps.process_diskbacked(str(path), **params)
    t_pipe = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps.tl.umap(adata, seed=params["seed"])
    t_umap = time.perf_counter() - t0

    labels_kept = np.asarray(adata.obs["leiden"]).astype(np.uint32)
    n_clusters = int(labels_kept.max()) + 1
    obs_names = np.asarray(adata.obs_names, dtype=object)
    backed = ad.read_h5ad(str(path), backed="r")
    full_names = pd.Index(np.asarray(backed.obs_names, dtype=object))
    var_names = [str(v) for v in backed.var_names]
    backed.file.close()
    pos = full_names.get_indexer(pd.Index(obs_names))
    assert (pos >= 0).all(), "kept cells must be a subset of the file's"
    labels_full = np.full(len(full_names), n_clusters, dtype=np.uint32)
    labels_full[pos] = labels_kept

    t0 = time.perf_counter()
    stats = streaming_cluster_stats_normalized(str(path), labels_full, n_clusters,
                                               params["target_sum"])
    t_stats = time.perf_counter() - t0

    sizes = np.asarray(stats["cluster_sizes"], dtype=np.int64)
    np.savez_compressed(out / "stats.npz", sums=stats["sums"], sq_sums=stats["sq_sums"],
                        det=stats["counts"], sizes=sizes)
    (out / "var_names.txt").write_text("\n".join(var_names))
    welch_markers(stats, var_names, n_top=50).to_csv(out / "markers.csv", index=False)
    np.save(out / "labels.npy", labels_kept)
    np.save(out / "umap.npy", np.asarray(adata.obsm["X_umap"], dtype=np.float32))
    np.savetxt(out / "obs_names.txt", obs_names, fmt="%s")
    (out / "hvg.txt").write_text("\n".join(sorted(map(str, adata.uns.get("hvg_names", [])))))
    (out / "summary.json").write_text(json.dumps({
        **env,
        "n_cells_file": int(len(full_names)),
        "n_cells_kept": int(len(obs_names)),
        "n_clusters": n_clusters,
        "leiden_quality": float(adata.uns["leiden"]["quality"]),
        "timing_s": {"pipeline": round(t_pipe, 1), "umap": round(t_umap, 1),
                     "streaming_stats": round(t_stats, 1)},
    }, indent=1))
    print(f"[polariseq {size}] {len(obs_names):,} of {len(full_names):,} cells, "
          f"{n_clusters} clusters; pipeline {t_pipe:.0f}s, umap {t_umap:.0f}s, "
          f"stats {t_stats:.0f}s")


def run_compare(size: str, resolution: float | None = None, out: Path | None = None) -> None:
    """Score a labelling against the published cell types. `out` is the
    folder holding labels.npy, obs_names.txt and markers.csv; by default the
    global run's, and fetal_matched.py passes its organ-by-organ one so both
    are judged by the same rule."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if out is None:
        out = out_dir(size, PARAMS["resolution"] if resolution is None else resolution)
    labels = np.load(out / "labels.npy").astype(np.int64)
    obs = np.loadtxt(out / "obs_names.txt", dtype=object)
    cells = pd.read_csv(CELLS, index_col="cell",
                        usecols=["cell", "Main_cluster_name", "Organ"])
    pub = cells.loc[obs, "Main_cluster_name"].to_numpy()
    organ = cells.loc[obs, "Organ"].to_numpy()

    ct = pd.crosstab(pd.Series(pub, name="published"), pd.Series(labels, name="cluster"))
    type_size = ct.sum(axis=1)
    cluster_size = ct.sum(axis=0)
    best = ct.idxmax(axis=1)
    recall = ct.max(axis=1) / type_size
    purity = pd.Series([ct.loc[t, best[t]] / cluster_size[best[t]] for t in ct.index],
                       index=ct.index)
    recovered = (recall >= RECOVERY) & (purity >= RECOVERY)
    majority = ct.idxmax(axis=0)                     # cluster -> published type
    same = (pd.Series(labels).map(majority).to_numpy() == pub)

    # A type split across several clusters is still recovered biology: a
    # reader annotating each cluster would label them all as that type. So
    # the same two numbers are also computed over every cluster the type
    # dominates, not just the largest one. Excitatory neurons in the trial
    # slice sat at recall 0.27 / purity 0.97 by the strict rule and 0.99 /
    # 0.97 by this one.
    own = {t: [c for c in ct.columns if majority[c] == t] for t in ct.index}
    held = pd.Series({t: ct.loc[t, cs].sum() for t, cs in own.items()})
    recall_split = held / type_size
    purity_split = pd.Series({t: (held[t] / cluster_size[cs].sum()) if cs else 0.0
                              for t, cs in own.items()})
    recovered_split = (recall_split >= RECOVERY) & (purity_split >= RECOVERY)

    # Markers: ours per cluster vs the published markers of the cluster's
    # majority type; the baseline is the same overlap against every other type.
    ours = pd.read_csv(out / "markers.csv")
    pubm = published_markers()
    rows = []
    for c in ct.columns:
        t = majority[c]
        mine = set(ours[ours.cluster == c].gene.head(N_OUR_MARKERS))
        if t not in pubm or not mine:
            continue
        hit = len(mine & set(pubm[t]))
        others = [len(mine & set(g)) for u, g in pubm.items() if u != t]
        rows.append({"cluster": int(c), "size": int(cluster_size[c]), "published_type": t,
                     "purity": round(float(ct.loc[t, c] / cluster_size[c]), 3),
                     "overlap": hit, "overlap_other_types_mean": round(float(np.mean(others)), 2),
                     "overlap_other_types_max": int(max(others)),
                     "shared_genes": ", ".join(sorted(mine & set(pubm[t])))})
    markers = pd.DataFrame(rows).sort_values("size", ascending=False)
    markers.to_csv(out / "marker_agreement.csv", index=False)

    per_type = pd.DataFrame({
        "cells": type_size, "best_cluster": best,
        "recall": recall.round(3), "purity": purity.round(3), "recovered": recovered,
        "clusters_of_this_type": [len(own[t]) for t in ct.index],
        "recall_split": recall_split.round(3), "purity_split": purity_split.round(3),
        "recovered_split": recovered_split,
    }).sort_values("cells", ascending=False)
    per_type.to_csv(out / "per_type.csv")
    ct.to_csv(out / "crosstab.csv")

    rare = per_type[per_type.cells < 1000]
    res = {
        "size": size,
        "resolution": PARAMS["resolution"] if resolution is None else resolution,
        "n_cells": int(len(labels)),
        "n_clusters": int(ct.shape[1]),
        "n_published_types": int(ct.shape[0]),
        "ari": round(float(adjusted_rand_score(pub, labels)), 3),
        "nmi": round(float(normalized_mutual_info_score(pub, labels)), 3),
        "nmi_organ": round(float(normalized_mutual_info_score(organ, labels)), 3),
        "cells_in_cluster_of_own_type": round(float(same.mean()), 4),
        "types_recovered": int(recovered.sum()),
        "types_recovered_allowing_splits": int(recovered_split.sum()),
        "rare_types": int(len(rare)),
        "rare_types_recovered": int(rare.recovered.sum()),
        "rare_types_recovered_allowing_splits": int(rare.recovered_split.sum()),
        "marker_overlap_mean": round(float(markers.overlap.mean()), 2),
        "marker_overlap_other_types_mean": round(float(markers.overlap_other_types_mean.mean()), 2),
        "clusters_with_marker_overlap": int((markers.overlap > 0).sum()),
        "clusters_scored": int(len(markers)),
        "recovery_rule": f"one cluster holds >= {RECOVERY:.0%} of the type and is >= {RECOVERY:.0%} that type",
        "recovery_rule_splits": f"the clusters this type dominates together hold >= {RECOVERY:.0%} "
                                f"of it and are >= {RECOVERY:.0%} that type",
        "published_markers": f"S8, q<0.05, protein-coding, fold change >= {MIN_FOLD}, top {N_PUBLISHED_MARKERS} by expression",
    }
    (out / "comparison.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(per_type.head(20).to_string())
    print(markers.head(20).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["polariseq", "compare"])
    ap.add_argument("--size", default="4062k", choices=list(FILES))
    ap.add_argument("--resolution", type=float, default=None,
                    help="Leiden resolution; the default is the campaign's 1.0, and "
                         "anything else writes to <size>_res<r>/")
    a = ap.parse_args()
    if a.phase == "polariseq":
        run_polariseq(a.size, a.resolution)
    else:
        run_compare(a.size, a.resolution)


if __name__ == "__main__":
    main()
