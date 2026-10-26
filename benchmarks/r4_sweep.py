"""The new methods against Polariseq as published, on one dataset: quality, time and memory.

    python benchmarks/r4_sweep.py DATA [DATA ...] --out OUT --project PROJECT
        --labels csv:FILE:COLUMN | obs:COLUMN | none  [--mode out-of-core|memory]
        [--threads 8] [--ram-gb 100] [--preset full|laptop]

Each pipeline is the analysis of examples/06_large_run.py (cells with at least 200 genes, genes in at
least 3 cells, PCA with 50 components, 15 neighbors, Leiden at resolution 1, seed 0) with one gene
method and one normalization:

  genes          seurat (Scanpy's ranking, as published), seurat_floor (the same ranking among genes
                 detected in more than 1% of cells), scarf (Scarf's floor-trend method)
  normalization  log (to 10,000 per cell over all genes, log; as published), scarf (Scarf's: over the
                 selected genes to 1,000, log, genes standardized before PCA)

On some pipelines UMAP is run over a grid: start (spectral as published, kmeans = Scarf's start,
multilevel = Polariseq's multilevel spectral start through 1,000 k-means groups, leiden = the same
through the Leiden clusters) x min_dist (0.3, 0.5) x epochs (100, 200), plus the published setting
(spectral, 0.1, 200). Leiden is also run at resolution 0.5 and with seeds 1 and 2 (agreement with
seed 0), so that a difference between methods can be set against the run-to-run variation.

Measured, and written to OUT/summary.json as soon as each is known:
  every step's seconds and the peak memory the process held in it (examples/_resources.ResourceLog,
  sampled every 0.5 s); the genes chosen; clusters against the published labels (ARI, NMI, purity,
  types recovered); each UMAP's islands, split types, pixel purity and the correlation of
  type-to-type distances with the principal-component space.
OUT also holds each pipeline's genes and clusters and every layout (float16), so that every number can
be recomputed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))
sys.path.insert(0, str(HERE.parent / "examples"))

PIPELINES = [("seurat", "log"), ("seurat_floor", "log"), ("seurat_floor", "scarf"),
             ("scarf", "log"), ("scarf", "scarf")]
GRID = {"full": [("seurat", "log"), ("seurat_floor", "scarf"), ("scarf", "scarf")],
        "laptop": [("seurat", "log"), ("seurat_floor", "scarf"), ("scarf", "scarf")],
        "recipe": []}
STARTS = {"full": ["spectral", "kmeans", "multilevel", "leiden"],
          "laptop": ["spectral", "kmeans", "multilevel", "leiden"]}
MIN_DISTS = {"full": [0.3, 0.5], "laptop": [0.5]}
EPOCHS = {"full": [100, 200], "laptop": [100, 200]}


def global_vs_profiles(ref: np.ndarray, emb: np.ndarray, groups: np.ndarray) -> float:
    """Spearman correlation between the distances of the labelled types' centroids in the layout
    and the distances of their mean expression profiles (`ref`, types x genes, log-normalized)."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    present = np.unique(groups[groups >= 0])
    present = present[present < len(ref)]
    cen = np.array([emb[groups == g].mean(0) for g in present])
    return float(spearmanr(pdist(ref[present]), pdist(cen)).statistic)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mode", default="out-of-core", choices=["out-of-core", "memory"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ram-gb", type=float, default=100)
    ap.add_argument("--preset", default="full", choices=["full", "laptop", "recipe"],
                    help="recipe: only the published UMAP and the recommended one (multilevel, 0.5, 100)")
    ap.add_argument("--seed", type=int, default=0, help="seed of PCA, neighbors, Leiden, UMAP")
    ap.add_argument("--pipelines", default="", help="comma list of gene:norm to run (default: all)")
    a = ap.parse_args()

    import pandas as pd

    import polariseq as ps
    from _resources import ResourceLog
    from r3_layout_purity import purity
    from r3_scarf_ablation import scores
    from r3_umap_cause_4m import centroid_spearman
    from r3_umap_fragmentation import islands

    out = Path(a.out)
    (out / "layouts").mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.setdefault("pipelines", {})
    summary["setup"] = {"data": a.data, "mode": a.mode, "threads": a.threads, "ram_gb": a.ram_gb,
                        "preset": a.preset, "labels": a.labels, "seed": a.seed, "host": platform.node(),
                        "machine": platform.platform(), "python": platform.python_version()}

    def save():
        summary_path.write_text(json.dumps(summary, indent=1, default=float))

    ps.project(a.project)
    ps.set_budget(ram_gb=a.ram_gb, threads=a.threads)
    data = a.data if len(a.data) > 1 else a.data[0]
    pipes = PIPELINES
    if a.pipelines:
        wanted = [tuple(p.split(":")) for p in a.pipelines.split(",")]
        pipes = [p for p in PIPELINES if p in wanted]

    with ResourceLog(watch=a.project, verbose=True) as res:
        def step(name):
            return res.step(name)

        def last_step():
            s = res.steps[-1]
            return {"seconds": round(s["seconds"], 2), "peak_rss_gb": round(s["peak_rss_gb"], 3)}

        for genes, norm in pipes:
            key = f"{genes}+{norm}"
            if key in summary["pipelines"] and summary["pipelines"][key].get("done"):
                log("already done:", key)
                continue
            rec = summary["pipelines"].setdefault(key, {"steps": {}, "umap": {}})
            P = f"{key}|"
            with step(P + "open+qc"):
                adata = ps.read_h5ad(data, mode=a.mode)
                ps.pp.calculate_qc_metrics(adata)
                ps.pp.filter_cells(adata, min_genes=200)
                ps.pp.filter_genes(adata, min_cells=3)
            rec["steps"]["open+qc"] = last_step()
            # the published labels of the cells kept
            kind = a.labels.split(":")
            if kind[0] == "csv":
                tab = pd.read_csv(kind[1], usecols=["cell", kind[2]], dtype={kind[2]: "category"}).set_index("cell")
                lab = tab[kind[2]].reindex(np.asarray(adata.obs_names).astype(str)).cat.codes.to_numpy()
            elif kind[0] == "obs":
                lab = pd.Categorical(np.asarray(adata.obs[kind[1]]).astype(str)).codes.astype(np.int64)
            else:
                lab = None
            rec["cells"] = int(adata.n_obs)
            with step(P + "normalize"):
                ps.pp.normalize_total(adata, target_sum=1e4)
                ps.pp.log1p(adata)
            rec["steps"]["normalize"] = last_step()
            ref_path = out / "reference_profiles.npy"
            if lab is not None and not ref_path.exists():
                # the published types' mean expression over all genes (log-normalized): one shared
                # reference for the global structure of every layout, whatever its pipeline
                from polariseq._polariseq import group_means
                codes = np.where(lab >= 0, lab, np.iinfo(np.uint32).max).astype(np.uint32)
                with step(P + "reference profiles"):
                    np.save(ref_path, group_means(adata.X, codes, int(lab.max()) + 1))
            ref = np.load(ref_path) if ref_path.exists() else None
            with step(P + "hvg"):
                if genes == "scarf":
                    ps.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="scarf", subset=True)
                else:
                    floor = int(0.01 * adata.n_obs) if genes == "seurat_floor" else None
                    ps.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", min_cells=floor,
                                                subset=True)
            rec["steps"]["hvg"] = last_step()
            names = [str(g) for g in adata.var_names]
            (out / f"genes_{key}.txt").write_text("\n".join(names))
            if norm == "scarf":
                with step(P + "renormalize"):
                    ps.pp.renormalize(adata, target_sum=1000)
                rec["steps"]["renormalize"] = last_step()
            with step(P + "pca"):
                ps.pp.pca(adata, 50, seed=a.seed, scale=(norm == "scarf"))
            rec["steps"]["pca"] = last_step()
            with step(P + "neighbors"):
                ps.pp.neighbors(adata, n_neighbors=15, seed=a.seed)
            rec["steps"]["neighbors"] = last_step()
            pcs = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
            clus = {}
            for r, s in ((0.5, 0), (1.0, 1), (1.0, 2), (1.0, 0)):   # resolution 1, seed 0 last: obs['leiden']
                with step(P + f"leiden r{r} s{s}"):
                    ps.tl.leiden(adata, resolution=r, seed=s + a.seed)
                clus[(r, s)] = np.asarray(adata.obs["leiden"]).astype(np.int64)
                if (r, s) == (1.0, 0):
                    rec["steps"]["leiden"] = last_step()
            np.save(out / f"leiden_{key}.npy", clus[(1.0, 0)].astype(np.int32))
            from sklearn.metrics import adjusted_rand_score as ari
            rec["leiden_seed_agreement_ari"] = {"seed0_vs_1": round(ari(clus[(1.0, 0)], clus[(1.0, 1)]), 4),
                                                "seed0_vs_2": round(ari(clus[(1.0, 0)], clus[(1.0, 2)]), 4)}
            if lab is not None:
                ok = lab >= 0
                rec["vs_labels"] = {"res1": scores(clus[(1.0, 0)][ok], lab[ok]),
                                    "res0.5": scores(clus[(0.5, 0)][ok], lab[ok]),
                                    "cells_with_label": int(ok.sum())}
            groups = lab if lab is not None else clus[(1.0, 0)]
            save()
            log(key, rec.get("vs_labels", {}).get("res1"), rec["leiden_seed_agreement_ari"])

            settings = [("spectral", 0.1, 200)]
            if a.preset == "recipe":
                settings += [("multilevel", 0.5, 100)]
            elif (genes, norm) in GRID[a.preset]:
                settings += [(st, md, ep) for st in STARTS[a.preset] for md in MIN_DISTS[a.preset]
                             for ep in EPOCHS[a.preset]]
            else:
                settings += [("multilevel", 0.5, 200)]
            for st, md, ep in settings:
                name = f"{st} md{md} ep{ep}"
                if name in rec["umap"]:
                    continue
                with step(P + "umap " + name):
                    ps.tl.umap(adata, seed=a.seed, init=st, min_dist=md, n_epochs=ep)
                emb = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
                m = {"time": last_step()}
                ok = groups >= 0
                m.update(islands(emb[ok], groups[ok]))
                m["pixel_label_purity"] = round(purity(emb[ok], groups[ok]), 4)
                m["centroid_spearman_own_pcs"] = round(centroid_spearman(pcs[ok], emb[ok], groups[ok]), 4)
                if ref is not None:
                    m["global_spearman_vs_type_profiles"] = round(global_vs_profiles(ref, emb[ok], groups[ok]), 4)
                rec["umap"][name] = m
                np.save(out / "layouts" / f"{key}__{name.replace(' ', '_')}.npy", emb.astype(np.float16))
                save()
                log(key, name, {k: m.get(k) for k in ("islands", "types_mostly_in_one_island",
                                                       "global_spearman_vs_type_profiles")},
                    m["time"])
            rec["done"] = True
            save()
            del adata
        summary["resources"] = {k: v for k, v in res.summary().items() if k != "steps"}
        save()
    res.save(out)
    log("done")


if __name__ == "__main__":
    main()
