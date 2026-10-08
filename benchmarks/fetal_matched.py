"""The fetal atlas analysed the way its authors analysed it.

Two comparisons whose settings follow the original analysis (Cao et al.
2020, Science 370, eaba7721, Methods), so that a difference between the
published result and Polariseq's reflects the software, not the choices:

  embed    The authors' whole-atlas embedding ("Clustering analysis of cells
           across organs"): 5,000 cells per cell type per organ, or all of a
           type's cells where fewer; PCA with 50 components on their
           cell-type-specific marker genes (q = 0); UMAP with 50 neighbors,
           minimum distance 0.1 and cosine distance. Cells are normalized to
           the geometric mean of library sizes and log-transformed, and genes
           are scaled to unit variance, as Monocle 3 does by default. PCA,
           the neighbor graph and UMAP run in Polariseq.

  split    One file per organ. Each organ is a contiguous block of rows in
           the atlas file, so a split is one ordered pass over it.
  organs   The authors' clustering was done organ by organ. Each organ runs
           through the same out-of-core pipeline and settings as the global
           analysis (fetal_biology.PARAMS), with the same streamed marker
           statistics.
  compare  The organs' clusters are combined and scored against the
           published cell types by the same rule as the global analysis
           (fetal_biology.run_compare).

    python benchmarks/fetal_matched.py embed
    python benchmarks/fetal_matched.py split
    python benchmarks/fetal_matched.py organs
    python benchmarks/fetal_matched.py compare
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import fetal_biology as fb  # noqa: E402

DATA = ROOT / "benchmarks/results/fetal_data"
FULL = DATA / "fetal4062k.h5ad"
CELLS = DATA / "fetal_cells.csv.gz"
DE = DATA / "GSE156793_S8_DE_gene_cells.csv.gz"
OUT = ROOT / "benchmarks/results/fetal"
EMBED = OUT / "matched_embedding"
ORGANS = OUT / "4062k_by_organ"
ORGAN_FILES = DATA / "by_organ"

PER_GROUP = 5000          # cells per cell type per organ, as published
N_PCS = 50
N_NEIGHBORS = 50
MIN_DIST = None          # Polariseq's default UMAP (multilevel start, 0.5, 100 rounds)
SEED = 0
WINDOW = 20_000           # source rows per contiguous read


def _source():
    """Row offsets and the file's cell and gene tables, without the matrix."""
    from anndata.io import read_elem
    with h5py.File(FULL, "r") as f:
        indptr = f["X/indptr"][:]
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
    return indptr, obs, var


def _rows(indptr, rows: np.ndarray, cols: np.ndarray | None, n_genes: int):
    """Selected rows (sorted file indices) as CSR, optionally restricted to
    `cols`, plus every selected row's total over all genes. One contiguous
    read per window, so each compressed chunk is decompressed once."""
    blocks, totals = [], []
    with h5py.File(FULL, "r") as f:
        data_ds, idx_ds = f["X/data"], f["X/indices"]
        n = len(indptr) - 1
        for w0 in range(0, n, WINDOW):
            w1 = min(w0 + WINDOW, n)
            sel = rows[(rows >= w0) & (rows < w1)]
            if not len(sel):
                continue
            a, b = int(indptr[w0]), int(indptr[w1])
            block = sp.csr_matrix((data_ds[a:b], idx_ds[a:b], indptr[w0:w1 + 1] - a),
                                  shape=(w1 - w0, n_genes))[sel - w0]
            totals.append(np.asarray(block.sum(axis=1)).ravel())
            blocks.append(block[:, cols] if cols is not None else block)
    return sp.vstack(blocks, format="csr"), np.concatenate(totals)


# ---------------------------------------------------------------- embed
def matched_sample():
    """The authors' balanced sample, z-scored on their marker genes, as a
    sparse matrix: (x, meta of the sampled rows, sorted file rows, number of
    type-organ groups, marker-gene count, read seconds)."""
    indptr, obs, var = _source()
    cells = pd.read_csv(CELLS, index_col="cell",
                        usecols=["cell", "Organ", "Main_cluster_name",
                                 "Global_umap_1", "Global_umap_2"])
    meta = cells.reindex(obs.index)

    # The published sample: 5,000 cells per cell type per organ.
    rng = np.random.default_rng(SEED)
    pos = np.arange(len(meta))
    picked = []
    for _, idx in meta.groupby(["Main_cluster_name", "Organ"]).indices.items():
        idx = pos[idx]
        picked.append(idx if len(idx) <= PER_GROUP
                      else rng.choice(idx, PER_GROUP, replace=False))
    rows = np.sort(np.concatenate(picked))

    # The published feature set: cell-type-specific markers with q = 0.
    de = pd.read_csv(DE, usecols=["gene_id", "qval"])
    marker_ids = set(de.loc[de.qval == 0, "gene_id"])
    cols = np.flatnonzero(var["gene_id"].isin(marker_ids).to_numpy())
    print(f"[embed] {len(rows):,} cells from {len(picked)} type-organ groups; "
          f"{len(cols):,} marker genes")

    t0 = time.perf_counter()
    x, totals = _rows(indptr, rows, cols, var.shape[0])
    t_read = time.perf_counter() - t0

    # Monocle 3's defaults: size factors against the geometric mean library
    # size, log(x + 1), genes scaled to unit variance. Scaling the columns of
    # the sparse matrix and letting PCA centre it is exactly PCA on z-scores,
    # without ever densifying the matrix.
    gm = float(np.exp(np.mean(np.log(np.maximum(totals, 1)))))
    x = sp.diags((gm / np.maximum(totals, 1)).astype(np.float32)) @ x
    x.data = np.log1p(x.data).astype(np.float32)
    mean = np.asarray(x.mean(axis=0)).ravel()
    sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(sq - mean ** 2, 1e-12))
    x = (x @ sp.diags((1.0 / sd).astype(np.float32))).tocsr().astype(np.float32)
    return x, meta.iloc[rows], rows, len(picked), len(cols), t_read


def embed() -> None:
    import polariseq as ps

    t_all = time.perf_counter()
    x, sub, rows, n_groups, n_cols, t_read = matched_sample()
    ps.set_budget(ram_gb=6, threads=4)
    adata = ps.AnnData(X=x)
    t0 = time.perf_counter()
    ps.pp.pca(adata, N_PCS, seed=SEED)
    t_pca = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, metric="cosine", seed=SEED)
    t_nn = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps.tl.umap(adata, min_dist=MIN_DIST, seed=SEED)
    t_umap = time.perf_counter() - t0

    EMBED.mkdir(parents=True, exist_ok=True)
    np.save(EMBED / "umap.npy", np.asarray(adata.obsm["X_umap"], dtype=np.float32))
    sub.to_csv(EMBED / "cells.csv.gz")
    summary = {
        "procedure": "Cao et al. 2020, Methods, 'Clustering analysis of cells across organs'",
        "cells": int(len(rows)), "groups": n_groups, "per_group": PER_GROUP,
        "marker_genes": int(n_cols), "marker_rule": "S8 cell-type-specific genes, q = 0",
        "normalization": "size factor to geometric-mean library size, log1p, genes scaled",
        "n_pcs": N_PCS, "n_neighbors": N_NEIGHBORS, "metric": "cosine",
        "min_dist": MIN_DIST, "seed": SEED,
        "timing_s": {"read": round(t_read, 1), "pca": round(t_pca, 1),
                     "neighbors": round(t_nn, 1), "umap": round(t_umap, 1),
                     "total": round(time.perf_counter() - t_all, 1)},
    }
    (EMBED / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- split
def split() -> None:
    import anndata
    anndata.settings.allow_write_nullable_strings = True   # anndata 0.13 on the cluster
    from anndata.io import sparse_dataset
    import fetal_prepare as fp

    indptr, obs, var = _source()
    organ = pd.read_csv(CELLS, index_col="cell", usecols=["cell", "Organ"]) \
        .Organ.reindex(obs.index).to_numpy()
    edges = np.flatnonzero(organ[1:] != organ[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(organ)]
    ORGAN_FILES.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with h5py.File(FULL, "r") as src:
        data_ds, idx_ds = src["X/data"], src["X/indices"]
        for lo, hi in zip(starts, ends):
            name = str(organ[lo])
            path = ORGAN_FILES / f"fetal_{name.lower()}.h5ad"
            t0 = time.perf_counter()
            tmp = path.with_suffix(".h5ad.partial")
            with h5py.File(tmp, "w") as out:
                X = None
                for w0 in range(lo, hi, WINDOW):
                    w1 = min(w0 + WINDOW, hi)
                    a, b = int(indptr[w0]), int(indptr[w1])
                    block = sp.csr_matrix((data_ds[a:b], idx_ds[a:b].astype(np.int32),
                                           indptr[w0:w1 + 1] - a),
                                          shape=(w1 - w0, var.shape[0]))
                    if X is None:
                        fp._write_empty_anndata(out, block)
                        X = sparse_dataset(out["X"])
                    else:
                        X.append(block)
                fp._finish_anndata(out, list(obs.index[lo:hi]), var,
                                   {"source": FULL.name, "organ": name,
                                    "rows": [int(lo), int(hi)]})
            os.replace(tmp, path)
            manifest[name] = {"path": str(path.relative_to(ROOT)), "cells": int(hi - lo),
                              "rows": [int(lo), int(hi)],
                              "build_s": round(time.perf_counter() - t0, 1)}
            print(f"[split] {name:<11} {hi - lo:>9,} cells  {manifest[name]['build_s']:>6.1f} s")
    (ORGAN_FILES / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------- organs
def organs() -> None:
    import anndata as ad
    import polariseq as ps
    from polariseq._polariseq import streaming_cluster_stats_normalized

    manifest = json.loads((ORGAN_FILES / "manifest.json").read_text())
    ps.set_budget(ram_gb=6, threads=4)
    spill = Path(ps.budget().spill_dir)
    ORGANS.mkdir(parents=True, exist_ok=True)
    for name, m in sorted(manifest.items(), key=lambda kv: -kv[1]["cells"]):
        out = ORGANS / name.lower()
        if (out / "labels.npy").exists():
            print(f"[organs] {name}: done already")
            continue
        out.mkdir(parents=True, exist_ok=True)
        path = ROOT / m["path"]
        t0 = time.perf_counter()
        adata = ps.process_diskbacked(str(path), **fb.PARAMS)
        t_pipe = time.perf_counter() - t0
        labels = np.asarray(adata.obs["leiden"]).astype(np.uint32)
        k = int(labels.max()) + 1
        names = np.asarray(adata.obs_names, dtype=object)
        backed = ad.read_h5ad(str(path), backed="r")
        full = pd.Index(np.asarray(backed.obs_names, dtype=object))
        var_names = [str(v) for v in backed.var_names]
        backed.file.close()
        where = full.get_indexer(pd.Index(names))
        labels_full = np.full(len(full), k, dtype=np.uint32)
        labels_full[where] = labels
        t0 = time.perf_counter()
        stats = streaming_cluster_stats_normalized(str(path), labels_full, k,
                                                   fb.PARAMS["target_sum"])
        t_stats = time.perf_counter() - t0
        fb.welch_markers(stats, var_names, n_top=50).to_csv(out / "markers.csv", index=False)
        np.save(out / "labels.npy", labels)
        np.savetxt(out / "obs_names.txt", names, fmt="%s")
        (out / "summary.json").write_text(json.dumps({
            "organ": name, "cells_file": int(len(full)), "cells_kept": int(len(names)),
            "clusters": k, "params": fb.PARAMS,
            "timing_s": {"pipeline": round(t_pipe, 1), "streaming_stats": round(t_stats, 1)},
        }, indent=1))
        print(f"[organs] {name:<11} {len(names):>9,} cells, {k:>3} clusters, "
              f"pipeline {t_pipe:.0f}s, stats {t_stats:.0f}s")
        shutil.rmtree(spill, ignore_errors=True)      # ~10 GB per large organ


# ---------------------------------------------------------------- compare
def compare() -> None:
    """Organ clusters become one labelling (organ-prefixed ids), scored with
    exactly the global analysis's rule."""
    labels, names, markers, offset = [], [], [], 0
    for d in sorted(p for p in ORGANS.iterdir() if (p / "labels.npy").exists()):
        lab = np.load(d / "labels.npy").astype(np.int64)
        labels.append(lab + offset)
        names.append(np.loadtxt(d / "obs_names.txt", dtype=object))
        mk = pd.read_csv(d / "markers.csv")
        mk["cluster"] = mk["cluster"] + offset
        markers.append(mk)
        offset += int(lab.max()) + 1
    np.save(ORGANS / "labels.npy", np.concatenate(labels))
    np.savetxt(ORGANS / "obs_names.txt", np.concatenate(names), fmt="%s")
    pd.concat(markers).to_csv(ORGANS / "markers.csv", index=False)
    fb.run_compare("4062k", None, out=ORGANS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["embed", "split", "organs", "compare"])
    {"embed": embed, "split": split, "organs": organs, "compare": compare}[
        ap.parse_args().phase]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
