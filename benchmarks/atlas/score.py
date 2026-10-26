#!/usr/bin/env python
"""Score one tool's run against the published labels, or compare the tools of one dataset.

    python score.py run --truth truth.npz --art ART_DIR --out score.json [--flags FLAGS.npz]
    python score.py compare --art polariseq=ART1 --art scarf=ART2 --art bpcells=ART3 --out compare.json \
                            [--ref-flags FLAGS.npz]

run      clusters against the labels (ARI, NMI, purity, labels recovered as a cluster of their own, rare
         labels recovered), the UMAP's islands, labels kept in one piece, pixel purity and global
         structure, and the doublet flags (number, share, clusters made mostly of flagged cells). The
         flags are the tool's own (Polariseq's qc_rows.npy, predicted_doublet.npy in ART_DIR) or the
         reference Scrublet's (--flags, for Scarf and BPCells).
compare  ARI between each pair of tools on the cells both analyzed, the overlap (Jaccard) of their
         selected genes, and with --ref-flags the agreement of Polariseq's doublet flags (Jaccard) and scores
         (Spearman) with the reference Scrublet's.

Definitions (as in the paper's analyses):
  recovered       one cluster holds at least half of the label's cells and is at least half that label
  rare            labels with fewer than 1,000 analyzed cells
  islands         connected groups of occupied pixels holding at least 0.05% of the cells, on a grid of
                  sqrt(n)/4 pixels (100 to 600) on the longer side, one-pixel gaps closed first
  in one piece    at least 80% of the label's cells in its largest island
  pixel purity    share of cells whose pixel (600 on the longer side) is mostly of their own label
  global          Spearman correlation between label-to-label distances of UMAP centroids and of the
                  labels' mean expression (truth.npz profiles, the 2,000 genes most variable across labels)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_art(art: Path) -> dict:
    out = {}
    if (art / "global_rows.npy").exists():
        out["rows"] = np.load(art / "global_rows.npy").astype(np.int64)
    else:
        out["rows"] = np.fromfile(art / "global_rows.f64", dtype=np.float64).astype(np.int64)
    if (art / "labels.npy").exists():
        out["labels"] = np.load(art / "labels.npy").astype(np.int64)
    else:
        out["labels"] = np.fromfile(art / "labels.i32", dtype=np.int32).astype(np.int64)
    if (art / "umap.npy").exists():
        out["umap"] = np.load(art / "umap.npy")
    if (art / "hvg.txt").exists():
        out["hvg"] = set((art / "hvg.txt").read_text().split())
    return out


def clusters_vs_labels(clusters: np.ndarray, types: np.ndarray, rare_below: int = 1000) -> dict:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    t = pd.crosstab(clusters, types)
    share_of_type = t / t.sum(axis=0)
    share_of_cluster = t.div(t.sum(axis=1), axis=0)
    size = t.sum(axis=0)
    rec = [c for c in t.columns if share_of_type[c].max() >= 0.5
           and share_of_cluster.loc[share_of_type[c].idxmax(), c] >= 0.5]
    rare = [c for c in t.columns if size[c] < rare_below]
    return {"clusters": int(t.shape[0]), "labels_present": int(t.shape[1]),
            "ari": round(float(adjusted_rand_score(types, clusters)), 4),
            "nmi": round(float(normalized_mutual_info_score(types, clusters)), 4),
            "purity": round(float(t.max(axis=1).sum() / len(clusters)), 4),
            "labels_recovered": len(rec), "rare_labels": len(rare),
            "rare_labels_recovered": len(set(rec) & set(rare))}


def raster(coords: np.ndarray, bins: int):
    lo, hi = np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    lo, hi = lo - 0.01 * span, hi + 0.01 * span
    w, h = hi - lo
    if w >= h:
        nx, ny = bins, max(1, int(round(bins * h / w)))
    else:
        nx, ny = max(1, int(round(bins * w / h))), bins
    ix = np.clip(((coords[:, 0] - lo[0]) / w * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip(((coords[:, 1] - lo[1]) / h * ny).astype(np.int64), 0, ny - 1)
    return iy * nx + ix, (nx, ny)


def umap_scores(emb: np.ndarray, types: np.ndarray, profiles: np.ndarray | None) -> dict:
    from scipy import ndimage
    from scipy.stats import spearmanr
    emb = np.asarray(emb, dtype=np.float64)
    bins = int(np.clip(np.sqrt(len(emb)) / 4, 100, 600))
    pix, (nx, ny) = raster(emb, bins)
    occ = np.bincount(pix, minlength=nx * ny).reshape(ny, nx) > 0
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)))
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    isl = lab.ravel()[pix]
    counts = np.bincount(isl, minlength=n + 1)[1:]
    present = np.unique(types)
    whole = np.array([np.bincount(isl[types == g]).max() / (types == g).sum() for g in present])
    pix6, _ = raster(emb, 600)
    k = int(types.max()) + 1
    key, cnt = np.unique(pix6 * k + types, return_counts=True)
    p = key // k
    o = np.lexsort((-cnt, p))
    first = np.r_[True, p[o][1:] != p[o][:-1]]
    out = {"islands": int((counts >= 0.0005 * len(emb)).sum()),
           "labels_in_one_piece": int((whole >= 0.8).sum()), "labels": int(len(present)),
           "pixel_purity": round(float(cnt[o][first].sum() / len(types)), 4)}
    if profiles is not None and len(present) > 3:
        cen = np.array([emb[types == g].mean(axis=0) for g in present])
        prof = profiles[present]
        top = np.argsort(prof.var(axis=0))[::-1][:2000]
        prof = prof[:, top]
        iu = np.triu_indices(len(present), 1)
        du = np.linalg.norm(cen[:, None] - cen[None], axis=2)[iu]
        de = np.linalg.norm(prof[:, None] - prof[None], axis=2)[iu]
        out["global_structure_spearman"] = round(float(spearmanr(du, de).correlation), 4)
    return out


def doublet_scores(rows: np.ndarray, clusters: np.ndarray, frows: np.ndarray, fflag: np.ndarray) -> dict:
    flagged = {"flagged": int(fflag.sum()), "cells_flag_checked": int(len(fflag)),
               "flagged_share": round(float(fflag.mean()), 4) if len(fflag) else None}
    o = np.argsort(frows)
    frows, fflag = frows[o], fflag[o]
    pos = np.clip(np.searchsorted(frows, rows), 0, len(frows) - 1)
    inflag = (frows[pos] == rows) & fflag[pos]
    share = np.bincount(clusters, weights=inflag) / np.maximum(np.bincount(clusters), 1)
    flagged.update(flagged_among_analyzed=int(inflag.sum()),
                   clusters_mostly_flagged=int((share >= 0.5).sum()),
                   max_cluster_flagged_share=round(float(share.max()), 4))
    return flagged


def run(a) -> dict:
    t = np.load(a.truth, allow_pickle=False)
    art = load_art(Path(a.art))
    types = t["label"][art["rows"]].astype(np.int64)
    res = {"cells_analyzed": int(len(types)), "clusters_vs_labels": clusters_vs_labels(art["labels"], types)}
    if "umap" in art:
        prof = t["profiles"] if "profiles" in t.files else None
        res["umap"] = umap_scores(art["umap"], types, prof)
    fl = None
    if a.flags:
        f = np.load(a.flags)
        fl = (f["global_row"].astype(np.int64), f["predicted"].astype(bool))
    elif (Path(a.art) / "predicted_doublet.npy").exists():
        fl = (np.load(Path(a.art) / "qc_rows.npy").astype(np.int64),
              np.load(Path(a.art) / "predicted_doublet.npy").astype(bool))
    if fl is not None:
        res["doublets"] = doublet_scores(art["rows"], art["labels"], *fl)
    return res


def doublet_agreement(art: Path, ref_flags: str) -> dict:
    """Polariseq's flags and scores against the reference Scrublet's, on the cells both scored."""
    from scipy.stats import spearmanr
    rows = np.load(art / "qc_rows.npy").astype(np.int64)
    sc = np.load(art / "doublet_score.npy")
    fl = np.load(art / "predicted_doublet.npy").astype(bool)
    f = np.load(ref_flags)
    common, i, j = np.intersect1d(rows, f["global_row"].astype(np.int64), return_indices=True)
    a, b = fl[i], f["predicted"].astype(bool)[j]
    x, y = sc[i], f["score"][j]
    ok = np.isfinite(x) & np.isfinite(y)
    union = int((a | b).sum())
    return {"cells_in_both": int(len(common)), "flagged_polariseq": int(a.sum()), "flagged_reference": int(b.sum()),
            "jaccard": round(int((a & b).sum()) / union, 4) if union else None,
            "score_spearman": round(float(spearmanr(x[ok], y[ok]).correlation), 4) if ok.sum() > 2 else None}


def compare(a) -> dict:
    from sklearn.metrics import adjusted_rand_score
    arts = {}
    for spec in a.art:
        name, path = spec.split("=", 1)
        arts[name] = load_art(Path(path))
    names = list(arts)
    out = {}
    if a.ref_flags and "polariseq" in arts:
        out["polariseq doublets vs reference Scrublet"] = doublet_agreement(Path(dict(
            x.split("=", 1) for x in a.art)["polariseq"]), a.ref_flags)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = arts[names[i]], arts[names[j]]
            common, ix, iy = np.intersect1d(x["rows"], y["rows"], return_indices=True)
            pair = f"{names[i]} vs {names[j]}"
            out[pair] = {"cells_in_both": int(len(common)),
                         "ari": round(float(adjusted_rand_score(x["labels"][ix], y["labels"][iy])), 4)}
            if "hvg" in x and "hvg" in y:
                out[pair]["hvg_jaccard"] = round(len(x["hvg"] & y["hvg"]) / len(x["hvg"] | y["hvg"]), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--truth")
    ap.add_argument("--art", action="append", required=True)
    ap.add_argument("--flags")
    ap.add_argument("--ref-flags", help="compare: the reference Scrublet's flags, to compare Polariseq's with")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "run":
        a.art = a.art[0]
        res = run(a)
    else:
        res = compare(a)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
