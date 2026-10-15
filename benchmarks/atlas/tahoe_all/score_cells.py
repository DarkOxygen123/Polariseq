#!/usr/bin/env python
"""Score the full Tahoe-100M Polariseq run from its per-cell table, and draw the cell-line x cluster heatmap.

    python score_cells.py --cells figures/cells.tsv.gz --out scores_dir [--check]

Reads only the columns leiden, cell_line_id, plate, umap1, umap2 of cells.tsv.gz, in chunks, into compact
arrays (about 15 bytes per cell). Writes into scores_dir:

  scores.json                  clusters vs cell lines (ARI, NMI, purity, lines recovered) and the UMAP scores
  line_by_cluster.csv          cells per cell line (rows) and cluster (columns)
  plate_by_cluster.csv         cells per plate and cluster
  heatmap_line_by_cluster.png  each line's share of cells in each cluster (rows sum to 1)

Definitions are those of ../node/score.py (recovered: one cluster holds at least half of the line's cells and
is at least half that line). ARI, NMI and purity are computed from the contingency table, so they need no
per-cell arrays; --check also runs score.py's own functions on the per-cell arrays and fails if any number
differs (use it on small inputs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "node"))
import score  # noqa: E402  (score.py of the showdown, for the shared definitions)

COLS = ["leiden", "cell_line_id", "plate", "umap1", "umap2"]
RAMP = ["#ffffff", "#f4aebd", "#c4516f", "#5e1c30"]  # paper's kNN graph/Leiden colour, light to dark


def read_cells(path: Path, chunk: int) -> dict:
    lines: dict[str, int] = {}
    plates: dict[str, int] = {}
    parts = {k: [] for k in ("cluster", "line", "plate", "umap")}
    dropped = 0
    for df in pd.read_csv(path, sep="\t", usecols=COLS, chunksize=chunk,
                          dtype={"cell_line_id": str, "plate": str, "umap1": np.float32, "umap2": np.float32}):
        keep = df["leiden"].notna() & df["cell_line_id"].notna()
        dropped += int((~keep).sum())
        df = df[keep]
        for name in df["cell_line_id"].unique():
            lines.setdefault(name, len(lines))
        for name in df["plate"].fillna("NA").unique():
            plates.setdefault(name, len(plates))
        parts["cluster"].append(df["leiden"].to_numpy().astype(np.int32))
        parts["line"].append(df["cell_line_id"].map(lines).to_numpy().astype(np.int16))
        parts["plate"].append(df["plate"].fillna("NA").map(plates).to_numpy().astype(np.int16))
        parts["umap"].append(df[["umap1", "umap2"]].to_numpy(dtype=np.float32))
        print(f"  read {sum(len(p) for p in parts['cluster']):,} cells", flush=True)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["line_names"] = np.array(sorted(lines, key=lines.get))
    out["plate_names"] = np.array(sorted(plates, key=plates.get))
    out["dropped"] = dropped
    return out


def table(rows: np.ndarray, cols: np.ndarray, nr: int, nc: int) -> np.ndarray:
    return np.bincount(rows.astype(np.int64) * nc + cols, minlength=nr * nc).reshape(nr, nc)


def from_contingency(t: np.ndarray) -> dict:
    """t: labels (rows) x clusters (columns) cell counts."""
    t = t[:, t.sum(axis=0) > 0]                                  # lines always have cells; drop unused ids
    n = int(t.sum())
    pair = lambda x: (x.astype(np.float64) * (x - 1) / 2).sum()
    a, b, ij, all_pairs = pair(t.sum(axis=1)), pair(t.sum(axis=0)), pair(t), n * (n - 1) / 2
    expected = a * b / all_pairs
    ari = (ij - expected) / ((a + b) / 2 - expected)
    p = t / n
    pl, pc = p.sum(axis=1), p.sum(axis=0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pl, pc)[nz])).sum()
    h = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    nmi = mi / ((h(pl) + h(pc)) / 2)
    share_of_line = t / t.sum(axis=1, keepdims=True)
    share_of_cluster = t / t.sum(axis=0, keepdims=True)
    best = share_of_line.argmax(axis=1)
    rec = (share_of_line.max(axis=1) >= 0.5) & (share_of_cluster[np.arange(len(t)), best] >= 0.5)
    rare = t.sum(axis=1) < 1000
    return {"clusters": int(t.shape[1]), "labels_present": int(t.shape[0]),
            "ari": round(float(ari), 4), "nmi": round(float(nmi), 4),
            "purity": round(float(t.max(axis=0).sum() / n), 4),
            "labels_recovered": int(rec.sum()), "rare_labels": int(rare.sum()),
            "rare_labels_recovered": int((rec & rare).sum())}, rec


def heatmap(t: np.ndarray, line_names: np.ndarray, recovered: np.ndarray, png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    t = t[:, t.sum(axis=0) > 0]
    share = t / t.sum(axis=1, keepdims=True)
    rows = np.argsort(-t.sum(axis=1), kind="stable")             # lines, largest first
    rank = np.empty(len(rows), dtype=np.int64)
    rank[rows] = np.arange(len(rows))
    owner = rank[share.argmax(axis=0)]                            # each cluster goes to the line it holds most of
    cols = np.lexsort((-t.sum(axis=0), owner))                    # grouped by that line, larger clusters first
    m = share[rows][:, cols]
    fig_w = min(24, 4 + 0.07 * m.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, 2 + 0.17 * m.shape[0]), dpi=200)
    im = ax.imshow(m, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                   cmap=LinearSegmentedColormap.from_list("leiden", RAMP))
    labels = [f"{line_names[r]}" + ("" if recovered[r] else "  ×") for r in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=6)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color("#222222" if recovered[r] else "#8a8a8a")
    ax.set_xticks([])
    ax.set_xlabel(f"{m.shape[1]} clusters, grouped by the cell line each holds most of", fontsize=8)
    ax.set_ylabel("cell line (× = not recovered as a cluster of its own)", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("share of the line's cells", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=5_000_000)
    ap.add_argument("--title", default="Tahoe-100M, Polariseq: cell lines across clusters")
    ap.add_argument("--check", action="store_true", help="also run score.py on the per-cell arrays")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    c = read_cells(a.cells, a.chunk)
    print(f"read {len(c['cluster']):,} cells ({c['dropped']:,} without cluster or line) in "
          f"{time.time() - t0:.0f} s", flush=True)
    ncl = int(c["cluster"].max()) + 1
    lt = table(c["line"], c["cluster"], len(c["line_names"]), ncl)
    pt = table(c["plate"], c["cluster"], len(c["plate_names"]), ncl)
    res, recovered = from_contingency(lt)
    out = {"cells": int(len(c["cluster"])), "cells_dropped": c["dropped"], "clusters_vs_lines": res}
    t1 = time.time()
    out["umap"] = score.umap_scores(c["umap"], c["line"].astype(np.int64), None)
    print(f"umap scores in {time.time() - t1:.0f} s", flush=True)
    plate_share = pt / np.maximum(pt.sum(axis=0, keepdims=True), 1)
    out["clusters_mostly_one_plate"] = {"share_at_least_0.5": int((plate_share.max(axis=0) >= 0.5).sum()),
                                        "share_at_least_0.9": int((plate_share.max(axis=0) >= 0.9).sum())}
    out["lines_not_recovered"] = sorted(c["line_names"][~recovered].tolist())
    if a.check:
        ref = score.clusters_vs_labels(c["cluster"], c["line"])
        diff = {k: (res[k], ref[k]) for k in ref if res[k] != ref[k]}
        out["check_against_score_py"] = "equal" if not diff else diff
        print("check against score.py:", out["check_against_score_py"], flush=True)
    clusters = [str(i) for i in range(ncl)]
    pd.DataFrame(lt, index=c["line_names"], columns=clusters).to_csv(a.out / "line_by_cluster.csv")
    pd.DataFrame(pt, index=c["plate_names"], columns=clusters).to_csv(a.out / "plate_by_cluster.csv")
    heatmap(lt, c["line_names"], recovered, a.out / "heatmap_line_by_cluster.png", a.title)
    out["seconds"] = round(time.time() - t0, 1)
    (a.out / "scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 1 if a.check and out["check_against_score_py"] != "equal" else 0


if __name__ == "__main__":
    sys.exit(main())
