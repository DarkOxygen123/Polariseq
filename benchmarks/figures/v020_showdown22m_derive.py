"""Source data of the 22-million-cell comparison figure from the showdown runs' per-cell outputs (2026-10-15).

    python benchmarks/figures/v020_showdown22m_derive.py <dir> [<out>]

<dir> is the comparison's results/scTab-22M folder (benchmarks/atlas, REPRODUCE.md): for polariseq and scarf
art/umap.npy, art/labels.npy and art/global_rows.npy; for bpcells art/labels.i32 and art/global_rows.f64 (raw
binaries); truth.npz (the published cell type of every input cell, by input row, from atlas/build_truth.py).

Writes to <out> (default v020_showdown_sctab22m/):
- sctab22m__<tool>__cell_types.npz  the UMAP of every cell, each pixel coloured by its majority published type
                                     (embedding_raster.raster; Polariseq and Scarf; BPCells has no UMAP)
- type_fates.csv                     per tool: published types recovered as a cluster of their own, kept together
                                     in a shared cluster, split across clusters (the rule used for the fetal atlas), and clusters
"""
from __future__ import annotations

import sys
from pathlib import Path

import embedding_raster as ER
import numpy as np
import pandas as pd

OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("v020_showdown_sctab22m")


def tool_arrays(d: Path, tool: str) -> tuple[np.ndarray, np.ndarray]:
    a = d / tool / "art"
    if tool == "bpcells":
        rows = np.fromfile(a / "global_rows.f64", dtype=np.float64).astype(np.int64)
        labels = np.fromfile(a / "labels.i32", dtype=np.int32)
    else:
        rows = np.load(a / "global_rows.npy").astype(np.int64)
        labels = np.load(a / "labels.npy").astype(np.int64)
    assert rows.shape == labels.shape, (tool, rows.shape, labels.shape)
    return rows, labels


def fates(types: np.ndarray, clusters: np.ndarray) -> tuple[int, int, int, int]:
    ok = types >= 0
    t, c = types[ok], clusters[ok]
    nt, nc = int(t.max()) + 1, int(c.max()) + 1
    tab = np.bincount(c * nt + t, minlength=nc * nt).reshape(nc, nt).astype(float)
    of_type = tab / np.maximum(tab.sum(axis=0, keepdims=True), 1)
    of_cluster = tab / np.maximum(tab.sum(axis=1, keepdims=True), 1)
    rec = shared = split = 0
    for j in range(nt):
        if tab[:, j].sum() == 0:
            continue
        k = int(of_type[:, j].argmax())
        if of_type[k, j] >= 0.5:
            if of_cluster[k, j] >= 0.5:
                rec += 1
            else:
                shared += 1
        else:
            split += 1
    return rec, shared, split, int(len(np.unique(c)))


def main() -> None:
    d = Path(sys.argv[1])
    truth = np.load(d / "truth.npz")
    label, names = truth["label"].astype(np.int64), truth["label_names"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for tool in ("polariseq", "scarf", "bpcells"):
        rows, clusters = tool_arrays(d, tool)
        types = label[rows]
        rec, shared, split, ncl = fates(types, clusters)
        rows_out.append({"tool": tool, "cells": len(rows), "recovered": rec, "kept_in_shared_cluster": shared,
                         "split": split, "clusters": ncl})
        print(tool, rows_out[-1])
        umap = d / tool / "art" / "umap.npy"
        if umap.exists():
            xy = np.load(umap)
            r = ER.raster(xy, types, bins=600)
            ER.save(OUT / f"sctab22m__{tool}__cell_types.npz", r, names, cells=len(rows))
            print("  raster written")
    pd.DataFrame(rows_out).to_csv(OUT / "type_fates.csv", index=False)


if __name__ == "__main__":
    main()
