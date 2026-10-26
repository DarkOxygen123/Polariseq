"""Downstream analysis on Polariseq's output: Scanpy's PAGA on the full fetal atlas.

    .venv/bin/python benchmarks/r3_downstream.py

The path a user takes: Polariseq's out-of-core pipeline on the 4.06-million-
cell atlas (the benchmark parameters and a 6 GB budget), ``AnnData.to_anndata()``,
then ``scanpy.tl.paga`` on the exported neighbor graph and Leiden clusters.
Nothing is recomputed in Scanpy. Records the time of each stage, the process's
peak footprint (after each stage, so the peak of each stage and the stages
before it), and the PAGA graph with each cluster's majority published
type and organ. Output:
``benchmarks/results/r3/downstream/``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
OUT = ROOT / "benchmarks/results/r3/downstream"


def main() -> int:
    import pandas as pd
    import polariseq as ps
    import scanpy as sc

    import fetal_biology as fb
    import resource

    def peak() -> float:   # Linux compute node, peak resident memory so far (KiB -> GiB)
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)

    OUT.mkdir(parents=True, exist_ok=True)
    peaks = {}
    ps.set_budget(ram_gb=fb.BUDGET_GB, threads=fb.THREADS)
    t0 = time.perf_counter()
    a = ps.process_diskbacked(str(fb.FILES["4062k"]), **fb.PARAMS)
    t_pipe = time.perf_counter() - t0
    peaks["polariseq_pipeline"] = peak()

    t0 = time.perf_counter()
    ad = a.to_anndata()  # clusters and graph keys in Scanpy's conventions; no setup
    del a
    t_export = time.perf_counter() - t0
    peaks["to_anndata"] = peak()

    t0 = time.perf_counter()
    sc.tl.paga(ad, groups="leiden")
    t_paga = time.perf_counter() - t0
    peaks["scanpy_paga"] = peak()
    import matplotlib
    matplotlib.use("Agg")
    sc.pl.paga(ad, threshold=0.05, show=False)  # computes node positions (Fruchterman-Reingold)

    cells = pd.read_csv(fb.CELLS, index_col="cell", usecols=["cell", "Main_cluster_name", "Organ"])
    ann = cells.loc[np.asarray(ad.obs_names)]
    groups = []
    for g in ad.obs["leiden"].cat.categories:
        m = (ad.obs["leiden"] == g).to_numpy()
        t = ann["Main_cluster_name"][m].value_counts()
        o = ann["Organ"][m].value_counts()
        groups.append({"cluster": str(g), "cells": int(m.sum()), "type": str(t.index[0]),
                       "type_share": round(float(t.iloc[0] / m.sum()), 3), "organ": str(o.index[0]),
                       "organ_share": round(float(o.iloc[0] / m.sum()), 3)})
    conn = ad.uns["paga"]["connectivities"].toarray()
    np.savez_compressed(OUT / "paga.npz", connectivities=conn, pos=np.asarray(ad.uns["paga"]["pos"]))
    summary = {"n_cells": int(ad.n_obs), "n_clusters": len(groups),
               "seconds": {"polariseq_pipeline": round(t_pipe, 1), "to_anndata": round(t_export, 1),
                           "scanpy_paga": round(t_paga, 1)},
               "peak_footprint_gb_after": peaks,
               "memory_measure": "peak resident memory (Linux getrusage), compute node", "groups": groups}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
