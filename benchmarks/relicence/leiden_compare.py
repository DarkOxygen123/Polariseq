#!/usr/bin/env python
"""Leiden (and Louvain) of Polariseq 0.1.x against the replacement, on the same graphs.

    python leiden_compare.py graph DATASET --out G.npz           # the default pipeline up to the graph
    python leiden_compare.py run G.npz --out R.json [--seeds 10] [--arms new,igraph] [--louvain]

graph   builds the fuzzy connectivity graph once with the current code (normalization, genes, PCA and
        neighbours are not affected by the replacement) and saves it with the published labels, if the
        dataset has any. DATASET is a name below or a path to an .h5ad.
run     clusters that graph with every arm and seed. Arms: `old` (0.1.x), `oldfix` (0.1.x with the
        aggregation fixed), `new` (the replacement), `igraph` (python-igraph's Leiden, the reference
        Scanpy uses), selected through POLARISEQ_LEIDEN_IMPL. Per arm: clusters, modularity (one formula
        for every arm), seconds, ARI and recovered labels against the published labels; and the ARI of
        every run against every `old` run, next to `old`'s own seed-to-seed ARI, when `old` ran.

The arms `old` and `oldfix` ran on the development build that held both implementations behind the
POLARISEQ_LEIDEN_IMPL switch; the release has only the new Leiden, so with it the arms are `new` and
`igraph`, and asking for `old` or `oldfix` stops with this explanation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "benchmarks/results/validation_data"
F = ROOT / "benchmarks/results/fetal_data"
K = ROOT / "examples/data/kang_2018"
# name -> (files, label source, PCA components, resolution, Harmony on). The settings are the paper's:
# example 06's defaults for the atlases, the comparison of conditions for Kang.
DATASETS = {
    "pbmc3k": (V / "pbmc3k.h5ad", None, 50, 1.0, False),
    "pbmc10k": (V / "pbmc10k.h5ad", None, 50, 1.0, False),
    "pbmc20k": (V / "pbmc20k.h5ad", None, 50, 1.0, False),
    "neuron10k": (V / "neuron10k.h5ad", None, 50, 1.0, False),
    "brain50k": (V / "brain50k.h5ad", None, 50, 1.0, False),
    "brain100k": (V / "brain100k.h5ad", None, 50, 1.0, False),
    "brain1306k": (V / "brain1306k.h5ad", None, 50, 1.0, False),
    "fetal60k": (F / "fetal60k.h5ad", "fetal", 50, 1.0, False),
    "fetal4062k": (F / "fetal4062k.h5ad", "fetal", 50, 1.0, False),
    "kang": ({"control": K / "control.h5ad", "IFN-beta": K / "ifn_beta.h5ad"}, "published_type", 30, 0.6, True),
}


def build_graph(a) -> int:
    import polariseq as ps
    files, label, n_pcs, resolution, harmony = DATASETS.get(a.dataset, (Path(a.dataset), a.label, 50, 1.0, False))
    ps.set_budget(ram_gb=a.ram_gb, threads=a.threads)
    ps.set_workflow("polariseq")
    if a.project:
        ps.project(a.project)
    t0 = time.perf_counter()
    if isinstance(files, dict):
        adata = ps.read_h5ad({k: str(v) for k, v in files.items()}, batch_key="condition")
    else:
        adata = ps.read_h5ad(str(files))
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=200)
    ps.pp.filter_genes(adata, min_cells=3)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    if harmony:   # the Kang analysis: Scanpy's steps, as in benchmarks/r3_kang.py
        ps.pp.highly_variable_genes(adata, n_top_genes=2000)
        ps.pp.pca(adata, n_components=n_pcs)
        ps.pp.harmony_integrate(adata, batch_key="condition")
        ps.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_harmony")
    else:         # example 06's defaults under the polariseq workflow
        ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        ps.pp.renormalize(adata, target_sum=1000)
        ps.pp.pca(adata, n_components=n_pcs, seed=0, scale=True)
        ps.pp.neighbors(adata, n_neighbors=15, seed=0)
    conn = adata.obsp["connectivities"]
    if not conn.has_sorted_indices:
        conn.sort_indices()
    out = {"indptr": np.asarray(conn.indptr, dtype=np.int64), "indices": np.asarray(conn.indices, dtype=np.uint32),
           "data": np.asarray(conn.data, dtype=np.float32), "resolution": resolution}
    if label == "fetal":
        import pandas as pd
        meta = pd.read_csv(F / "fetal_cells.csv.gz", usecols=["cell", "Main_cluster_name"]).set_index("cell")
        names = np.asarray(adata.obs["cell"] if "cell" in adata.obs else adata.obs_names).astype(str)
        out["labels"] = meta["Main_cluster_name"].reindex(names).fillna("NA").to_numpy().astype(str)
    elif label:
        out["labels"] = np.asarray(adata.obs[label]).astype(str)
    np.savez(a.out, **out, meta=json.dumps({"dataset": a.dataset, "cells": int(adata.n_obs), "n_pcs": n_pcs,
                                            "harmony": harmony, "seconds": round(time.perf_counter() - t0, 1),
                                            "polariseq": ps.__version__}))
    print(f"{a.dataset}: {adata.n_obs:,} cells, {conn.nnz:,} graph entries -> {a.out}")
    return 0


def modularity(indptr, indices, data, labels, resolution) -> float:
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    two_m = float(data.astype(np.float64).sum())
    same = labels[rows] == labels[indices]
    internal = float(data[same].astype(np.float64).sum())
    k = np.bincount(labels, weights=np.bincount(rows, weights=data.astype(np.float64), minlength=len(labels)))
    return internal / two_m - resolution * float(((k / two_m) ** 2).sum())


def recovered(labels: np.ndarray, truth: np.ndarray) -> int:
    import pandas as pd
    t = pd.crosstab(labels, truth)
    share_of_type, share_of_cluster = t / t.sum(axis=0), t.div(t.sum(axis=1), axis=0)
    return int(sum(share_of_type[c].max() >= 0.5 and share_of_cluster.loc[share_of_type[c].idxmax(), c] >= 0.5
                   for c in t.columns))


def run_arm(arm, g, seed, louvain):
    from polariseq._polariseq import fast_community, leiden_cluster
    indptr, indices, data, n, res = g
    if arm == "igraph":
        import igraph as ig
        import random
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        mask = indices > rows
        graph = ig.Graph(n=n, edges=np.column_stack([rows[mask], indices[mask].astype(np.int64)]).tolist(),
                         directed=False)
        graph.es["weight"] = data[mask].astype(float).tolist()
        ig.set_random_number_generator(random.Random(seed))
        t0 = time.perf_counter()
        part = graph.community_leiden(weights="weight", resolution=res, n_iterations=2,
                                      objective_function="modularity")
        sec = time.perf_counter() - t0
        ig.set_random_number_generator(random)
        return np.asarray(part.membership, dtype=np.int64), sec
    if arm in ("old", "oldfix"):
        raise SystemExit(f"arm {arm!r} needs the development build with the 0.1.x Leiden; the release "
                         "has the new Leiden only (arms: new, igraph)")
    os.environ["POLARISEQ_LEIDEN_IMPL"] = arm
    ip = indptr.astype(np.uint32)
    t0 = time.perf_counter()
    r = (fast_community if louvain else leiden_cluster)(ip, indices, data, n, res, seed)
    return np.asarray(r["labels"], dtype=np.int64), time.perf_counter() - t0


def run(a) -> int:
    from sklearn.metrics import adjusted_rand_score as ari
    z = np.load(a.graph, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    indptr, indices, data = z["indptr"], z["indices"], z["data"]
    n, res = len(indptr) - 1, float(z["resolution"])
    truth = None
    if "labels" in z:
        _, truth = np.unique(z["labels"], return_inverse=True)
        keep = z["labels"] != "NA"
    g = (indptr, indices, data, n, res)
    arms = a.arms.split(",")
    kinds = [("leiden", False)] + ([("louvain", True)] if a.louvain else [])
    report = {"graph": meta, "cells": n, "entries": int(len(indices)), "resolution": res, "seeds": a.seeds}
    for kind, louvain in kinds:
        labels: dict[str, list[np.ndarray]] = {}
        per = {}
        for arm in arms:
            if louvain and arm == "igraph":
                continue
            labels[arm], rows = [], []
            for s in range(a.seeds):
                lab, sec = run_arm(arm, g, s, louvain)
                labels[arm].append(lab)
                row = {"seed": s, "seconds": round(sec, 3), "clusters": int(len(np.unique(lab))),
                       "modularity": round(modularity(indptr, indices, data, lab, res), 6)}
                if truth is not None:
                    row["ari_labels"] = round(float(ari(truth[keep], lab[keep])), 4)
                    row["labels_recovered"] = recovered(lab[keep], truth[keep])
                rows.append(row)
                print(f"  {kind} {arm} seed {s}: {row}", flush=True)
            per[arm] = {"runs": rows}
        old = labels.get("old")  # only on the development build (module docstring)
        base = [ari(x, y) for x, y in combinations(old, 2)] if old else []
        for arm, labs in labels.items():
            own = [ari(x, y) for x, y in combinations(labs, 2)]
            runs = per[arm]["runs"]
            summary = {k: [min(r[k] for r in runs), float(np.median([r[k] for r in runs])), max(r[k] for r in runs)]
                       for k in runs[0] if k != "seed"}
            if old:
                vs_old = [ari(x, y) for x in labs for y in old if x is not y]
                summary["ari_vs_old"] = [round(min(vs_old), 4), round(float(np.median(vs_old)), 4),
                                         round(max(vs_old), 4)]
            summary["ari_seed_to_seed"] = [round(min(own), 4), round(float(np.median(own)), 4), round(max(own), 4)]
            per[arm]["summary_min_median_max"] = summary
        if old:
            per["old"]["summary_min_median_max"]["ari_old_seed_to_seed"] = [round(min(base), 4),
                                                                              round(float(np.median(base)), 4),
                                                                              round(max(base), 4)]
        report[kind] = per
    Path(a.out).write_text(json.dumps(report, indent=1))
    for kind, _ in kinds:
        print(f"\n{meta['dataset']} {kind}: min / median / max over seeds")
        for arm, v in report[kind].items():
            s = v["summary_min_median_max"]
            print(f"  {arm:7s} k {s['clusters']}  Q {s['modularity']}  s {s['seconds']}  "
                  + (f"ARI vs old {s['ari_vs_old']}  " if "ari_vs_old" in s else "")
                  + f"seed-to-seed {s['ari_seed_to_seed']}"
                  + (f"  ARI labels {s['ari_labels']}  recovered {s['labels_recovered']}" if 'ari_labels' in s else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graph")
    g.add_argument("dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--label", default=None, help="obs column with published labels (for a path)")
    g.add_argument("--ram-gb", type=float, default=6)
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--project", default=None)
    r = sub.add_parser("run")
    r.add_argument("graph")
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--arms", default="new,igraph")
    r.add_argument("--louvain", action="store_true")
    a = ap.parse_args()
    return build_graph(a) if a.cmd == "graph" else run(a)


if __name__ == "__main__":
    sys.exit(main())
