"""The fetal atlas handed to Scanpy's PAGA, and Paris on the same graph, with the final 0.2.0.
Run on the cluster in the Scanpy
environment with the 0.2.0 build on the path (cluster/v020_downstream_paris.sbatch):

    python v020_downstream_paris.py OUT_DIR SPILL_DIR

The path a user takes, as benchmarks/r3_downstream.py: Polariseq's out-of-core analysis of all 4,062,980 fetal
cells under the default (the node runs' settings: 6 GB budget, four threads, seed 0, cells with at least 200 genes),
``to_anndata()``, then ``scanpy.tl.paga`` on the exported graph and Leiden clusters, each stage timed with the
process's peak resident memory after it. The clusters are checked against the out-of-core node run
(runs/v020/fetal4062k_polariseq-spill_t4). The exported neighbour graph is then saved and Paris (Polariseq's port,
as benchmarks/validate_paris.py) runs on it in a child process, alone, timed with its peak resident memory. The
hierarchy is scored on the Mac (benchmarks/validate_paris.py fetal_tree, with OUT_DIR as its folder).
"""
import glob
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

B = os.environ["B"]  # the working folder (REPRODUCE.md)
FILE = f"{B}/data/paper/fetal_data/fetal4062k.h5ad"
CELLS = f"{B}/data/paper/fetal_data/fetal_cells.csv.gz"
RUN = glob.glob(f"{B}/runs/v020/fetal4062k_polariseq-spill_t4/artifacts/*/")
PARAMS = dict(n_hvg=2000, n_pcs=50, n_neighbors=15, target_sum=1e4, resolution=1.0, seed=0, min_genes=200)


def peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 2)  # Linux: KiB


def paris_child(out: Path) -> None:
    from polariseq._polariseq import paris_dendrogram

    z = np.load(out / "fetal_graph.npz")
    # the port takes float64 weights (as validate_paris.py fed it from Polariseq's own graph); the exported graph
    # is float32, and float32 values are exact in float64
    ip, ix, dx, n = (z["indptr"].astype(np.uint32), z["indices"].astype(np.uint32), z["data"].astype(np.float64),
                     int(z["n"]))
    loaded = peak_gb()
    t = time.perf_counter()
    d = paris_dendrogram(ip, ix, dx, n)
    s = time.perf_counter() - t
    np.save(out / "fetal_dendrogram.npy", d)
    (out / "fetal_paris.json").write_text(json.dumps(
        {"seconds": round(s, 1), "peak_resident_gb": peak_gb(), "graph_loaded_resident_gb": loaded, "n": n,
         "nnz": int(dx.size)}, indent=1))


def main() -> None:
    out, spill = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    import polariseq as ps
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    ps.set_budget(ram_gb=6.0, threads=4, spill_dir=spill)
    peaks = {}
    t0 = time.perf_counter()
    a = ps.process_diskbacked(FILE, **PARAMS)
    t_pipe = time.perf_counter() - t0
    peaks["polariseq_pipeline"] = peak_gb()
    labels = np.asarray(a.obs["leiden"]).astype(np.int64)
    obs = np.asarray(a.obs_names, dtype=object)
    check = {}
    if RUN:
        run_obs = np.array([s for s in open(RUN[0] + "obs_names.txt").read().split("\n") if s], dtype=object)
        run_lab = np.load(RUN[0] + "labels.npy")
        same = bool(np.array_equal(run_obs, obs))
        check = {"node_run": RUN[0], "same_cells_as_node_run": same,
                 "ari_with_node_run": float(adjusted_rand_score(run_lab, labels)) if same else None}
        print("clusters against the node run:", check, flush=True)

    t0 = time.perf_counter()
    ad = a.to_anndata()  # clusters and graph keys in Scanpy's conventions; no setup
    del a
    t_export = time.perf_counter() - t0
    peaks["to_anndata"] = peak_gb()

    g = ad.obsp["connectivities"].tocsr()  # saved before PAGA, for Paris below
    np.savez(out / "fetal_graph.npz", indptr=g.indptr, indices=g.indices, data=g.data, n=g.shape[0])
    np.savetxt(out / "obs_names.txt", np.asarray(ad.obs_names, dtype=object), fmt="%s")
    np.save(out / "labels.npy", labels)
    del g

    t0 = time.perf_counter()
    sc.tl.paga(ad, groups="leiden")
    t_paga = time.perf_counter() - t0
    peaks["scanpy_paga"] = peak_gb()

    cells = pd.read_csv(CELLS, index_col="cell", usecols=["cell", "Main_cluster_name", "Organ"])
    ann = cells.loc[np.asarray(ad.obs_names)]
    groups = []
    for g in ad.obs["leiden"].cat.categories:
        m = (ad.obs["leiden"] == g).to_numpy()
        t = ann["Main_cluster_name"][m].value_counts()
        o = ann["Organ"][m].value_counts()
        groups.append({"cluster": str(g), "cells": int(m.sum()), "type": str(t.index[0]),
                       "type_share": round(float(t.iloc[0] / m.sum()), 3), "organ": str(o.index[0]),
                       "organ_share": round(float(o.iloc[0] / m.sum()), 3)})
    np.savez_compressed(out / "paga.npz", connectivities=ad.uns["paga"]["connectivities"].toarray())
    summary = {"n_cells": int(ad.n_obs), "n_clusters": len(groups), "polariseq": ps.__version__,
               "scanpy": sc.__version__, "threads": 4, "budget_gb": 6.0, "params": PARAMS,
               "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
               "load_at_start": os.getloadavg()[0],
               "seconds": {"polariseq_pipeline": round(t_pipe, 1), "to_anndata": round(t_export, 1),
                           "scanpy_paga": round(t_paga, 1)},
               "peak_resident_gb_after": peaks, **check, "groups": groups}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1), flush=True)

    del ad
    subprocess.run([sys.executable, __file__, "_paris", str(out)], check=True)
    print(json.loads((out / "fetal_paris.json").read_text()), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "_paris":
        paris_child(Path(sys.argv[2]))
    else:
        main()
