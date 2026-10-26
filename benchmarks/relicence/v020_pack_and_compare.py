"""The compute-node runs of the final 0.2.0 (runs/v020): packed embeddings and Polariseq's two
modes compared. Run on the cluster (Slurm job):

    python v020_pack_and_compare.py OUTDIR

1. OUTDIR/cluster_embeddings_v020/<dataset>__<arm>__<run>.npz, as benchmarks/figures/pack_embeddings.py writes them
   (umap, cluster, ref_cluster: the same cell's cluster in Polariseq in memory, obs_names; fetal atlas: published
   cell type and organ), for Polariseq in memory and out of core at 4 threads. The other tools' packed runs are
   in benchmarks/results/cluster_embeddings (their artifacts are no longer on the cluster); the agreement of the tools'
   clusters is computed from the packed files (benchmarks/relicence/v020_agreement.py).
2. OUTDIR/modes_cmp_v020.json, as benchmarks/r4_modes_compare.py: in memory against out of core (4 and 8 threads)
   and out of core at 4 against 8 threads, with the selected genes of each run (OUTDIR/hvg_v020_<run>.txt).
"""
from __future__ import annotations

import os
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

B = os.environ["B"]  # the working folder (REPRODUCE.md)
V020 = f"{B}/runs/v020"
ARM_OF = {"Polariseq in-memory": "polariseq-ram", "Polariseq out-of-core": "polariseq-spill"}


def v020_dir(ds: str, arm: str, threads: int) -> str:
    return glob.glob(f"{V020}/{ds}_{arm}_t{threads}/artifacts/*/")[0]


def load(d: str) -> dict:
    names = np.array([s for s in open(d + "obs_names.txt").read().split("\n") if s])
    out = {"dir": d, "names": names, "labels": np.load(d + "labels.npy").astype(np.int32),
           "umap": np.load(d + "umap.npy").astype(np.float32), "hvg": open(d + "hvg.txt").read().split()}
    pca = Path(d) / "pca.npy"
    out["pca"] = np.load(pca) if pca.exists() else None
    assert len(names) == len(out["labels"]) == len(out["umap"]), d
    return out


def angles(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    qa, _ = np.linalg.qr(pa.astype(np.float64))
    qb, _ = np.linalg.qr(pb.astype(np.float64))
    return np.degrees(np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)))


def main() -> None:
    out = Path(sys.argv[1])
    emb = out / "cluster_embeddings_v020"
    emb.mkdir(parents=True, exist_ok=True)
    fetal = pd.read_csv(f"{B}/data/paper/fetal_data/fetal_cells.csv.gz",
                        usecols=["cell", "Main_cluster_name", "Organ"],
                        dtype={"Main_cluster_name": "category", "Organ": "category"}).set_index("cell")
    summary = []
    for ds in ("brain1306k", "fetal4062k"):
        runs = {"Polariseq out-of-core": load(v020_dir(ds, "polariseq-spill", 4)),
                "Polariseq in-memory": load(v020_dir(ds, "polariseq-ram", 4))}
        ref = pd.Series(runs["Polariseq in-memory"]["labels"], index=runs["Polariseq in-memory"]["names"])
        for tool, r in runs.items():
            run_id = Path(r["dir"]).name
            rc = ref.reindex(r["names"]).fillna(-1).astype(np.int32).to_numpy()
            extra = {}
            if ds == "fetal4062k":
                f = fetal.reindex(r["names"])
                extra = {"cell_type": f.Main_cluster_name.cat.codes.to_numpy().astype(np.int16),
                         "cell_type_names": np.array(fetal.Main_cluster_name.cat.categories, dtype=object).astype(str),
                         "organ": f.Organ.cat.codes.to_numpy().astype(np.int16),
                         "organ_names": np.array(fetal.Organ.cat.categories, dtype=object).astype(str)}
            np.savez_compressed(emb / f"{ds}__{ARM_OF[tool]}__{run_id.split('__')[-1]}.npz", umap=r["umap"],
                                cluster=r["labels"], ref_cluster=rc, obs_names=r["names"].astype(str),
                                run_id=np.array(run_id), **extra)
            summary.append({"dataset": ds, "tool": tool, "run_id": run_id, "cells": int(len(r["names"])),
                            "clusters": int(len(np.unique(r["labels"]))),
                            "matched_to_polariseq_in_memory": int((rc >= 0).sum())})
            print(summary[-1], flush=True)
        for tool, r in runs.items():
            (out / f"hvg_v020_{Path(r['dir']).name}.txt").write_text("\n".join(r["hvg"]) + "\n")
    (emb / "summary.json").write_text(json.dumps(summary, indent=1))
    # the two modes, and the threads
    modes = {}
    pairs = {"brain1306k": [(("polariseq-ram", 4), ("polariseq-spill", 4))],
             "fetal4062k": [(("polariseq-ram", 4), ("polariseq-spill", 4)), (("polariseq-ram", 8), ("polariseq-spill", 8)),
                            (("polariseq-spill", 4), ("polariseq-spill", 8))]}
    for ds, ps in pairs.items():
        for a, b in ps:
            ra, rb = load(v020_dir(ds, *a)), load(v020_dir(ds, *b))
            assert (ra["names"] == rb["names"]).all()
            ang = angles(ra["pca"], rb["pca"])
            k = f"{ds} {a} vs {b}"
            modes[k] = {"same_genes": set(ra["hvg"]) == set(rb["hvg"]), "pc_max_angle_deg": float(ang.max()),
                        "ari": float(ari(ra["labels"], rb["labels"])),
                        "clusters": [int(len(set(ra["labels"]))), int(len(set(rb["labels"])))]}
            print(k, modes[k], flush=True)
    (out / "modes_cmp_v020.json").write_text(json.dumps(modes, indent=1))


if __name__ == "__main__":
    main()
