"""Pack the embedding of every reported cluster run into one labelled .npz per run.

Run on the cluster (Slurm job): python benchmarks/figures/pack_embeddings.py OUTDIR
Edit R, CELLS and RUNS below to point at your own node runs (benchmarks/slurm/node_run.sbatch writes one folder
per run, <dataset>_<arm>_t4, under benchmarks/results/node).
Each file <dataset>__<arm>__<run>.npz holds, row for row over the cells the run kept:
  umap (n, 2) float32      the run's own UMAP coordinates
  cluster (n,) int32       the run's own Leiden / Scarf clusters
  ref_cluster (n,) int32   the cluster of the same cell (matched by name) in Polariseq in memory; -1 if absent
  cell_type (n,) int16     fetal atlas only: code of the authors' Main_cluster_name; names in cell_type_names
  organ (n,) int16         fetal atlas only: code of the organ; names in organ_names
  obs_names                cell names, as written by the run
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = "benchmarks/results/node"
CELLS = "benchmarks/results/fetal_data/fetal_cells.csv.gz"
RUNS = {
    "brain1306k": {"polariseq-ram": "brain1306k__polariseq-ram__rep1__20261003T171951",
                   "polariseq-spill": "brain1306k__polariseq-spill__rep1__20261003T172626",
                   "scanpy-tuned-int64": "brain1306k__scanpy-tuned-int64__rep1__20261003T171918",
                   "scarf": "brain1306k__scarf__rep1__20261004T150332"},
    "fetal4062k": {"polariseq-ram": "fetal4062k__polariseq-ram__rep1__20261003T173528",
                   "polariseq-spill": "fetal4062k__polariseq-spill__rep1__20261003T175755",
                   "scanpy-tuned-int64": "fetal4062k__scanpy-tuned-int64__rep1__20261004T130538",
                   "scarf": "fetal4062k__scarf__rep1__20261004T125231"},
}
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
fetal = pd.read_csv(CELLS,
                    usecols=["cell", "Main_cluster_name", "Organ"], dtype={"Main_cluster_name": "category", "Organ": "category"})
fetal = fetal.set_index("cell")
summary = []
for ds, arms in RUNS.items():
    def load(arm):
        d = glob.glob(f"{R}/{ds}_{arm}_t4/artifacts/{arms[arm]}")[0] + "/"
        names = np.array([s for s in open(d + "obs_names.txt").read().split("\n") if s])
        return d, names, np.load(d + "umap.npy").astype(np.float32), np.load(d + "labels.npy").astype(np.int32)
    _, ref_names, _, ref_labels = load("polariseq-ram")
    ref = pd.Series(ref_labels, index=ref_names)
    for arm, run in arms.items():
        d, names, umap, labels = load(arm)
        assert len(names) == len(umap) == len(labels), (arm, len(names), umap.shape, labels.shape)
        rc = ref.reindex(names).fillna(-1).astype(np.int32).to_numpy()
        extra = {}
        if ds == "fetal4062k":
            f = fetal.reindex(names)
            extra = {"cell_type": f.Main_cluster_name.cat.codes.to_numpy().astype(np.int16),
                     "cell_type_names": np.array(fetal.Main_cluster_name.cat.categories, dtype=object).astype(str),
                     "organ": f.Organ.cat.codes.to_numpy().astype(np.int16),
                     "organ_names": np.array(fetal.Organ.cat.categories, dtype=object).astype(str)}
        np.savez_compressed(out / f"{ds}__{arm}__{run.split('__')[-1]}.npz", umap=umap, cluster=labels,
                            ref_cluster=rc, obs_names=names.astype(str), run_id=np.array(run), **extra)
        row = {"dataset": ds, "arm": arm, "run_id": run, "cells": int(len(names)),
               "clusters": int(len(np.unique(labels))), "matched_to_polariseq_in_memory": int((rc >= 0).sum()),
               "umap_finite": bool(np.isfinite(umap).all())}
        if extra:
            row["cells_with_published_type"] = int((extra["cell_type"] >= 0).sum())
        summary.append(row)
        print(row, flush=True)
json.dump(summary, open(out / "summary.json", "w"), indent=1)
