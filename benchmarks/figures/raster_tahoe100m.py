"""Rasters of the Tahoe-100M UMAP (all 95,596,109 cells), by cell line, by cluster and by plate, with the same
code as the gallery (embedding_raster.py), plus a 20,000-cell sample for review; with CELL_LINES_CSV (cell line,
name and organ of origin, from Tahoe's cell-line metadata), also by organ of origin.
Run on the cluster: python raster_tahoe100m.py CELLS_TSV_GZ OUTDIR [CELL_LINES_CSV]"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import embedding_raster as ER

src, out = sys.argv[1], Path(sys.argv[2])
lines_csv = sys.argv[3] if len(sys.argv) > 3 else None
# the run the table comes from: its results folder (tahoe-all: the 0.1.1 run; tahoe-all-020: 0.2.0)
run_id = os.environ.get("RUN_ID") or f"Tahoe-100M ({Path(src).resolve().parents[2].name})"
prefix = "tahoe100M__polariseq-spill"
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(src, sep="\t", usecols=["leiden", "cell_line_id", "plate", "umap1", "umap2"],
                dtype={"leiden": "int32", "cell_line_id": "category", "plate": "category",
                       "umap1": "float32", "umap2": "float32"})
xy = d[["umap1", "umap2"]].to_numpy()
assert np.isfinite(xy).all()
cl_names, cl_codes = np.unique(d.leiden.to_numpy(), return_inverse=True)
ER.save(out / f"{prefix}__clusters.npz", ER.raster(xy, cl_codes), cl_names, run_id=run_id, cells=len(d))
ln = d.cell_line_id.cat
pl = d.plate.cat
ER.save(out / f"{prefix}__plates.npz", ER.raster(xy, pl.codes.to_numpy()), pl.categories, run_id=run_id, cells=len(d))
ER.save(out / f"{prefix}__cell_lines.npz", ER.raster(xy, ln.codes.to_numpy()), ln.categories,
        run_id=run_id, cells=len(d))
if lines_csv:
    organ_of = pd.read_csv(lines_csv).drop_duplicates("cell_line_id").set_index("cell_line_id").organ
    organ = pd.Categorical(ln.categories.map(organ_of))
    assert not pd.isna(np.asarray(organ, dtype=object)).any(), "a cell line without an organ"
    codes = organ.codes[ln.codes.to_numpy()]
    ER.save(out / f"{prefix}__organs.npz", ER.raster(xy, codes), organ.categories, run_id=run_id, cells=len(d))
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(d), 20000, replace=False))
s = d.iloc[idx].rename(columns={"leiden": "cluster"})
s.insert(0, "row", idx)
s.to_csv(out / f"{prefix}.sample20k.csv.gz", index=False)
lines = d.groupby("cell_line_id", observed=True).size().rename("cells")
lines.to_csv(out / "tahoe100M_cells_per_line.csv")
json.dump({"cells": int(len(d)), "clusters": int(len(cl_names)), "cell_lines": int(len(ln.categories)),
           "plates": int(len(d.plate.cat.categories))},
          open(out / "tahoe100M_summary.json", "w"))
print("done", len(d), len(cl_names), len(ln.categories))
