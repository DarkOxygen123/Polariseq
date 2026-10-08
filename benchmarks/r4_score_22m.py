"""Score 22M-cell runs of examples/06_large_run.py against scTab's published cell types.

    python benchmarks/r4_score_22m.py RUN_DIR [RUN_DIR ...] --out OUT.json

Reads each RUN_DIR/cells.tsv.gz (columns leiden, cell_type, umap1, umap2, row for row over the cells
kept) and writes the same measures as the sweeps: clusters against the published types (ARI, NMI,
purity, types recovered) and the UMAP's islands and whole types.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_scarf_ablation import scores  # noqa: E402
from r3_umap_fragmentation import islands  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for r in a.runs:
    d = pd.read_csv(Path(r) / "cells.tsv.gz", sep="\t", usecols=["leiden", "cell_type", "umap1", "umap2"],
                    dtype={"leiden": "int32", "cell_type": "category", "umap1": "float32", "umap2": "float32"})
    types = d.cell_type.cat.codes.to_numpy().astype(np.int64)
    out = {"cells": int(len(d)), "clusters_vs_types": scores(d.leiden.to_numpy(), types)}
    out["umap"] = islands(d[["umap1", "umap2"]].to_numpy(), types)
    res[r] = out
    print(r, json.dumps(out), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
