#!/usr/bin/env python
"""Leiden of the Python environment running this on a saved connectivity graph (the Tahoe-100M run's).

    python leiden_param.py GRAPH_DIR --seed 0 --out labels.npy [--resolution 1.0] [--inputs mapped]

GRAPH_DIR holds connectivities_{indptr,indices,data}.*.npy (a run's results/neighbors). Writes the labels
and a JSON beside them with the version, seconds, clusters and modularity. Run it under monitor.py for memory.
--inputs loaded (default) reads the graph into memory first; mapped reads it in place from the files. On the cluster's
Lustre the mapped graph stalled Leiden at 0.02-0.1 cores for hours, also with madvise(WILLNEED) every 30 s,
which is why 0.2.0 never reads graph inputs in place from a network file system.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", choices=("loaded", "mapped"), default="loaded")
    a = ap.parse_args()
    import polariseq as ps
    from polariseq._polariseq import leiden_cluster
    g = Path(a.graph)
    # loaded: read into memory in one pass; Leiden's random access to a memory-mapped graph on the cluster's shared
    # file system stalls whenever the node drops the cached pages (0.03 cores for hours without the keeper)
    t0 = time.perf_counter()
    mode = "c" if a.inputs == "mapped" else None
    load = lambda k: np.load(next(g.glob(f"connectivities_{k}.*.npy")), mmap_mode=mode)
    indptr, indices, data = load("indptr"), load("indices"), load("data")
    read_s = time.perf_counter() - t0
    n = len(indptr) - 1
    t0 = time.perf_counter()
    r = leiden_cluster(np.asarray(indptr, dtype=np.uint32), np.asarray(indices, dtype=np.uint32),
                       np.asarray(data, dtype=np.float32), n, a.resolution, a.seed)
    sec = time.perf_counter() - t0
    np.save(a.out, np.asarray(r["labels"], dtype=np.int32))
    info = {"polariseq": ps.__version__, "seed": a.seed, "cells": n, "seconds": round(sec, 1), "read_seconds": round(read_s, 1),
            "inputs": a.inputs,
            "clusters": int(r["n_communities"]), "modularity": float(r["quality"])}
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
