#!/usr/bin/env python
"""Memory of the neighbours step as a function of the number of cells, on a real embedding.

    python neighbors_memory.py EMBEDDING.npy N --project DIR [--seed 0] [--threads 16] [--ram-gb 150]

Draws N random rows of the embedding (a .npy of float32 cells x components, opened memory-mapped), puts
them in a Polariseq object and runs `pp.neighbors` as the one-call analysis does: results in memory-mapped
files of a project. Run under `benchmarks/atlas/monitor.py`, which samples resident and anonymous memory,
and with POLARISEQ_TRACE=1, which makes NN-Descent print its phases, to see which phase holds the peak.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("embedding")
    ap.add_argument("n", type=int)
    ap.add_argument("--project", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ram-gb", type=float, default=150)
    ap.add_argument("--neighbors", type=int, default=15)
    a = ap.parse_args()
    import polariseq as ps

    ps.set_budget(ram_gb=a.ram_gb, threads=a.threads)
    ps.project(a.project)
    src = np.load(a.embedding, mmap_mode="r")
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(src.shape[0], size=a.n, replace=False)) if a.n < src.shape[0] else slice(None)
    t0 = time.perf_counter()
    emb = np.ascontiguousarray(src[rows], dtype=np.float32)
    print(f"[neighbors_memory] {emb.shape[0]:,} cells x {emb.shape[1]} read in {time.perf_counter() - t0:.0f} s",
          flush=True)
    adata = ps.AnnData(X=sp.csr_matrix((emb.shape[0], 1), dtype=np.float32))
    adata.obsm["X_pca"] = emb
    # a named dataset in a project: the graph goes to memory-mapped result files, as in a real run
    adata.name = f"n{emb.shape[0]}"
    t0 = time.perf_counter()
    ps.pp.neighbors(adata, n_neighbors=a.neighbors, seed=0)
    print(f"[neighbors_memory] neighbors done in {time.perf_counter() - t0:.0f} s; "
          f"{adata.obsp['connectivities'].nnz:,} connectivity entries; inputs "
          f"{adata.uns['neighbors'].get('inputs', 'n/a')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
