#!/usr/bin/env python
"""The published labels of a dataset, once, for scoring every tool's run.

    python build_truth.py --input SHARDS --out DATASET_DIR/truth.npz [--label-col cell_type] [--workers 16]

Writes truth.npz: label (int16, one per global row: the position in the concatenated input files),
label_names, and profiles: per label, the mean of log1p(counts scaled to 10,000 per cell) over the cells
with at least min_genes genes, for every gene (labels x genes, float32). The profiles give the
expression distances between types that the UMAP's global structure is compared with (score.py).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, Steps, emit, shard_files  # noqa: E402


def labels_of(path: str, col: str):
    import h5py
    import numpy as np
    with h5py.File(path, "r") as h:
        g = h["obs"][col]
        if hasattr(g, "keys"):
            return g["codes"][:].astype(np.int64), g["categories"][:].astype(str)
        vals = g[:].astype(str)
    cats, codes = np.unique(vals, return_inverse=True)
    return codes.astype(np.int64), cats


def profile_sums(args):
    """Per label of this file: sum of log1p(CP10k) per gene and cell count (cells passing QC)."""
    path, lab, n_labels, block = args
    import h5py
    import numpy as np
    from scipy.sparse import csr_matrix
    with h5py.File(path, "r") as h:
        X = h["X"]
        indptr = X["indptr"][:].astype(np.int64)
        n, g = int(X.attrs["shape"][0]), int(X.attrs["shape"][1])
        sums = np.zeros((n_labels, g), np.float64)
        counts = np.zeros(n_labels, np.int64)
        for r0 in range(0, n, block):
            r1 = min(r0 + block, n)
            a, b = indptr[r0], indptr[r1]
            m = csr_matrix((X["data"][a:b].astype(np.float32), X["indices"][a:b], indptr[r0:r1 + 1] - a),
                           shape=(r1 - r0, g))
            lens = np.diff(m.indptr)
            lib = np.asarray(m.sum(axis=1)).ravel()
            m.data = np.log1p(m.data / np.repeat(lib, lens) * P["target_sum"])
            keep = lens >= P["min_genes"]
            ind = csr_matrix((np.ones(int(keep.sum())), (lab[r0:r1][keep], np.flatnonzero(keep))),
                             shape=(n_labels, r1 - r0))
            sums += (ind @ m).toarray()
            counts += np.bincount(lab[r0:r1][keep], minlength=n_labels)
    return sums, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-col", default="cell_type")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--no-profiles", action="store_true")
    a = ap.parse_args()
    st = Steps()
    import multiprocessing as mp

    import numpy as np
    files = shard_files(a.input)
    with st("labels"):
        names: dict[str, int] = {}
        per_file = []
        for f in files:
            codes, cats = labels_of(f, a.label_col)
            remap = np.array([names.setdefault(c, len(names)) for c in cats], dtype=np.int64)
            per_file.append(remap[codes])
        label = np.concatenate(per_file).astype(np.int16)
        label_names = np.array(sorted(names, key=names.get))
    out = {"label": label, "label_names": label_names, "label_col": a.label_col}
    if not a.no_profiles:
        with st("profiles"):
            with mp.get_context("fork").Pool(a.workers) as pool:
                parts = pool.map(profile_sums, [(f, lab, len(label_names), 50_000)
                                                for f, lab in zip(files, per_file)])
            sums = sum(p[0] for p in parts)
            counts = sum(p[1] for p in parts)
            out["profiles"] = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
            out["profile_cells"] = counts
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    emit({"cells": int(len(label)), "labels": int(len(label_names)), "files": len(files)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
