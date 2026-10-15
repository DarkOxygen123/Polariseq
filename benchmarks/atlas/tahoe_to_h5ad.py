#!/usr/bin/env python
"""Convert downloaded Tahoe-100M parquet files to .h5ad shards that all three tools read (compute node).

    python tahoe_to_h5ad.py --parquet DIR --out DIR [--files-per-shard 35] [--workers 16] [--delete-parquet]
                            [--start-index N]

Each parquet row is one cell: a list of gene tokens and their counts (the first token, 1 with value -2, is
a marker and is dropped). Tokens map to genes through gene_metadata.parquet (in DIR); the genes are the
columns in token order, so each cell's gene indices stay sorted. Each shard (35 files, about 990,000
cells) is an AnnData file: X raw counts as CSR (float32 values, int32 indices, int64 row pointers),
obs: plate, sample, cell_line_id, drug, sublibrary (the capture: one parquet file each), var: the
Ensembl id as index and gene_symbol. Shards are written in parallel, each first to a .tmp name.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

OBS = ["plate", "sample", "cell_line_id", "drug"]
STR = h5py.string_dtype("utf-8")


def enc(obj, kind: str, version: str) -> None:
    obj.attrs["encoding-type"] = kind
    obj.attrs["encoding-version"] = version


def write_strings(g, name: str, values) -> None:
    d = g.create_dataset(name, data=np.asarray(values, dtype=object), dtype=STR)
    enc(d, "string-array", "0.2.0")


def write_categorical(g, name: str, values: np.ndarray) -> None:
    cats, codes = np.unique(values.astype(str), return_inverse=True)
    c = g.create_group(name)
    enc(c, "categorical", "0.2.0")
    c.attrs["ordered"] = False
    c.create_dataset("codes", data=codes.astype(np.int16 if len(cats) < 32000 else np.int32))
    write_strings(c, "categories", cats)


def write_shard(args) -> tuple[str, int, int]:
    path, files, lut, genes = args
    tmp = Path(str(path) + ".tmp")
    n_genes = len(genes["ensembl_id"])
    with h5py.File(tmp, "w") as h:
        enc(h, "anndata", "0.1.0")
        X = h.create_group("X")
        enc(X, "csr_matrix", "0.1.0")
        data = X.create_dataset("data", shape=(0,), maxshape=(None,), dtype="f4", chunks=(1 << 20,))
        indices = X.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i4", chunks=(1 << 20,))
        indptr = [np.zeros(1, np.int64)]
        obs = {c: [] for c in OBS + ["sublibrary"]}
        nnz = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=16384, columns=["genes", "expressions", "BARCODE_SUB_LIB_ID"] + OBS):
                g, e = b.column("genes"), b.column("expressions")
                off = g.offsets.to_numpy().astype(np.int64)
                off -= off[0]
                tok = g.flatten().to_numpy()
                val = e.flatten().to_numpy()
                lens = np.diff(off)
                row = np.repeat(np.arange(len(lens)), lens)
                col = lut[tok]
                keep = col >= 0
                row, col, val = row[keep], col[keep], val[keep]
                same = row[1:] == row[:-1]
                if np.any(col[1:][same] <= col[:-1][same]):   # keep each cell's genes sorted, as CSR expects
                    o = np.lexsort((col, row))
                    row, col, val = row[o], col[o], val[o]
                k = len(col)
                data.resize((nnz + k,))
                indices.resize((nnz + k,))
                data[nnz:] = val.astype(np.float32)
                indices[nnz:] = col.astype(np.int32)
                indptr.append(nnz + np.cumsum(np.bincount(row, minlength=len(lens))))
                nnz += k
                for c in OBS:
                    obs[c].append(b.column(c).to_numpy(zero_copy_only=False).astype(str))
                obs["sublibrary"].append(np.array([s.rsplit("-", 1)[-1] for s in
                                                   b.column("BARCODE_SUB_LIB_ID").to_pylist()]))
        ptr = np.concatenate(indptr)
        n = len(ptr) - 1
        X.create_dataset("indptr", data=ptr)
        X.attrs["shape"] = np.array([n, n_genes], dtype=np.int64)
        o = h.create_group("obs")
        enc(o, "dataframe", "0.2.0")
        o.attrs["_index"] = "_index"
        cols = OBS + ["sublibrary"]
        o.attrs["column-order"] = cols
        write_strings(o, "_index", np.arange(n).astype(str))
        for c in cols:
            write_categorical(o, c, np.concatenate(obs[c]))
        v = h.create_group("var")
        enc(v, "dataframe", "0.2.0")
        v.attrs["_index"] = "_index"
        v.attrs["column-order"] = ["gene_symbol"]
        write_strings(v, "_index", genes["ensembl_id"])
        write_strings(v, "gene_symbol", genes["gene_symbol"])
        for k in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
            enc(h.create_group(k), "dict", "0.1.0")
    os.replace(tmp, path)
    return str(path), n, nnz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files-per-shard", type=int, default=35)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--delete-parquet", action="store_true", help="delete each shard's parquet files once written")
    ap.add_argument("--start-index", type=int, default=0,
                    help="number of the first shard (the files are N.h5ad, N+1.h5ad, ...): lets new shards follow "
                         "shards that already exist in the same folder")
    a = ap.parse_args()
    src = Path(a.parquet)
    gm = pq.read_table(src / "gene_metadata.parquet").to_pandas().sort_values("token_id")
    if gm["ensembl_id"].duplicated().any():   # keep names unique: suffix repeats with the token
        dup = gm["ensembl_id"].duplicated(keep=False)
        gm.loc[dup, "ensembl_id"] = gm.loc[dup, "ensembl_id"] + "_" + gm.loc[dup, "token_id"].astype(str)
    lut = np.full(int(gm["token_id"].max()) + 1, -1, np.int64)
    lut[gm["token_id"].to_numpy()] = np.arange(len(gm))
    genes = {"ensembl_id": gm["ensembl_id"].to_numpy(str), "gene_symbol": gm["gene_symbol"].fillna("").to_numpy(str)}
    files = sorted(str(p) for p in src.glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {src}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in range(0, len(files), a.files_per_shard):
        dest = out / f"{a.start_index + s // a.files_per_shard}.h5ad"
        if not dest.exists():
            jobs.append((dest, files[s:s + a.files_per_shard], lut, genes))
    print(f"{len(files)} parquet files, {len(gm):,} genes, {len(jobs)} shards to write", flush=True)
    total = 0
    with get_context("fork").Pool(a.workers) as pool:
        for path, n, nnz in pool.imap_unordered(write_shard, jobs):
            total += n
            print(f"{path}: {n:,} cells, {nnz:,} values", flush=True)
            if a.delete_parquet:
                for f in next(j[1] for j in jobs if str(j[0]) == path):
                    Path(f).unlink(missing_ok=True)
    print(f"done: {total:,} cells in {len(jobs)} new shards", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
