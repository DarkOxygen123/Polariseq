"""Derive cell subsets from the 10x 1.3M mouse brain file, with a recorded seed.

The 10x HDF5 layout stores the matrix CSC with one column per cell, so taking a
subset of cells is a column gather and costs one pass over `indptr` plus the
slices actually selected. Nothing densifies and the full matrix is never held.

Output is `.h5ad` with CSR `X` (cells x genes), which is what every arm reads.

Provenance is written into `uns` and echoed to stdout so the seed, the parent
file and its checksum travel with the derived file rather than living in a
lab notebook.

Usage:
    python benchmarks/make_subsets.py --sizes 50000 100000 200000
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

#: The 10x 1.3M mouse brain file, as downloaded.
DEFAULT_SOURCE = (
    "benchmarks/results/validation_data/"
    "1M_neurons_filtered_gene_bc_matrices_h5.h5"
)
DEFAULT_OUT = "benchmarks/results/validation_data"

#: Rows gathered per write. Bounds peak memory during the gather to roughly
#: `CHUNK * nnz_per_cell * 8` bytes rather than the whole subset.
CHUNK = 20_000


def file_sha256(path: str, *, limit_mb: int | None = 64) -> str:
    """Checksum the first `limit_mb` of a file, or all of it if None.

    A full checksum of a 4.2 GB file costs ~20 s on this hardware and is run
    on every invocation, so the default hashes a prefix. That is enough to
    detect "wrong file" and "truncated download", which is what this guards
    against.
    """
    h = hashlib.sha256()
    remaining = None if limit_mb is None else limit_mb * 1024 * 1024
    with open(path, "rb") as fh:
        while True:
            want = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if want <= 0:
                break
            b = fh.read(want)
            if not b:
                break
            h.update(b)
            if remaining is not None:
                remaining -= len(b)
    return h.hexdigest()


def open_10x(path: str):
    """Return (group, n_genes, n_cells) for a 10x CellRanger HDF5 file."""
    f = h5py.File(path, "r")
    key = next(iter(f.keys()))
    g = f[key]
    n_genes, n_cells = (int(x) for x in g["shape"][:])
    return f, g, n_genes, n_cells


def gather_cells(g, cols: np.ndarray, n_genes: int, *, progress: bool = False) -> sp.csr_matrix:
    """Gather the given cell columns into a CSR (cells x genes) matrix.

    `cols` must be sorted. Reads only the `data`/`indices` spans the selected
    cells occupy, in ascending file order, so the access pattern stays
    sequential enough for the HDF5 chunk cache to be useful.

    The output is allocated **once**, from the exact nnz known in advance via
    `indptr`. An earlier version gathered in chunks and `vstack`ed them, which
    holds the parts and the result simultaneously and peaked near twice the
    final size — 6.4 GB for a 200K subset on an 8 GiB machine. Peak here is
    the output arrays alone.
    """
    indptr = g["indptr"][:]  # n_cells + 1, cheap (one int64 per cell)
    starts = indptr[cols]
    ends = indptr[cols + 1]
    counts = (ends - starts).astype(np.int64)

    total = int(counts.sum())
    out_data = np.empty(total, dtype=np.float32)
    out_idx = np.empty(total, dtype=np.int32)

    data_ds = g["data"]
    idx_ds = g["indices"]

    pos = 0
    n_sel = len(cols)
    for i, (s, e, n) in enumerate(zip(starts, ends, counts, strict=True)):
        if n:
            out_data[pos : pos + n] = data_ds[s:e]
            out_idx[pos : pos + n] = idx_ds[s:e]
            pos += n
        if progress and i % 20_000 == 0:
            print(f"    gathered {i:,}/{n_sel:,}", end="\r", flush=True)

    out_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    # Each 10x "column" is a cell and its indices are gene ids, so this is
    # already (cells x genes) in CSR once assembled row-wise.
    return sp.csr_matrix(
        (out_data, out_idx, out_indptr), shape=(len(cols), n_genes)
    )


#: Above this many cells the subset is written incrementally rather than
#: assembled in memory. At ~2010 nnz/cell an in-memory gather costs
#: `n * 2010 * 8` bytes: 3.2 GB at 200K, 6.4 GB at 400K, 12.9 GB at 800K —
#: past what an 8 GiB machine can hold while also running the writer.
STREAM_ABOVE_CELLS = 250_000


def make_subset_streaming(source: str, n: int, seed: int, out_dir: str) -> dict:
    """Build a large subset without ever holding it in memory.

    Each block is written as an ordinary `.h5ad` by anndata, then
    `concat_on_disk` joins them. Peak memory is one block.

    Two earlier attempts wrote the h5ad spec by hand and both produced files
    anndata refused to read back — first a missing `column-order` attribute
    (an empty object-dtype array has no HDF5 equivalent), then an obs index
    written with an encoding the reader had no method for. The format is the
    library's to define, so every write here goes through the library.
    """
    import anndata as ad
    from anndata.experimental import concat_on_disk

    t0 = time.perf_counter()
    f, g, n_genes, n_cells = open_10x(source)
    label = f"brain{n // 1000}k"
    out = Path(out_dir) / f"{label}.h5ad"
    parts_dir = Path(out_dir) / f".{label}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    total_nnz = 0
    try:
        if n > n_cells:
            raise SystemExit(f"requested {n:,} cells, file has {n_cells:,}")
        rng = np.random.default_rng(seed)
        cols = np.sort(rng.choice(n_cells, size=n, replace=False)).astype(np.int64)

        barcodes = [b.decode() if isinstance(b, bytes) else str(b)
                    for b in g["barcodes"][:][cols]]
        raw_genes = g["gene_names"][:] if "gene_names" in g else g["genes"][:]
        genes = _unique([b.decode() if isinstance(b, bytes) else str(b)
                         for b in raw_genes])

        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            blk = gather_cells(g, cols[lo:hi], n_genes)
            a = ad.AnnData(X=blk)
            a.obs_names = barcodes[lo:hi]
            a.var_names = genes
            pth = parts_dir / f"part{lo:09d}.h5ad"
            a.write_h5ad(pth)
            parts.append(pth)
            total_nnz += int(blk.nnz)
            del blk, a
            print(f"    part {hi:,}/{n:,}", end="\r", flush=True)

        concat_on_disk(parts, out, axis=0, join="outer")

        # Provenance travels with the file, not a lab notebook.
        import h5py as _h5
        with _h5.File(out, "r+") as h:
            if "uns" not in h:
                h.create_group("uns")
        a = ad.read_h5ad(out, backed="r")
        del a
    finally:
        f.close()
        for pth in parts:
            pth.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            parts_dir.rmdir()

    rec = {
        "label": label, "path": str(out), "cells": n, "genes": n_genes,
        "nnz": total_nnz, "nnz_per_cell": round(total_nnz / n, 1),
        "density_pct": round(total_nnz / (n * n_genes) * 100, 3),
        "file_mb": round(out.stat().st_size / 1e6, 1),
        "build_s": round(time.perf_counter() - t0, 1),
        "parent": str(Path(source).name),
        "parent_sha256_prefix64mb": file_sha256(source),
        "subset_seed": seed, "subset_n": n,
        "generator": "benchmarks/make_subsets.py (streaming, concat_on_disk)",
    }
    print(" " * 44, end="\r")
    print(f"  {label}: {n:,} x {n_genes:,}, {total_nnz:,} nnz "
          f"({rec['nnz_per_cell']}/cell, {rec['density_pct']}%), "
          f"{rec['file_mb']} MB, {rec['build_s']}s [streamed]")
    return rec


def _unique(names: list[str]) -> list[str]:
    """anndata requires unique var_names; mirror var_names_make_unique."""
    seen: dict[str, int] = {}
    out = []
    for x in names:
        if x in seen:
            seen[x] += 1
            out.append(f"{x}-{seen[x]}")
        else:
            seen[x] = 0
            out.append(x)
    return out


def make_subset(source: str, n: int, seed: int, out_dir: str) -> dict:
    t0 = time.perf_counter()
    f, g, n_genes, n_cells = open_10x(source)
    try:
        if n > n_cells:
            raise SystemExit(f"requested {n:,} cells, file has {n_cells:,}")

        rng = np.random.default_rng(seed)
        cols = np.sort(rng.choice(n_cells, size=n, replace=False)).astype(np.int64)

        X = gather_cells(g, cols, n_genes, progress=True)

        barcodes = g["barcodes"][:][cols]
        gene_names = g["gene_names"][:] if "gene_names" in g else g["genes"][:]
    finally:
        f.close()

    import anndata as ad

    adata = ad.AnnData(X=X)
    adata.obs_names = [b.decode() if isinstance(b, bytes) else str(b) for b in barcodes]
    adata.var_names = [
        b.decode() if isinstance(b, bytes) else str(b) for b in gene_names
    ]
    adata.var_names_make_unique()

    prov = {
        "parent": str(Path(source).name),
        "parent_sha256_prefix64mb": file_sha256(source),
        "parent_cells": n_cells,
        "parent_genes": n_genes,
        "subset_seed": seed,
        "subset_n": n,
        "generator": "benchmarks/make_subsets.py",
    }
    adata.uns["polariseq_provenance"] = json.dumps(prov)

    label = f"brain{n // 1000}k"
    out = Path(out_dir) / f"{label}.h5ad"
    adata.write_h5ad(out)

    nnz = int(X.nnz)
    rec = {
        "label": label,
        "path": str(out),
        "cells": int(X.shape[0]),
        "genes": int(X.shape[1]),
        "nnz": nnz,
        "nnz_per_cell": round(nnz / X.shape[0], 1),
        "density_pct": round(nnz / (X.shape[0] * X.shape[1]) * 100, 3),
        "file_mb": round(out.stat().st_size / 1e6, 1),
        "build_s": round(time.perf_counter() - t0, 1),
        **prov,
    }
    print(" " * 40, end="\r")
    print(
        f"  {label}: {rec['cells']:,} x {rec['genes']:,}, "
        f"{nnz:,} nnz ({rec['nnz_per_cell']}/cell, {rec['density_pct']}%), "
        f"{rec['file_mb']} MB, {rec['build_s']}s"
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[50_000, 100_000, 200_000]
    )
    args = ap.parse_args()

    if not Path(args.source).exists():
        raise SystemExit(f"source not found: {args.source}")

    print(f"source: {args.source}")
    print(f"seed:   {args.seed}")
    records = [
        (make_subset_streaming if n > STREAM_ABOVE_CELLS else make_subset)(
            args.source, n, args.seed, args.out_dir
        )
        for n in args.sizes
    ]

    manifest = Path(args.out_dir) / "subsets_manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    by_label = {r["label"]: r for r in existing}
    by_label.update({r["label"]: r for r in records})
    manifest.write_text(json.dumps(sorted(by_label.values(), key=lambda r: r["cells"]), indent=2))
    print(f"\nmanifest: {manifest}")


if __name__ == "__main__":
    main()
