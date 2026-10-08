"""The human fetal atlas (Cao et al. 2020) as benchmark inputs.

Cao, J. et al. A human cell atlas of fetal gene expression. Science 370,
eaba7721 (2020). GEO GSE156793: 4,062,980 cells from 15 organs profiled by
sci-RNA-seq3, 63,561 genes, and the authors' annotation of every cell (77
main cell types, their organ, and their global UMAP coordinates).

    python benchmarks/fetal_prepare.py convert   # loom -> fetal4062k.h5ad + fetal_cells.csv.gz
    python benchmarks/fetal_prepare.py subsets   # fetal1000k, fetal2000k (nested, seed 0)
    python benchmarks/fetal_prepare.py all

Inputs are the GEO files, downloaded into ``benchmarks/results/fetal_data``:

    GSE156793_S3_gene_count.loom   (gunzipped from .loom.gz, 21.4 GB)
    GSE156793_S1_metadata_cells.txt.gz

The loom holds a dense genes x cells matrix in gzip-compressed 64 x 64
chunks. It is read in blocks of cells by worker processes and appended as
CSR rows (cells x genes) to one h5ad through anndata's own writer, so the
format stays the library's to define and no second copy is ever on disk.

The outputs are gzip-compressed h5ad (the laptop's disk holds the 4M file,
two subsets and an out-of-core spill of the largest at once only that way).
Both tools read compressed h5ad; on brain5k the compressed file is 3.4x
smaller and Polariseq's out-of-core run 6 % slower. The benchmarked files
carry the matrix and cell ids only, as the brain ladder does; the published
annotation lives in ``fetal_cells.csv.gz``, joined by cell id.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/results/fetal_data"
LOOM = DATA / "GSE156793_S3_gene_count.loom"
META = DATA / "GSE156793_S1_metadata_cells.txt.gz"
FULL = DATA / "fetal4062k.h5ad"
CELLS = DATA / "fetal_cells.csv.gz"
MANIFEST = DATA / "fetal_manifest.json"

#: Nested subsets: the first n cells of one seed-0 permutation, so each
#: smaller file's cells are inside the next.
SUBSETS = {"fetal1000k": 1_000_000, "fetal2000k": 2_000_000}
SEED = 0

#: Cells per worker read. A block is read dense (63,561 genes x BLOCK
#: float32 = 520 MB at 2048) and converted to CSR in the worker.
BLOCK = 2048
WORKERS = 3
#: HDF5 chunk and compression for the outputs.
H5_KW = {"compression": "gzip", "compression_opts": 4, "chunks": (1 << 18,)}

#: Published per-cell annotation kept for the biology comparison.
META_COLS = ["sample", "Organ", "Main_cluster_name", "Organ_cell_lineage", "Assay",
             "Development_day", "Fetus_id", "Sex", "All_reads", "Exon_reads",
             "Global_umap_1", "Global_umap_2"]


def _rel(p: Path) -> str:
    """Repo-relative where possible (a trial run may write elsewhere)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _unique(names: list[str]) -> list[str]:
    """Gene symbols, made unique, without inventing a name that already
    exists: this annotation has real genes called SNORD116-1, so the naive
    "append -1" rule collides with them (it left 8 duplicates in the first
    build). Suffixes advance until the name is free."""
    taken = set(names)
    seen: dict[str, int] = {}
    out = []
    for x in names:
        if x not in seen:
            seen[x] = 0
            out.append(x)
            continue
        while True:
            seen[x] += 1
            candidate = f"{x}-{seen[x]}"
            if candidate not in taken:
                break
        taken.add(candidate)
        out.append(candidate)
    return out


def _read_block(lo: int, hi: int) -> tuple[int, sp.csr_matrix]:
    with h5py.File(LOOM, "r") as f:
        dense = f["matrix"][:, lo:hi]               # genes x cells
    m = sp.csr_matrix(dense.T)                      # cells x genes
    m.indices = m.indices.astype(np.int32)
    m.data = m.data.astype(np.float32)
    return lo, m


def _write_empty_anndata(f: h5py.File, first: sp.csr_matrix) -> None:
    """X from the first block through anndata's writer, then an int64 indptr
    (the full matrix passes 2**31 non-zeros), ready for appends."""
    from anndata.io import write_elem

    write_elem(f, "X", first, dataset_kwargs=H5_KW)
    ip = f["X/indptr"][:].astype(np.int64)
    del f["X/indptr"]
    f["X"].create_dataset("indptr", data=ip, maxshape=(None,), chunks=(1 << 16,),
                          compression="gzip")


def _narrow_indptr(f: h5py.File) -> None:
    """Back to int32 when the final non-zero count fits it.

    The append path needs a 64-bit indptr in case the matrix passes 2**31
    non-zeros, but leaving it 64-bit where 32 would do makes the index
    arrays disagree — and scipy refuses a CSR whose indices and indptr have
    different integer types: `sc.pp.calculate_qc_metrics` dies in
    `eliminate_zeros` with "Output dtype not compatible with inputs". Above
    2**31 non-zeros the wide indptr is unavoidable, and anndata's own writer
    produces the same mixed pair (the 1.3M brain file is one), so a file
    that large cannot be preprocessed by scanpy without repairing it first.
    """
    ip = f["X/indptr"]
    if int(ip[-1]) < 2**31 and ip.dtype != np.int32:
        values = ip[:].astype(np.int32)
        del f["X/indptr"]
        f["X"].create_dataset("indptr", data=values, maxshape=(None,),
                              chunks=(1 << 16,), compression="gzip")


def _finish_anndata(f: h5py.File, obs_names: list[str], var: pd.DataFrame,
                    uns: dict) -> None:
    from anndata.io import write_elem

    _narrow_indptr(f)

    write_elem(f, "obs", pd.DataFrame(index=pd.Index(obs_names, name="cell")))
    write_elem(f, "var", var)
    for k in ("obsm", "varm", "obsp", "varp", "layers"):
        write_elem(f, k, {})
    write_elem(f, "uns", {"provenance": json.dumps(uns)})
    f.attrs["encoding-type"] = "anndata"
    f.attrs["encoding-version"] = "0.1.0"


def convert() -> dict:
    from anndata.io import sparse_dataset

    t0 = time.perf_counter()
    with h5py.File(LOOM, "r") as f:
        n_genes, n_cells = f["matrix"].shape
        cell_ids = [x.decode() for x in f["col_attrs/sample"][:]]
        symbols = _unique([x.decode() for x in f["row_attrs/gene_short_name"][:]])
        gene_ids = [x.decode() for x in f["row_attrs/gene_id"][:]]
        gene_types = [x.decode() for x in f["row_attrs/gene_type"][:]]
    print(f"loom: {n_genes:,} genes x {n_cells:,} cells")

    # The published annotation, aligned to the loom's cell order.
    meta = pd.read_csv(META, usecols=META_COLS).set_index("sample")
    missing = pd.Index(cell_ids).difference(meta.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} loom cells have no metadata row")
    meta = meta.loc[cell_ids]
    meta.index.name = "cell"
    meta.to_csv(CELLS)
    print(f"annotation: {CELLS.name}, {meta['Main_cluster_name'].nunique()} main types")

    var = pd.DataFrame({"gene_id": gene_ids, "gene_type": gene_types},
                       index=pd.Index(symbols, name="gene"))
    tmp = FULL.with_suffix(".h5ad.partial")
    nnz = 0
    starts = list(range(0, n_cells, BLOCK))
    with h5py.File(tmp, "w") as out, ProcessPoolExecutor(WORKERS) as pool:
        X = None
        # map() yields in submission order, so rows are appended in file order.
        for i, (lo, m) in enumerate(pool.map(_read_block, starts,
                                             [min(s + BLOCK, n_cells) for s in starts])):
            if X is None:
                _write_empty_anndata(out, m)
                X = sparse_dataset(out["X"])
            else:
                X.append(m)
            nnz += int(m.nnz)
            if i % 50 == 0:
                done = lo + m.shape[0]
                rate = done / (time.perf_counter() - t0)
                print(f"  {done:,}/{n_cells:,} cells, {nnz:,} nnz, "
                      f"{(n_cells - done) / rate / 60:.0f} min left", flush=True)
        uns = {"source": "GEO GSE156793 (Cao et al. 2020, Science 370:eaba7721)",
               "file": LOOM.name, "file_bytes": LOOM.stat().st_size,
               "cells": n_cells, "genes": n_genes, "nnz": nnz,
               "generator": "benchmarks/fetal_prepare.py convert"}
        _finish_anndata(out, cell_ids, var, uns)
    os.replace(tmp, FULL)
    rec = {**uns, "label": "fetal4062k", "path": _rel(FULL),
           "nnz_per_cell": round(nnz / n_cells, 1),
           "file_mb": round(FULL.stat().st_size / 1e6, 1),
           "build_s": round(time.perf_counter() - t0, 1)}
    print(json.dumps(rec, indent=1))
    return rec


def make_subset(label: str, n: int) -> dict:
    """The first `n` cells of the seed-0 permutation, in file order."""
    from anndata.io import read_elem, sparse_dataset

    t0 = time.perf_counter()
    out_path = DATA / f"{label}.h5ad"
    with h5py.File(FULL, "r") as src:
        n_cells = int(src["X"].attrs["shape"][0])
        rows = np.sort(np.random.default_rng(SEED).permutation(n_cells)[:n])
        indptr = src["X/indptr"][:]
        obs = read_elem(src["obs"])
        var = read_elem(src["var"])
        data_ds, idx_ds = src["X/data"], src["X/indices"]
        tmp = out_path.with_suffix(".h5ad.partial")
        nnz = 0
        with h5py.File(tmp, "w") as out:
            X = None
            # Gather in row windows of the source: one contiguous read per
            # window, then keep the selected rows, so the compressed chunks
            # are each decompressed once.
            win = 20_000
            for w0 in range(0, n_cells, win):
                w1 = min(w0 + win, n_cells)
                sel = rows[(rows >= w0) & (rows < w1)]
                if not len(sel):
                    continue
                a, b = int(indptr[w0]), int(indptr[w1])
                block = sp.csr_matrix(
                    (data_ds[a:b], idx_ds[a:b], (indptr[w0:w1 + 1] - a)),
                    shape=(w1 - w0, var.shape[0]))[sel - w0]
                block.indices = block.indices.astype(np.int32)
                if X is None:
                    _write_empty_anndata(out, block)
                    X = sparse_dataset(out["X"])
                else:
                    X.append(block)
                nnz += int(block.nnz)
            uns = {"source": "GEO GSE156793 (Cao et al. 2020)", "parent": FULL.name,
                   "subset_seed": SEED, "subset_n": n, "cells": n, "nnz": nnz,
                   "generator": "benchmarks/fetal_prepare.py subsets (nested)"}
            _finish_anndata(out, list(obs.index[rows]), var, uns)
    os.replace(tmp, out_path)
    rec = {**uns, "label": label, "path": _rel(out_path),
           "genes": int(var.shape[0]), "nnz_per_cell": round(nnz / n, 1),
           "file_mb": round(out_path.stat().st_size / 1e6, 1),
           "build_s": round(time.perf_counter() - t0, 1)}
    print(f"  {label}: {n:,} cells, {nnz:,} nnz, {rec['file_mb']:,} MB, {rec['build_s']} s")
    return rec


def verify(path: Path) -> None:
    """Both readers open it, and shapes agree."""
    import anndata as ad

    import polariseq as ps

    a = ad.read_h5ad(path, backed="r")
    lazy = ps.read_h5ad(str(path), lazy=True)
    assert (lazy.n_obs, lazy.n_vars) == a.shape, (lazy.shape, a.shape)
    print(f"  verified {path.name}: {a.shape[0]:,} x {a.shape[1]:,}")
    a.file.close()


def _manifest(update: dict) -> None:
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m.update(update)
    MANIFEST.write_text(json.dumps(m, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["convert", "subsets", "all"])
    a = ap.parse_args()
    if a.phase in ("convert", "all"):
        rec = convert()
        verify(FULL)
        _manifest({rec["label"]: rec})
    if a.phase in ("subsets", "all"):
        for label, n in SUBSETS.items():
            rec = make_subset(label, n)
            verify(DATA / f"{label}.h5ad")
            _manifest({label: rec})


if __name__ == "__main__":
    main()
