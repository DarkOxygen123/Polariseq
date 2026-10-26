#!/usr/bin/env python
"""Scarf arm of the showdown, with the released package (pip install scarf==0.32.3).

    python drv_scarf.py convert --input SHARDS --zarr DIR
    python drv_scarf.py run     --zarr DIR --out ART --threads 16 [--doublet-flags NPZ] [--doublets flag|remove]

convert  streams every .h5ad file into one Zarr store with Scarf's H5adReader and H5adToZarr. The files
         are read one after the other as one dataset; each cell's id is its global row. Scarf 0.32.3's
         consume_group drops the last cell of a file (it reads indptr[i:i+batch] where i:i+batch+1 is
         needed); the reader below is the corrected one of the earlier benchmarks (scarf_pipeline.py).
run      Scarf's own pipeline as its documentation gives it: DataStore (per-cell counts and genes
         computed when opened), filter_cells, mark_hvgs, make_graph (normalization, PCA and the kNN graph
         in one call), run_leiden_clustering, run_umap (200 rounds).

Scarf has no doublet detection. --doublet-flags takes the flags of the reference Scrublet
(doublets_ref.py, run and measured as its own add-on) and stores them as cell columns doublet_score and
doublet_flag, where a researcher reviews them; 'remove' then filters the flagged cells with filter_cells.

Artifacts in --out: global_rows.npy, labels.npy, umap.npy, hvg.txt.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, Steps, emit, shard_files  # noqa: E402


def _consume_rows(self, batch_size: int):
    """Row batches of one h5ad matrix (CSR) as COO, the last cell included."""
    import numpy as np
    from scipy.sparse import coo_matrix
    grp = self.h5[self.matrixKey]
    n, m = self.nCells, self.nFeatures
    indptr = grp["indptr"]
    for s in range(0, n, batch_size):
        p = indptr[s:min(s + batch_size, n) + 1].astype(np.int64)
        rows = np.repeat(np.arange(len(p) - 1), np.diff(p))
        yield coo_matrix((grp["data"][p[0]:p[-1]], (rows, grp["indices"][p[0]:p[-1]])),
                         shape=(len(p) - 1, m))


class Chain:
    """Several H5adReaders read as one dataset by H5adToZarr; cells are named by their global row."""

    def __init__(self, files: list[str]):
        import numpy as np
        import scarf
        self.r = [scarf.H5adReader(f) for f in files]
        ids = self.r[0].feat_ids()
        for x, f in zip(self.r[1:], files[1:]):
            if not np.array_equal(x.feat_ids(), ids):
                raise SystemExit(f"{f} does not have the genes of {files[0]}")
        self.nCells = sum(x.nCells for x in self.r)
        self.nFeatures = self.r[0].nFeatures
        self.matrixDtype = "uint32"     # counts, as Scarf stores 10x data

    def cell_ids(self):
        import numpy as np
        return np.arange(self.nCells).astype(str)

    def get_cell_columns(self):
        return iter(())

    def feat_ids(self):
        return self.r[0].feat_ids()

    def feat_names(self):
        return self.r[0].feat_names()

    def get_feat_columns(self):
        return self.r[0].get_feat_columns()

    def consume(self, batch_size: int):
        for x in self.r:
            yield from _consume_rows(x, batch_size)


def convert(a) -> int:
    st = Steps()
    with st("import"):
        import scarf
    files = shard_files(a.input)
    with st("convert"):
        chain = Chain(files)
        writer = scarf.H5adToZarr(chain, zarr_loc=a.zarr, chunk_size=(2000, 1000))
        writer.dump(batch_size=2000)
    emit({"tool": "scarf", "version": scarf.__version__, "phase": "convert", "cells_in": chain.nCells,
          "genes_in": chain.nFeatures, "zarr_gb": du_gb(a.zarr)})
    return 0


def run(a) -> int:
    st = Steps()
    with st("import"):
        import numpy as np
        import scarf
    with st("load"):
        ds = scarf.DataStore(a.zarr, nthreads=a.threads, min_features_per_cell=0,
                             min_cells_per_feature=P["min_cells"])
    summary = {"tool": "scarf", "version": scarf.__version__, "phase": "run", "threads": a.threads,
               "cells_in": int(ds.cells.N)}
    with st("qc"):
        ds.filter_cells(attrs=["RNA_nFeatures"], lows=[P["min_genes"] - 1], highs=[np.inf])
    summary["cells_after_qc"] = int(ds.cells.active_index("I").size)
    summary["genes_after_qc"] = int(ds.RNA.feats.active_index("I").size)

    if a.doublet_flags:
        with st("doublets_attach"):
            f = np.load(a.doublet_flags)
            ids = np.asarray(ds.cells.fetch_all("ids")).astype(np.int64)
            score = np.full(len(ids), np.nan, dtype=np.float32)
            flag = np.zeros(len(ids), dtype=np.float32)
            pos = np.searchsorted(ids, f["global_row"])     # ids are 0..n-1 in order
            score[pos] = f["score"]
            flag[pos] = f["predicted"].astype(np.float32)
            ds.cells.insert("doublet_score", score, overwrite=True)
            ds.cells.insert("doublet_flag", flag, overwrite=True)
        summary["doublets_flagged"] = int(flag[ds.cells.active_index("I")].sum())
        if a.doublets == "remove":
            with st("doublet_removal"):
                ds.filter_cells(attrs=["doublet_flag"], lows=[-1], highs=[0.5])
            summary["cells_after_doublets"] = int(ds.cells.active_index("I").size)

    with st("hvg"):
        ds.mark_hvgs(top_n=P["n_hvg"], blacklist="^$", show_plot=False)
    with st("pca"):    # normalization, PCA and the kNN graph are one call in Scarf
        ds.make_graph(feat_key="hvgs", dims=P["n_pcs"], k=P["neighbors"], rand_state=P["seed"])
    with st("leiden"):
        ds.run_leiden_clustering(resolution=P["resolution"], random_seed=P["seed"])
    if not a.no_umap:
        with st("umap"):
            ds.run_umap(n_epochs=P["umap_epochs"], parallel=True, nthreads=a.threads,
                        random_seed=P["seed"])

    art = Path(a.out)
    art.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(ds.cells.fetch("RNA_leiden_cluster"), dtype=np.int64)
    np.save(art / "labels.npy", (labels - labels.min()).astype(np.int32))
    np.save(art / "global_rows.npy", np.asarray(ds.cells.fetch("ids")).astype(np.int64))
    if not a.no_umap:
        np.save(art / "umap.npy", np.column_stack(
            [np.asarray(ds.cells.fetch(f"RNA_UMAP{k}"), dtype=np.float32) for k in (1, 2)]))
    names = np.asarray(ds.RNA.feats.fetch_all("ids"))
    hvg = np.asarray(ds.RNA.feats.fetch_all("I__hvgs"), dtype=bool)
    (art / "hvg.txt").write_text("\n".join(map(str, names[hvg])) + "\n")
    summary.update(cells_analyzed=int(len(labels)), hvg=int(hvg.sum()),
                   clusters=int(len(np.unique(labels))), zarr_gb=du_gb(a.zarr))
    emit(summary)
    return 0


def du_gb(path) -> float | None:
    try:
        return round(int(subprocess.run(["du", "-sk", str(path)], capture_output=True,
                                        text=True).stdout.split()[0]) * 1024 / 1e9, 2)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["convert", "run"])
    ap.add_argument("--input", help="folder, pattern or comma-separated .h5ad files (convert)")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", help="artifacts folder (run)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--doublet-flags", help="doublets_ref.py output (.npz)")
    ap.add_argument("--doublets", choices=["flag", "remove"], default="flag")
    ap.add_argument("--no-umap", action="store_true")
    a = ap.parse_args()
    if a.phase == "convert":
        import scarf.readers
        scarf.readers.H5adReader.consume_group = _consume_rows
        return convert(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
