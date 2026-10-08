"""One Scarf run of the fixed pipeline, in Scarf's own environment.

Run by ``scarf_arm.py`` with the interpreter of ``.venv-scarf`` (Scarf 0.32
pins numpy 1.26 and numba <= 0.61, which do not support the Python 3.14 the
other arms use), as two invocations:

    python scarf_pipeline.py convert CONFIG.json   # h5ad -> Zarr, untimed
    python scarf_pipeline.py run CONFIG.json       # the timed pipeline

Same contract as the R arms: ``STEP <name> <begin> <end>`` on stderr on a
clock from process start, artifacts written after the last timed step, a
JSON summary on the last line of stdout.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

T0 = time.perf_counter()


@contextmanager
def step(name: str):
    b = time.perf_counter() - T0
    yield
    e = time.perf_counter() - T0
    print(f"STEP {name} {b:.4f} {e:.4f}", file=sys.stderr, flush=True)


def _consume_rows(self, batch_size: int):
    """Row batches of the h5ad matrix as COO, for ``H5adReader.consume``.

    Replaces Scarf 0.32.3's ``consume_group``, which reads
    ``indptr[i : i + batch]`` instead of ``[i : i + batch + 1]`` and so never
    writes the last cell (its own check then raises "All cells might not
    have been successfully written"), and which assumes CSR: a CSC-stored
    matrix (PBMC 3k) is read here whole and converted, which is fine at the
    sizes such files come in. Conversion only; the analysis is Scarf's.
    """
    import numpy as np
    from scipy.sparse import coo_matrix, csc_matrix

    grp = self.h5[self.matrixKey]
    n, m = self.nCells, self.nFeatures
    if grp.attrs.get("encoding-type") == "csc_matrix":
        x = csc_matrix((grp["data"][:], grp["indices"][:], grp["indptr"][:]),
                       shape=(n, m)).tocsr()
        for s in range(0, n, batch_size):
            yield x[s:s + batch_size].tocoo()
        return
    indptr = grp["indptr"]
    for s in range(0, n, batch_size):
        p = indptr[s:min(s + batch_size, n) + 1].astype(np.int64)
        rows = np.repeat(np.arange(len(p) - 1), np.diff(p))
        yield coo_matrix((grp["data"][p[0]:p[-1]], (rows, grp["indices"][p[0]:p[-1]])),
                         shape=(len(p) - 1, m))


def convert(cfg: dict) -> None:
    import scarf

    scarf.readers.H5adReader.consume_group = _consume_rows
    reader = scarf.H5adReader(cfg["h5ad_path"])
    writer = scarf.H5adToZarr(reader, zarr_loc=cfg["zarr_dir"], chunk_size=(2000, 1000))
    writer.dump(batch_size=2000)


def run(cfg: dict) -> None:
    import numpy as np

    with step("import"):
        import scarf

    if cfg.get("warm_zarr"):
        # Scarf compiles Numba code on first use. One untimed pass of the
        # same pipeline over a small synthetic store (12,000 cells, above
        # every size threshold) keeps compilation out of the timed steps, as
        # the Scanpy arm's warm-up does.
        with step("warmup"):
            w = scarf.DataStore(cfg["warm_zarr"], nthreads=cfg["threads"],
                                min_features_per_cell=0, min_cells_per_feature=0)
            w.filter_cells(attrs=["RNA_nFeatures"], lows=[10], highs=[np.inf])
            w.mark_hvgs(top_n=1000, blacklist="^$", show_plot=False)
            w.make_graph(feat_key="hvgs", dims=cfg["n_pcs"], k=cfg["n_neighbors"],
                         rand_state=cfg["seed"])
            w.run_leiden_clustering(resolution=cfg["resolution"], random_seed=cfg["seed"])
            if cfg.get("with_umap"):
                w.run_umap(n_epochs=20, parallel=True, nthreads=cfg["threads"],
                           random_seed=cfg["seed"])
            del w

    with step("load"):
        ds = scarf.DataStore(
            cfg["zarr_dir"], nthreads=cfg["threads"],
            min_features_per_cell=0, min_cells_per_feature=0,
        )
    with step("qc"):
        pass  # Scarf computes per-cell counts and features when it opens the store.
    with step("filter_cells"):
        ds.filter_cells(attrs=["RNA_nFeatures"], lows=[cfg["min_genes"] - 1],
                        highs=[np.inf])
    # Scarf normalises and log-transforms lazily inside make_graph; the
    # markers are kept, near zero, so the fusion is visible (arms/base.py).
    with step("normalize"):
        pass
    with step("log1p"):
        pass
    with step("hvg"):
        ds.mark_hvgs(top_n=cfg["n_hvg"], blacklist="^$", show_plot=False)
    with step("pca"):
        # PCA, the kNN search and the graph weights are one call in Scarf.
        ds.make_graph(feat_key="hvgs", dims=cfg["n_pcs"], k=cfg["n_neighbors"],
                      rand_state=cfg["seed"])
    with step("neighbors"):
        pass
    with step("leiden"):
        ds.run_leiden_clustering(resolution=cfg["resolution"], random_seed=cfg["seed"])
    with step("umap"):
        # Scarf's own UMAP of the same graph, on the same threads and for the
        # 200 epochs the other tools run (Scarf's default is 300; its other
        # settings, spread 2 and minimum distance 1, are left as shipped).
        if cfg.get("with_umap"):
            ds.run_umap(n_epochs=200, parallel=True, nthreads=cfg["threads"],
                        random_seed=cfg["seed"])

    a = Path(cfg["artifacts_dir"])
    a.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(ds.cells.fetch("RNA_leiden_cluster"), dtype=np.int64)
    np.save(a / "labels.npy", (labels - labels.min()).astype(np.int32))
    if cfg.get("with_umap"):
        np.save(a / "umap.npy", np.column_stack(
            [np.asarray(ds.cells.fetch(f"RNA_UMAP{k}"), dtype=np.float32) for k in (1, 2)]))
    names = np.asarray(ds.RNA.feats.fetch_all("ids"))
    hvg = np.asarray(ds.RNA.feats.fetch_all("I__hvgs"), dtype=bool)
    (a / "hvg.txt").write_text("\n".join(map(str, names[hvg])))
    (a / "obs_names.txt").write_text("\n".join(map(str, ds.cells.fetch("ids"))))
    print(json.dumps({
        "n_obs_after": int(len(labels)),
        "n_vars_after": int(hvg.sum()),
        "n_clusters": int(len(np.unique(labels))),
        "scarf_version": scarf.__version__,
    }))


if __name__ == "__main__":
    mode, cfg_path = sys.argv[1], sys.argv[2]
    cfg = json.loads(Path(cfg_path).read_text())
    {"convert": convert, "run": run}[mode](cfg)
