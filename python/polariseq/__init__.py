"""Polariseq: high-performance omics data analysis.

A Rust-backed toolkit for high-dimensional biological datasets (cells × features),
with a clean Python API. Designed to run from low-resource laptops up to
multi-node GPU/HPU clusters via a swappable compute backend.

Quick start::

    import polariseq as ps
    adata = ps.read_h5ad("data.h5ad")          # read an .h5ad file
    ps.pp.pca(adata, n_components=50)           # PCA via Rust/faer
    ps.pp.neighbors(adata, n_neighbors=15)      # kNN graph
    ps.tl.leiden(adata, resolution=1.0)         # Leiden clustering

The ``AnnData``-like container holds the matrix and annotations; pipeline
functions mutate it in place (scanpy convention).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np

from polariseq import _results
from polariseq._frame import as_column as _as_column
from polariseq._frame import concat_frames as _concat_frames
from polariseq._frame import labels_column as _labels_column
from polariseq._frame import read_frame as _read_frame
from polariseq._frame import take as _take
from polariseq._polariseq import (
    DiskMatrix,
    SparseMatrix,
    core_version,
    fast_community,
    fuzzy_connectivities,
    fuzzy_connectivities_into,
    harmony_correct,
    leiden_cluster,
    neighbors_dense,
    nndescent_dense,
    pca_dense,
    pca_sparse,
    pca_sparse_scaled,
    read_h5ad_csr,
    read_h5ad_many,
    read_h5ad_metadata,
    read_h5ad_rows,
    read_h5ad_x,
    scan_10x_h5,
    scan_h5ad,
    scan_h5ad_genes,
    scan_h5ad_meta,
    triple,
    umap_embed,
    umap_kmeans_labels,
    umap_kmeans_start,
    umap_multilevel_layout,
    umap_spectral_layout,  # noqa: F401  (the spectral start alone, for diagnostics)
    validate_h5ad_file,  # noqa: F401  (public re-export)
)
from polariseq._polariseq import (
    disk_backed_preprocess as _disk_backed_preprocess,
)
from polariseq._polariseq import (
    read_10x_h5 as _read_10x_h5_raw,
)
from polariseq._polariseq import (
    read_10x_mtx as _read_10x_mtx_raw,
)
from polariseq._polariseq import (
    set_threads as _set_threads,
)
from polariseq._polariseq import (
    thread_count as _thread_count,
)
from polariseq._project import Project, ProjectNotSet, project
from polariseq._project import current as _current_project
from polariseq._project import require as _require_project
from polariseq.planning import estimate as _estimate
from polariseq.planning import max_cells as _max_cells
from polariseq.resources import Budget, Resources, detect_resources

__all__ = [
    "__version__",
    "AnnData",
    "SparseMatrix",
    "Budget",
    "resources",
    "budget",
    "set_budget",
    "set_workflow",
    "workflow",
    "BudgetWarning",
    "plan",
    "max_cells",
    "read_h5ad",
    "project",
    "Project",
    "ProjectNotSet",
    "DiskMatrix",
    "pp",
    "tl",
    "get",
    "set_resources",
    "get_build_info",
    "core_version",
    "triple",
]

__version__ = "0.1.1"

# The active budget: what Polariseq may use. Detected from the machine on
# first use and overridable with `set_budget`. Detection is lazy so that
# importing the package never shells out or touches the filesystem.
_BUDGET: Budget | None = None


#: Set once the worker pool has been sized; rayon can only be built once.
def _apply_threads(b: Budget) -> None:
    """Size the worker pool from the budget, and never fatally.

    Rayon's global pool is built on first use and cannot be resized, so a
    thread count set after some parallel work has already run cannot take
    effect. That is worth a warning — silently ignoring it would leave the
    user believing a setting they do not have — but not worth failing a
    pipeline over. Asking again for the count the pool already has is a
    no-op, so this runs every time a budget is set rather than only once:
    an earlier "only once" flag let `set_budget(threads=4)` return quietly
    after the detected count had been applied first (measured 2026-09-19:
    12 threads on a 12-thread laptop, 48 on a cluster node).
    """
    try:
        _set_threads(b.threads)
    except RuntimeError as exc:
        import warnings

        warnings.warn(
            f"could not set the worker pool to {b.threads} threads ({exc}); "
            f"continuing with {_thread_count()}",
            RuntimeWarning,
            stacklevel=3,
        )


#: Which defaults the steps use when an argument is left out (see :func:`set_workflow`).
WORKFLOWS = ("polariseq", "scanpy")
#: POLARISEQ_WORKFLOW=scanpy starts a process in the 'scanpy' workflow (benchmark harnesses, tests).
_WORKFLOW = os.environ.get("POLARISEQ_WORKFLOW", "polariseq")
if _WORKFLOW not in WORKFLOWS:
    raise ValueError(f"POLARISEQ_WORKFLOW must be one of {WORKFLOWS}, not {_WORKFLOW!r}")


def set_workflow(name: str) -> None:
    """Choose the defaults of the analysis steps.

    ``'polariseq'`` (the default): genes ranked as Scanpy's seurat flavour ranks them, among the genes
    detected in more than 1% of the cells; in the one-call pipelines, each cell normalized again over
    the selected genes (to 1,000, then log, as Scarf does) and each gene standardized before PCA; UMAP
    from Polariseq's multilevel spectral start with a minimum distance of 0.5 and 100 rounds above
    10,000 cells. On the human fetal atlas these clusters matched the published cell types far better
    (normalized mutual information 0.80 against 0.55) and the UMAP of 4 million cells was one coherent
    layout instead of 70 islands, in less time.

    ``'scanpy'``: the steps as Scanpy defines them, as in Polariseq before: the seurat ranking over all
    genes, no second normalization, UMAP from the spectral start with a minimum distance of 0.1 and 500
    rounds up to 10,000 cells, else 200. Use it to compare with Scanpy step for step.

    An argument given explicitly always wins over the workflow.
    """
    global _WORKFLOW
    if name not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {WORKFLOWS}, not {name!r}")
    _WORKFLOW = name


def workflow() -> str:
    """The active workflow (see :func:`set_workflow`)."""
    return _WORKFLOW


def budget() -> Budget:
    """The active :class:`~polariseq.resources.Budget`, detecting on first use."""
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = Budget.detect()
        _apply_threads(_BUDGET)
    return _BUDGET


def resources() -> Resources:
    """What this machine has: RAM (total and available), cores, spill space.

    ``available_source`` says how the available figure was obtained, because
    a measured number and an estimate deserve different trust::

        >>> ps.resources()
        Resources(ram=8.6 GB total, 3.1 GB available [vm_stat], cores=4P/8, ...)
    """
    return detect_resources(budget().spill_dir)


def set_budget(
    *,
    ram_gb: float | None = None,
    threads: int | None = None,
    spill_dir: str | None = None,
    spill_gb: float | None = None,
    allow_lossy: bool | None = None,
) -> Budget:
    """Override part of the budget; anything omitted stays as detected.

    Args:
        ram_gb: Ceiling for in-memory data structures.
        threads: Worker threads for parallel kernels.
        spill_dir: Directory for out-of-core scratch files.
        spill_gb: Ceiling for those files.
        allow_lossy: Permit the planner to trade accuracy for fit. **Off by
            default.** Polariseq will pick a slower exact plan over a faster
            approximate one unless asked, and reports any approximation it
            does make.

    Returns:
        The new active budget.
    """
    global _BUDGET
    # Not `budget()`: on first use that detects the machine *and* sizes the
    # worker pool from the detected cores, after which the pool cannot take
    # the thread count asked for here.
    base = _BUDGET if _BUDGET is not None else Budget.detect()
    _BUDGET = base.with_overrides(
        ram_gb=ram_gb,
        threads=threads,
        spill_dir=spill_dir,
        spill_gb=spill_gb,
        allow_lossy=allow_lossy,
    )
    if _BUDGET.spill_dir:
        import os

        os.makedirs(_BUDGET.spill_dir, exist_ok=True)
    _apply_threads(_BUDGET)
    return _BUDGET


def set_resources(
    max_ram_gb: float | None = None,
    max_ssd_gb: float | None = None,
    spill_dir: str | None = None,
) -> Budget:
    """Deprecated alias for :func:`set_budget`.

    Kept because it is documented in the beta quickstart. Note the change in
    behaviour: limits are now *detected* from the machine by default instead
    of defaulting to 8 GB regardless of what the machine has.
    """
    return set_budget(ram_gb=max_ram_gb, spill_gb=max_ssd_gb, spill_dir=spill_dir)


def plan(
    path: str | list[str] | dict[str, str],
    *,
    genes: str = "intersection",
    n_hvg: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    storage: str | None = None,
    with_umap: bool = True,
    verbose: bool = True):
    """Predict what analysing ``path`` will cost, before running anything.

    Scans the file's metadata (milliseconds, no matrix data), then costs each
    pipeline step against the active budget and picks a storage mode.

    Args:
        path: Dataset to plan for; or several, as a list of paths or a dict
            from each dataset's label to its path, planned as the one dataset
            :func:`read_h5ad` reads from them.
        genes: For several datasets, the genes kept: ``"intersection"``
            (default) or ``"union"`` (see :func:`read_h5ad`).
        n_hvg, n_pcs, n_neighbors: Pipeline settings that affect the cost.
        storage: Force ``"ram"``, ``"compressed"`` or ``"spill"``.
        with_umap: Include the UMAP step.
        verbose: Print the report as well as returning the plan.

    Returns:
        A :class:`~polariseq.planning.Plan` — per-step peaks, the limiting
        step, and whether it fits; for several datasets, each one's shape and
        the genes it keeps in ``datasets``.
    """
    if isinstance(path, (str, os.PathLike)):
        meta = scan_h5ad_meta(os.fspath(path))
        shape = (int(meta["n_cells"]), int(meta["n_genes"]), int(meta["nnz"]))
        datasets: list[dict[str, Any]] = []
    else:
        datasets, n_genes = _plan_datasets(path, genes)
        shape = (sum(d["n_cells"] for d in datasets), n_genes, sum(d["nnz"] for d in datasets))
    p = _estimate(
        *shape,
        budget(),
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        storage=storage,
        with_umap=with_umap,
        n_files=max(1, len(datasets)),
        densest_nnz_per_row=_densest_nnz_per_row(datasets),
    )
    p.datasets = datasets
    if verbose:
        print(p.report())
    return p


def _densest_nnz_per_row(datasets: list[dict[str, Any]]) -> int | None:
    """Values per cell of the densest of several files, which the planner
    charges the gene-selection pass at; ``None`` for a single file, whose
    blocks are planned at its average."""
    if len(datasets) < 2:
        return None
    return int(max(d["nnz"] / max(1, d["n_cells"]) for d in datasets))


def max_cells(
    *,
    n_genes: int = 30_000,
    nnz_per_cell: int = 2_000,
    storage: str | None = None,
    **kwargs: Any,
) -> int:
    """Largest cell count that fits the active budget, at this density.

    The scaling question the other way round — not "will this run?" but "how
    big can it get?"::

        >>> ps.max_cells()                      # in RAM on this machine
        >>> ps.max_cells(storage="spill")       # streamed from SSD
    """
    return _max_cells(
        budget(), n_genes=n_genes, nnz_per_cell=nnz_per_cell, storage=storage, **kwargs
    )


def get_build_info() -> dict[str, Any]:
    """Return a dict describing the linked Rust core."""
    def _release_triple(v: str) -> str:
        # Compare only the X.Y.Z release triple — pre-release labels differ
        # between PEP 440 ("0.1.0b1") and Cargo ("0.1.0-beta.1"/"0.1.0").
        import re
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
        return m.group(0) if m else v

    return {
        "python_version": __version__,
        "rust_core_version": core_version(),
        "matched": _release_triple(__version__) == _release_triple(core_version()),
    }


def _coerce_x(value: Any) -> Any:
    """Normalise a matrix assigned to ``AnnData.X``.

    Sparse input becomes a Rust-owned :class:`SparseMatrix` (one copy, then
    every pipeline step works on it in place); dense input stays a numpy array.
    """
    # A memory-mapped array (a result the Rust core wrote to the project)
    # stays one: its pages load when read.
    if isinstance(value, (SparseMatrix, DiskMatrix, np.memmap)) or value is None:
        return value
    if hasattr(value, "tocsr") and hasattr(value, "indptr"):
        return _sparse_from_scipy(value)
    return np.asarray(value)


def _sparse_from_scipy(x: Any) -> SparseMatrix:
    """Copy a scipy sparse matrix into a :class:`SparseMatrix` (single copy;
    ``int32`` indices are reinterpreted as ``uint32`` without a cast)."""
    x = x.tocsr()

    def _u32(a: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a)
        if a.dtype == np.int32:
            return a.view(np.uint32)
        return a.astype(np.uint32, copy=False)

    return SparseMatrix.from_arrays(
        np.ascontiguousarray(x.data, dtype=np.float32),
        _u32(x.indices),
        _u32(x.indptr),
        int(x.shape[0]),
        int(x.shape[1]),
    )


def _sparse_to_scipy(m: SparseMatrix) -> Any:
    """Copy a :class:`SparseMatrix` out to a ``scipy.sparse.csr_matrix``."""
    import scipy.sparse as sp

    data, indices, indptr = m.to_arrays()
    return sp.csr_matrix((data, indices.astype(np.int32), indptr.astype(np.int32)), shape=m.shape)


# Convenience methods on the Rust class so users never need the helpers above.
SparseMatrix.to_scipy = _sparse_to_scipy
SparseMatrix.from_scipy = staticmethod(_sparse_from_scipy)


class AnnData:
    """An in-memory cells × features container.

    A lightweight, opinionated alternative to ``anndata.AnnData``. Holds the data
    matrix (dense or sparse), per-cell (``obs``) and per-gene (``var``) metadata,
    and pipeline outputs (``obsm`` for embeddings, ``obsp`` for graphs, cluster
    labels in ``obs``).

    Attributes:
        X: The ``(n_obs × n_vars)`` data matrix — a numpy array (dense) or a
            Rust-owned :class:`SparseMatrix` (sparse). Assigning a scipy sparse
            matrix converts it once; ``adata.X.to_scipy()`` converts back.
        obs: Per-cell metadata (dict of column-name → array).
        var: Per-gene metadata (dict of column-name → array).
        obsm: Per-cell multi-dimensional annotations (e.g. ``X_pca``).
        obsp: Pairwise per-cell annotations (e.g. neighbor ``distances``).
        uns: Unstructured annotations.
    """

    #: The files the counts were read from, and which of their rows and genes
    #: this object holds (``None``: all, in order). An analysis normalized in
    #: memory reads its raw counts back from them (``tl.compare_conditions``).
    _source: dict[str, Any] | None = None
    _rows: np.ndarray | None = None
    _cols: np.ndarray | None = None

    def __init__(
        self,
        X: np.ndarray,
        obs: dict[str, np.ndarray] | None = None,
        var: dict[str, np.ndarray] | None = None,
        obs_names: list[str] | None = None,
        var_names: list[str] | None = None,
    ) -> None:
        self.X = X
        self.obs = obs or {}
        self.var = var or {}
        self.obs_names = obs_names or [f"cell_{i}" for i in range(self.n_obs)]
        self.var_names = var_names or [f"gene_{i}" for i in range(self.n_vars)]
        self.obsm: dict[str, np.ndarray] = {}
        self.obsp: dict[str, np.ndarray] = {}
        self.uns: dict[str, Any] = {}

    @property
    def X(self) -> Any:
        """The data matrix (numpy array or :class:`SparseMatrix`)."""
        return self._X

    @X.setter
    def X(self, value: Any) -> None:
        self._X = _coerce_x(value)

    @property
    def n_obs(self) -> int:
        """Number of observations (cells)."""
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        """Number of variables (features/genes)."""
        return self.X.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_obs, n_vars)``."""
        return (self.n_obs, self.n_vars)

    def __repr__(self) -> str:
        return (
            f"AnnData(n_obs={self.n_obs}, n_vars={self.n_vars}, "
            f"obs_keys={list(self.obs)}, obsm_keys={list(self.obsm)})"
        )

    def __getitem__(self, key: Any) -> AnnData:
        """A subset: ``adata[cells]`` or ``adata[cells, genes]``.

        Each axis takes a boolean mask (such as
        ``adata.obs["condition"] == "control"``), positions, names, or a
        slice. The result is a new object, independent of this one, holding
        the selected cells with their annotations, embeddings and graph
        (restricted to them). Out of core nothing is copied: the subset reads
        the same on-disk counts through its own filters, and cells and genes
        keep their stored order.
        """
        return _subset_by_key(self, key)

    def copy(self) -> AnnData:
        """An independent copy (out of core, of the view, not the counts)."""
        return _subset(self, None, None)

    def to_anndata(self) -> Any:
        """Convert to an ``anndata.AnnData``, for Scanpy and the tools built on it.

        ``X`` becomes a scipy CSR matrix (or stays a numpy array); ``obs``,
        ``var``, ``obsm``, ``obsp`` and ``uns`` are carried over. The result
        of :func:`process_diskbacked` never held the matrix: it converts with
        no ``X``, its per-gene statistics in ``var`` indexed by gene name,
        and every per-cell result. Requires the ``anndata`` package.
        """
        import anndata as ad
        import pandas as pd

        disk = bool((self.uns.get("pca") or {}).get("disk_backed"))
        var_cols = {k: _as_column(v) for k, v in self.var.items() if k != "name"}
        if disk:
            x = None
            names = self.var.get("name")
            n = len(next(iter(var_cols.values()))) if var_cols else 0
            var_index = [str(g) for g in names] if names is not None else [str(i) for i in range(n)]
        else:
            x = self.X
            x = None if _is_disk(x) else (x.to_scipy() if _is_sparse(x) else x)
            var_index = [str(g) for g in self.var_names]
        obs = pd.DataFrame({k: _as_column(v) for k, v in self.obs.items()},
                           index=pd.Index([str(o) for o in self.obs_names]))
        var = pd.DataFrame(var_cols, index=pd.Index(var_index))
        out = ad.AnnData(X=x, obs=obs, var=var)
        for k, v in self.obsm.items():
            out.obsm[k] = np.asarray(v)
        for k, v in self.obsp.items():
            out.obsp[k] = v
        out.uns.update(self.uns)
        batch_key = getattr(self, "batch_key", None)
        if batch_key in out.obs:
            # As anndata.concat labels datasets: categorical, in file order.
            out.obs[batch_key] = pd.Categorical(out.obs[batch_key], categories=self.labels)
        # Scanpy's conventions, so its tools (PAGA, plots by cluster) run on
        # the result as they would on Scanpy's own: clusterings as
        # categorical strings, and the neighbor graph's keys where Scanpy
        # looks for them.
        clusterings = {"louvain"}
        leiden = self.uns.get("leiden")
        if isinstance(leiden, dict):
            clusterings.add((leiden.get("params") or {}).get("key_added", "leiden"))
        clusterings |= {k for k, v in self.uns.items() if isinstance(v, dict) and "dendrogram" in v}
        for k in clusterings:
            if k in out.obs and np.issubdtype(out.obs[k].dtype, np.integer):
                v = out.obs[k].to_numpy()
                out.obs[k] = pd.Categorical(v.astype(str), categories=[str(c) for c in np.unique(v)])
        nb = self.uns.get("neighbors")
        if isinstance(nb, dict) and "connectivities" in out.obsp:
            params = {k: v for k, v in nb.items()
                      if k not in ("connectivities", "distances", "params",
                                   "connectivities_key", "distances_key")}
            scanpy_nb = {"connectivities_key": "connectivities",
                         "params": {"method": "umap", **params}}
            if "distances" in out.obsp:
                scanpy_nb["distances_key"] = "distances"
            out.uns["neighbors"] = scanpy_nb
        return out


class LazyAnnData:
    """Lazy-loading data container. Returns instantly with metadata;
    the data matrix materializes only when a compute operation needs it.

    Usage is identical to :class:`AnnData` — all ``ps.pp.*`` and ``ps.tl.*``
    functions work transparently. The difference: ``read_h5ad`` returns in
    milliseconds instead of minutes for large files.

    Attributes:
        source_path: Path to the backing file.
        source_format: 'h5ad', '10x_h5', or '10x_mtx'.
        n_obs: Number of cells (available immediately).
        n_vars: Number of genes (available immediately).
        est_memory_gb: Estimated RAM needed when materialized.
        X: The data matrix (triggers materialization on first access).
    """

    def __init__(self, source_path: str, source_format: str, *, mode: str = "auto",
                 name: str | None = None, obs_columns: list[str] | None = None) -> None:
        import time
        t0 = time.perf_counter()

        self.source_path = source_path
        self.source_format = source_format
        self._init_state()

        # Scan metadata (fast — reads only HDF5 attributes and small arrays).
        if source_format == "h5ad":
            meta = scan_h5ad(source_path)
            self._n_obs = meta["n_obs"]
            self._n_vars = meta["n_vars"]
            self._nnz = meta["n_elements"]
            self._encoding = meta["encoding"]
            self._est_memory_gb = meta["est_memory_gb"]
            self.obs_names = meta.get("obs_names") or [f"cell_{i}" for i in range(self._n_obs)]
            self.var_names = meta.get("var_names") or [f"gene_{i}" for i in range(self._n_vars)]
            # The file's annotations, compact, in every mode: out of core the
            # matrix is never loaded, so they cannot wait for it.
            self.obs.update(_read_frame(source_path, "obs", obs_columns))
            self.var.update(_read_frame(source_path, "var"))
            self._source = {"paths": [source_path], "maps": None,
                            "n_rows": self._n_obs, "n_cols": self._n_vars}
        elif source_format == "10x_h5":
            meta = scan_10x_h5(source_path)
            self._n_obs = meta["n_cells"]
            self._n_vars = meta["n_genes"]
            self._nnz = meta["nnz"]
            self._encoding = "csr"
            self._est_memory_gb = meta["est_memory_gb"]
            self.obs_names = meta.get("barcodes") or [f"cell_{i}" for i in range(self._n_obs)]
            self.var_names = meta.get("gene_names") or [f"gene_{i}" for i in range(self._n_vars)]
        elif source_format == "10x_mtx":
            # .mtx has no fast metadata scan — read the header line only.
            self._scan_mtx_header(source_path)
        else:
            raise ValueError(f"unknown format: {source_format}")

        # Where the analysis runs is decided now, from the file's shape and
        # the budget, before any data are read; every step follows it.
        self.name = name or _default_name(source_path)
        self.mode = _choose_mode(mode, self._n_obs, self._n_vars, self._nnz, source_format,
                                 source_path)
        self._scan_time = time.perf_counter() - t0

    def _init_state(self) -> None:
        """The empty containers of an object whose matrix is not yet read."""
        self._X: Any = None
        self._materialized = False
        self._closed = False
        self._temp_files: list[str] = []  # track files we create for cleanup
        self._mmap_handles: list[Any] = []  # track mmap handles
        self.obs: dict[str, Any] = {}
        self.var: dict[str, Any] = {}
        self.obsm: dict[str, np.ndarray] = {}
        self.obsp: dict[str, Any] = {}
        self.uns: dict[str, Any] = {}
        self._source: dict[str, Any] | None = None
        self._rows: np.ndarray | None = None
        self._cols: np.ndarray | None = None

    def _scan_mtx_header(self, dir_path: str) -> None:
        """Read just the matrix.mtx header line for dimensions."""
        from pathlib import Path
        p = Path(dir_path) / "matrix.mtx"
        if not p.exists():
            p = Path(dir_path) / "matrix.mtx.gz"
        if not p.exists():
            raise FileNotFoundError(f"no matrix.mtx in {dir_path}")

        import gzip
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt") as f:
            for line in f:
                if not line.startswith("%"):
                    parts = line.split()
                    self._n_vars = int(parts[0])  # genes
                    self._n_obs = int(parts[1])  # cells
                    self._nnz = int(parts[2])
                    break

        self._encoding = "mtx"
        self._est_memory_gb = self._nnz * 8 / 1e9
        self.obs_names = [f"cell_{i}" for i in range(self._n_obs)]
        # Try reading gene names from features.tsv/genes.tsv
        for fname in ["features.tsv", "genes.tsv", "features.tsv.gz", "genes.tsv.gz"]:
            fp = Path(dir_path) / fname
            if fp.exists():
                opener = gzip.open if str(fp).endswith(".gz") else open
                with opener(fp, "rt") as f:
                    self.var_names = [
                        line.split("\t")[1].strip() if "\t" in line else line.strip()
                        for line in f
                    ]
                break
        else:
            self.var_names = [f"gene_{i}" for i in range(self._n_vars)]

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_obs, self._n_vars)

    @property
    def est_memory_gb(self) -> float:
        return self._est_memory_gb

    @property
    def is_materialized(self) -> bool:
        return self._materialized

    def head(self, n: int = 10) -> dict:
        """Preview the first n cells — partial read, fast.

        Returns a dict with data/indices/indptr/n_cells/n_genes.
        For .h5ad this is a true partial read; 10x formats (mtx/h5) have no
        random-access layout, so they are materialized first (raw-count files
        are typically small).
        """
        if self.source_format == "h5ad":
            return self._read_rows(0, min(n, self._n_obs))
        self._materialize()
        X = self._X[: min(n, self._n_obs)]
        if _is_sparse(X):
            return {
                "data": X.data,
                "indices": X.indices,
                "indptr": X.indptr,
                "n_cells": X.shape[0],
                "n_genes": X.shape[1],
            }
        return {
            "data": np.asarray(X),
            "indices": None,
            "indptr": None,
            "n_cells": X.shape[0],
            "n_genes": X.shape[1],
        }

    def sample(self, n: int = 100, seed: int = 42):
        """Preview a random sample of cells as a scipy CSR matrix."""
        import scipy.sparse as sp
        if self.source_format != "h5ad":
            # No random access for 10x formats — materialize and slice.
            self._materialize()
            rng_ = np.random.default_rng(seed)
            idx = rng_.choice(self._n_obs, size=min(n, self._n_obs), replace=False)
            X = self._X[np.sort(idx)]
            return X if _is_sparse(X) else sp.csr_matrix(np.asarray(X))
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(self._n_obs, size=min(n, self._n_obs), replace=False))
        parts = []
        start = indices[0]
        prev = indices[0]
        for idx in list(indices[1:]) + [indices[-1] + 1]:
            if idx > prev + 1 or idx == indices[-1] + 1:
                if start <= prev:
                    result = self._read_rows(start, prev + 1)
                    parts.append(sp.csr_matrix(
                        (
                            np.asarray(result["data"], dtype=np.float32),
                            np.asarray(result["indices"], dtype=np.int32),
                            np.asarray(result["indptr"], dtype=np.int32),
                        ),
                        shape=(result["n_cells"], self._n_vars),
                    ))
                start = idx
            prev = idx
        return sp.vstack(parts).tocsr() if len(parts) > 1 else parts[0]

    def _read_rows(self, r0: int, r1: int) -> dict:
        """Rows ``[r0, r1)`` of the counts, as :func:`read_h5ad_rows` returns them."""
        return read_h5ad_rows(self.source_path, r0, r1)

    def _source_record(self) -> dict:
        """What identifies the counts: an on-disk copy recorded with the same
        record is current."""
        return {**_file_record(self.source_path), "n_obs": self._n_obs,
                "n_vars": self._n_vars, "nnz": self._nnz}

    def _copy_counts(self, prefix: str, block_rows: int) -> DiskMatrix:
        """Copy the counts into a block store at ``prefix``."""
        return DiskMatrix.from_h5ad(self.source_path, prefix, block_rows)

    def _materialize(self) -> None:
        """Load the full data matrix. This is the expensive operation."""
        if self._materialized:
            return

        if self.mode == "out-of-core":
            self._X = _open_out_of_core(self)
            self._materialized = True
            return

        import time
        t0 = time.perf_counter()
        print(
            f"[polariseq] Materializing {self.source_path.split('/')[-1]} "
            f"({self._n_obs:,} × {self._n_vars:,}, ~{self._est_memory_gb:.1f} GB)..."
        )

        # The one place a lazily-read matrix becomes resident, so the one
        # place the in-memory mode can say it will not fit the budget.
        _warn_if_over_budget(self._n_obs, self._n_vars, self._nnz, self.source_path)

        # Load eagerly to avoid recursive materialization.
        # Names were already read at scan time and may have been edited by
        # the user since; obs/var columns set in memory take precedence over
        # the file's, so materializing never clobbers earlier work.
        if self.source_format == "h5ad":
            # The annotations were read with the metadata; only X is new.
            self._X = _read_h5ad_matrix(self.source_path)
        elif self.source_format == "10x_h5":
            full = _tenx_dict_to_anndata(_read_10x_h5_raw(self.source_path))
            self._X = full.X
        elif self.source_format == "10x_mtx":
            full = _tenx_dict_to_anndata(_read_10x_mtx_raw(self.source_path))
            self._X = full.X
            if full.obs_names and len(full.obs_names) == self._n_obs:
                self.obs_names = full.obs_names

        self._materialized = True
        self._materialize_time = time.perf_counter() - t0
        print(f"[polariseq] Ready in {self._materialize_time:.1f}s")

    @property
    def X(self) -> Any:
        self._materialize()
        return self._X

    @X.setter
    def X(self, value: Any) -> None:
        self._X = _coerce_x(value)
        self._materialized = True

    @property
    def out_dir(self) -> Any:
        """This dataset's folder in the current project (``None`` without one):
        its on-disk copy of the counts and its intermediates."""
        proj = _current_project()
        return proj.run_dir(self.name) if proj is not None else None

    def to_anndata(self) -> Any:
        """Convert to an ``anndata.AnnData`` (see :meth:`AnnData.to_anndata`)."""
        return AnnData.to_anndata(self)

    def __repr__(self) -> str:
        status = "materialized" if self._materialized else f"lazy (~{self._est_memory_gb:.1f} GB on materialize)"
        return (
            f"LazyAnnData(n_obs={self._n_obs:,}, n_vars={self._n_vars:,}, "
            f"format={self.source_format}, mode={self.mode}, {status})"
        )

    def __getitem__(self, key: Any) -> AnnData:
        """A subset, as :meth:`AnnData.__getitem__`: the matrix is read (in
        memory) or opened (out of core) first."""
        return _subset_by_key(self, key)

    def copy(self) -> AnnData:
        """An independent copy, as :meth:`AnnData.copy`."""
        return _subset(self, None, None)

    def close(self) -> None:
        """Release all resources: free matrix memory, close mmaps, delete temp files.

        After close(), the object reverts to metadata-only state. Accessing
        ``adata.X`` will re-materialize from disk.
        """
        import gc

        # Free the matrix.
        self._X = None
        self._materialized = False

        # Close any mmap handles.
        for handle in self._mmap_handles:
            with contextlib.suppress(Exception):
                handle.close()
        self._mmap_handles.clear()

        # Delete temporary files we created.
        import os
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        self._temp_files.clear()

        # Clear cached embeddings and graphs (they may be large).
        self.obsm.clear()
        self.obsp.clear()

        gc.collect()
        self._closed = True

    def __enter__(self) -> LazyAnnData:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — automatically clean up."""
        self.close()

    def __del__(self) -> None:
        """Destructor — best-effort cleanup (may not run in all cases)."""
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass  # avoid errors during garbage collection

    def validate(self) -> dict[str, Any]:
        """Check data integrity without materializing. Returns a report."""
        import os
        from pathlib import Path

        report: dict[str, Any] = {"path": self.source_path, "format": self.source_format}
        report["exists"] = Path(self.source_path).exists() if self.source_format != "10x_mtx" else Path(self.source_path).exists()
        report["size_mb"] = 0

        if self.source_format == "10x_mtx":
            p = Path(self.source_path)
            total = sum(
                f.stat().st_size for f in p.glob("**/*") if f.is_file()
            )
            report["size_mb"] = total / 1e6
            report["has_matrix"] = (p / "matrix.mtx").exists() or (p / "matrix.mtx.gz").exists()
            report["has_barcodes"] = any((p / f).exists() for f in ["barcodes.tsv", "barcodes.tsv.gz"])
            report["has_features"] = any((p / f).exists() for f in ["features.tsv", "genes.tsv", "features.tsv.gz", "genes.tsv.gz"])
        else:
            if report["exists"]:
                report["size_mb"] = os.path.getsize(self.source_path) / 1e6

        report["n_obs"] = self._n_obs
        report["n_vars"] = self._n_vars
        report["nnz"] = self._nnz
        report["sparsity"] = 1 - self._nnz / max(1, self._n_obs * self._n_vars)
        report["encoding"] = self._encoding
        report["est_ram_gb"] = self._est_memory_gb
        report["scan_time_s"] = self._scan_time
        return report



class MergedLazyAnnData(LazyAnnData):
    """Several ``.h5ad`` files analyzed as one dataset: what :func:`read_h5ad`
    returns for a list of paths.

    The files' cells follow one another over one set of genes, matched by
    name, and each cell's dataset is recorded in ``obs[batch_key]``. The
    matrix is the one ``anndata.concat`` builds, genes in its order and each
    row's entries sorted by gene, but the files are never merged by hand: in memory they are read into one matrix,
    each row remapped as it arrives; out of core they are copied once, the
    same way, into one block store in the project folder. Every step then
    runs as it does on a single file.

    Attributes:
        sources: The files, in order.
        labels: Each file's dataset label, as recorded in ``obs[batch_key]``.
        batch_key: The ``obs`` column holding the labels.
        genes: The genes kept, ``"intersection"`` or ``"union"``.
    """

    def __init__(self, sources: list[tuple[str, str]], *, batch_key: str = "dataset",
                 genes: str = "intersection", mode: str = "auto",
                 name: str | None = None, obs_columns: list[str] | None = None) -> None:
        import time

        t0 = time.perf_counter()
        self._init_state()
        self.labels = [label for label, _ in sources]
        self.sources = [path for _, path in sources]
        self.source_path = None  # several files: see `sources`
        self.source_format = "h5ad"
        self.batch_key, self.genes = batch_key, genes
        metas = [scan_h5ad(p) for p in self.sources]
        for m, path in zip(metas, self.sources, strict=True):
            if m["encoding"] != "csr":
                raise NotImplementedError(
                    f"{path}: X is stored as {m['encoding']}; several files are read as one "
                    "from CSR matrices (anndata's default for sparse data)")
        self.var_names, self._maps = _merged_genes(
            [m.get("var_names") or [] for m in metas], genes, self.sources)
        self._counts = [int(m["n_obs"]) for m in metas]
        self._n_obs, self._n_vars = sum(self._counts), len(self.var_names)
        # The files' entries summed: exact for a union, and an upper bound for
        # an intersection until the matrix is read.
        self._nnz = sum(int(m["n_elements"]) for m in metas)
        self._encoding = "csr"
        self._est_memory_gb = self._nnz * 8 / 1e9
        self.obs_names = []
        for m in metas:
            self.obs_names.extend(m.get("obs_names") or [f"cell_{i}" for i in range(m["n_obs"])])
        # Each file's annotations, joined as anndata.concat joins them, then
        # each cell's dataset label.
        frames = [_read_frame(p, "obs", obs_columns, missing_ok=obs_columns is not None)
                  for p in self.sources]
        if obs_columns is not None:
            absent = [c for c in obs_columns if not any(c in f for f in frames)]
            if absent:
                raise KeyError(f"no file has the obs column(s) {absent}")
        self.obs.update(_concat_frames(frames, self._counts))
        if batch_key in self.obs:
            import warnings

            warnings.warn(f"obs['{batch_key}'] from the files is replaced by each cell's dataset "
                          "label; pass another batch_key to keep it", stacklevel=3)
        self.obs[batch_key] = _labels_column(self.labels, self._counts)
        self._source = {"paths": list(self.sources), "maps": self._maps,
                        "n_rows": self._n_obs, "n_cols": self._n_vars}
        self.name = name or _merged_name(self.labels)
        self.mode = _choose_mode(mode, self._n_obs, self._n_vars, self._nnz, "h5ad",
                                 self.sources[0])
        self._scan_time = time.perf_counter() - t0

    def _read_rows(self, r0: int, r1: int) -> dict:
        starts = np.cumsum([0, *self._counts])
        data, indices, indptr = [], [], [np.zeros(1, dtype=np.int64)]
        for k, path in enumerate(self.sources):
            a, b = max(r0, int(starts[k])), min(r1, int(starts[k + 1]))
            if a < b:
                rows = read_h5ad_rows(path, a - int(starts[k]), b - int(starts[k]))
                d, ix, ip = _remap_rows(rows, self._maps[k])
                data.append(d)
                indices.append(ix)
                indptr.append(ip[1:] + indptr[-1][-1])
        return {
            "data": np.concatenate(data) if data else np.zeros(0, np.float32),
            "indices": np.concatenate(indices) if indices else np.zeros(0, np.uint32),
            "indptr": np.concatenate(indptr),
            "n_cells": max(0, r1 - r0),
            "n_genes": self._n_vars,
        }

    def _source_record(self) -> dict:
        return {"sources": [_file_record(p) for p in self.sources], "genes": self.genes,
                "n_obs": self._n_obs, "n_vars": self._n_vars}

    def _copy_counts(self, prefix: str, block_rows: int) -> DiskMatrix:
        return DiskMatrix.from_h5ad_many(self.sources, self._maps, self._n_vars, prefix,
                                         block_rows)

    def _materialize(self) -> None:
        """Read the files as one matrix: in memory, or out of core as one
        block store in the project folder."""
        if self._materialized:
            return
        if self.mode == "out-of-core":
            self._X = _open_out_of_core(self)
            self._materialized = True
            return
        import time

        t0 = time.perf_counter()
        print(f"[polariseq] Materializing {len(self.sources)} datasets as one "
              f"({self._n_obs:,} × {self._n_vars:,}, ~{self._est_memory_gb:.1f} GB)...")
        _warn_if_over_budget(self._n_obs, self._n_vars, self._nnz, self.name)
        self._X = read_h5ad_many(self.sources, self._maps, self._n_vars, nnz_hint=self._nnz)
        self._nnz = int(self._X.nnz)
        self._materialized = True
        self._materialize_time = time.perf_counter() - t0
        print(f"[polariseq] Ready in {self._materialize_time:.1f}s")

    def validate(self) -> dict[str, Any]:
        """Check the files without reading their matrices."""
        return {
            "paths": list(self.sources),
            "labels": list(self.labels),
            "format": "h5ad",
            "exists": all(os.path.exists(p) for p in self.sources),
            "size_mb": sum(os.path.getsize(p) for p in self.sources if os.path.exists(p)) / 1e6,
            "n_obs": self._n_obs,
            "n_vars": self._n_vars,
            "nnz": self._nnz,
            "genes": self.genes,
            "est_ram_gb": self._est_memory_gb,
            "scan_time_s": self._scan_time,
        }

    def __repr__(self) -> str:
        status = ("materialized" if self._materialized
                  else f"lazy (~{self._est_memory_gb:.1f} GB on materialize)")
        return (f"LazyAnnData(n_obs={self._n_obs:,}, n_vars={self._n_vars:,}, "
                f"{len(self.sources)} datasets in obs['{self.batch_key}'], genes={self.genes}, "
                f"mode={self.mode}, {status})")


def _tenx_dict_to_anndata(result: dict) -> AnnData:
    """Convert a 10x reader result dict to an AnnData container."""
    return AnnData(
        X=result["matrix"],
        obs_names=result.get("barcodes") or [f"cell_{i}" for i in range(result["n_cells"])],
        var_names=result.get("gene_names") or [f"gene_{i}" for i in range(result["n_genes"])],
    )


def read_10x_h5(path: str, *, lazy: bool = True) -> AnnData | LazyAnnData:
    """Read a 10x Genomics ``.h5`` file (Cell Ranger format).

    Args:
        path: Path to the ``filtered_feature_bc_matrix.h5`` file.
        lazy: If True (default), return instantly with metadata only.
            The data matrix materializes when first accessed. Set False
            to load everything eagerly.

    Returns:
        A :class:`LazyAnnData` (default) or :class:`AnnData` if ``lazy=False``.
    """
    if lazy:
        return LazyAnnData(path, "10x_h5")
    result = _read_10x_h5_raw(path)
    return _tenx_dict_to_anndata(result)


def read_10x_mtx(dir: str, *, lazy: bool = True) -> AnnData | LazyAnnData:
    """Read a 10x MatrixMarket directory (``matrix.mtx`` + ``barcodes.tsv``
    + ``features.tsv`` or ``genes.tsv``).

    Args:
        dir: Directory containing the three files.
        lazy: If True (default), return instantly with metadata only.

    Returns:
        A :class:`LazyAnnData` (default) or :class:`AnnData` if ``lazy=False``.
    """
    if lazy:
        return LazyAnnData(dir, "10x_mtx")
    result = _read_10x_mtx_raw(dir)
    return _tenx_dict_to_anndata(result)


def read_h5ad(path: str | list[str] | dict[str, str], *, sparse: bool = True,
              lazy: bool = True, mode: str = "auto", name: str | None = None,
              batch_key: str = "dataset",
              genes: str = "intersection",
              obs_columns: list[str] | None = None) -> AnnData | LazyAnnData:
    """Read a `.h5ad` file, or several as one dataset. With ``lazy=True``
    (default), returns instantly with metadata; the data matrix materializes
    on first compute operation.

    By default, sparse X data is read into a Rust-owned :class:`SparseMatrix`
    (no densification, no copy into Python). Set ``sparse=False`` to force a
    dense numpy array.

    Args:
        path: Path to the ``.h5ad`` file; or several, as a list of paths or
            a dict from each dataset's label to its path. Several files are
            analyzed as one dataset: their cells one after another over one
            set of genes, matched by name, each cell labelled with its
            dataset (the file name, or the dict's key) in ``obs[batch_key]``.
            The matrix is the one ``anndata.concat`` builds (each row's
            entries sorted by gene), without the files being merged first.
        sparse: If True (default), return CSR for sparse files.
        lazy: If True (default), return instantly (metadata only).
        mode: Where the analysis runs. ``"auto"`` (default) lets the planning
            step choose from the file's shape and the budget: ``"memory"``
            when the matrix fits, else ``"out-of-core"``, in which the counts
            are copied once to the project folder and every step reads them
            in blocks. Either may be forced. The results are the same.
        name: The dataset's folder name in the project (default: the file
            name without its extension; for several files, their labels
            joined by ``+``).
        batch_key: For several files, the ``obs`` column of dataset labels.
        genes: For several files, the genes kept: ``"intersection"``
            (default; the genes every file has, in the first file's order) or
            ``"union"`` (every gene, zero where a file lacks it; sorted by
            name unless every file lists the same genes), as the inner and
            outer joins of ``anndata.concat`` order them.
        obs_columns: The file's cell annotations to read into ``obs``
            (default: all of them). Labels are kept as categoricals, one small
            integer per cell. For several files, a column one file lacks is
            missing for its cells.

    Returns:
        A :class:`LazyAnnData` (default) or :class:`AnnData` if ``lazy=False``.
    """
    if not isinstance(path, (str, os.PathLike)):
        if not sparse:
            raise ValueError("several files are read into a sparse matrix: drop sparse=False")
        if not lazy and mode == "out-of-core":
            raise ValueError("an out-of-core analysis is lazy: drop lazy=False")
        merged = MergedLazyAnnData(_sources(path), batch_key=batch_key, genes=genes,
                                   mode=mode if lazy else "memory", name=name,
                                   obs_columns=obs_columns)
        if lazy:
            return merged
        out = AnnData(X=merged.X, obs=merged.obs, obs_names=merged.obs_names,
                      var_names=merged.var_names)
        out._source = merged._source
        return out
    path = os.fspath(path)
    if lazy:
        return LazyAnnData(path, "h5ad", mode=mode, name=name, obs_columns=obs_columns)
    if mode == "out-of-core":
        raise ValueError("an out-of-core analysis is lazy: drop lazy=False")
    shape = scan_h5ad(path)
    if shape["n_elements"] > MAX_IN_MEMORY_VALUES:
        raise ValueError(_too_large_for_memory(shape["n_elements"], path))
    _warn_if_over_budget(shape["n_obs"], shape["n_vars"], shape["n_elements"], path)
    return _read_h5ad_eager(path, sparse, obs_columns)


def _read_h5ad_matrix(path: str, sparse: bool = True) -> Any:
    """The X of a `.h5ad`, in memory: CSR when sparse, else dense."""
    if sparse:
        try:
            return read_h5ad_csr(path)
        except RuntimeError:
            pass  # X is dense, not sparse
    return read_h5ad_x(path)[0]


def _read_h5ad_eager(path: str, sparse: bool = True,
                     obs_columns: list[str] | None = None) -> AnnData:
    """Load a `.h5ad` fully into memory, with no budget check (callers do it)."""
    meta = read_h5ad_metadata(path)
    x_data = _read_h5ad_matrix(path, sparse)
    out = AnnData(
        X=x_data,
        obs=_read_frame(path, "obs", obs_columns),
        var=_read_frame(path, "var"),
        obs_names=meta.get("obs_names", None),
        var_names=meta.get("var_names", None),
    )
    n, g = out.shape
    out._source = {"paths": [path], "maps": None, "n_rows": n, "n_cols": g}
    return out


class _Preprocessing:
    """The ``ps.pp`` namespace: preprocessing functions."""

    @staticmethod
    def _ensure_materialized(adata: AnnData | LazyAnnData) -> AnnData:
        """Auto-materialize a LazyAnnData before compute. No-op for AnnData."""
        if isinstance(adata, LazyAnnData):
            adata._materialize()
        return adata

    def scrublet(
        self,
        adata: AnnData | LazyAnnData,
        *,
        batch_key: str | None = None,
        sim_doublet_ratio: float = 2.0,
        expected_doublet_rate: float = 0.05,
        stdev_doublet_rate: float = 0.02,
        n_neighbors: int | None = None,
        n_prin_comps: int = 30,
        min_counts: float = 3,
        min_cells: int = 3,
        min_gene_variability_pctl: float = 85.0,
        threshold: float | None = None,
        random_state: int = 0,
    ) -> None:
        """Score each cell's chance of being a doublet, and call doublets.

        Scrublet (Wolock, Lopez & Klein, *Cell Systems* 2019), run in the Rust
        core as its reference implementation defines it: doublets are
        simulated by adding the counts of random pairs of cells, and each cell
        is scored by the share of simulated doublets among its nearest
        neighbours. The threshold separating doublets is found from the
        simulated doublets' scores unless given.

        Doublets form within one capture (one 10x channel), so score each on
        its own: pass the ``obs`` column that names it as ``batch_key``.
        Scores are computed from raw counts: in memory run this before
        ``normalize_total``; out of core the counts are read from the on-disk
        copy, so it can run at any point.

        Adds ``obs['doublet_score']``, ``obs['predicted_doublet']`` and
        ``uns['scrublet']`` (per batch: the threshold, the detected and
        estimated doublet rates, and the simulated doublets' scores, which
        :func:`polariseq.pl.scrublet_score_distribution` draws), under the
        names Scanpy's ``pp.scrublet`` uses. With a project folder the two
        columns are memory-mapped files in ``runs/<name>/results/scrublet/``.

        Args:
            adata: The data (mutated in place).
            batch_key: ``obs`` column of the capture each cell came from.
            sim_doublet_ratio: Doublets simulated per cell.
            expected_doublet_rate: Expected share of doublets.
            stdev_doublet_rate: Uncertainty of that share.
            n_neighbors: Neighbours per cell (default ``round(0.5 √n)``).
            n_prin_comps: Principal components of the embedding.
            min_counts, min_cells: A gene is used if it reaches ``min_counts``
                normalized counts in ``min_cells`` cells.
            min_gene_variability_pctl: Genes are used above this percentile
                of variability (the v-score).
            threshold: Score above which a cell is called a doublet.
            random_state: Seed of the simulation.
        """
        import pandas as pd

        from polariseq._polariseq import scrublet as _scrublet

        adata = self._ensure_materialized(adata)
        X = adata.X
        if not _is_disk(X):
            if "normalize" in adata.uns or "log1p" in adata.uns:
                raise ValueError("doublets are scored from raw counts: run ps.pp.scrublet before "
                                 "normalize_total and log1p (out of core it can run at any point)")
            if not _is_sparse(X):
                X = SparseMatrix.from_dense(np.ascontiguousarray(X, dtype=np.float32))
        n = adata.n_obs
        if batch_key is None:
            codes, labels = np.zeros(n, dtype=np.uint32), ["all"]
        else:
            if batch_key not in adata.obs:
                raise KeyError(f"batch_key {batch_key!r} is not in obs")
            col = adata.obs[batch_key]
            cat = col if isinstance(col, pd.Categorical) else pd.Categorical(np.asarray(col))
            c = np.asarray(cat.codes, dtype=np.int64)
            codes = np.where(c < 0, 2**32 - 1, c).astype(np.uint32)
            labels = [str(x) for x in cat.categories]
        scores = _results.new(adata, "scrublet", "doublet_score", (n,), np.float32)
        calls = _results.new(adata, "scrublet", "predicted_doublet", (n,), np.bool_)
        params = {"sim_doublet_ratio": sim_doublet_ratio,
                  "expected_doublet_rate": expected_doublet_rate,
                  "stdev_doublet_rate": stdev_doublet_rate, "n_neighbors": n_neighbors,
                  "n_prin_comps": n_prin_comps, "min_counts": min_counts, "min_cells": min_cells,
                  "min_gene_variability_pctl": min_gene_variability_pctl,
                  "threshold": threshold, "random_state": random_state}
        found = _scrublet(
            X, codes, len(labels), scores, calls,
            sim_doublet_ratio=float(sim_doublet_ratio),
            expected_doublet_rate=float(expected_doublet_rate),
            stdev_doublet_rate=float(stdev_doublet_rate), n_neighbors=n_neighbors,
            n_prin_comps=int(n_prin_comps), min_counts=float(min_counts),
            min_cells=int(min_cells), min_gene_variability_pctl=float(min_gene_variability_pctl),
            threshold=threshold, seed=int(random_state),
            # The selected genes of the batches scored together stay under a
            # quarter of the budget.
            max_batch_bytes=max(1 << 26, budget().ram_bytes // 4),
        )
        adata.obs["doublet_score"] = _results.view(scores)
        adata.obs["predicted_doublet"] = _results.view(calls)
        per = {}
        for b in found:
            label = labels[b.pop("batch")]
            b["doublet_scores_sim"] = _results.keep(adata, "scrublet", f"sim_{label}",
                                                    b["doublet_scores_sim"])
            note = b.pop("note")
            if note:
                import warnings

                warnings.warn(f"scrublet, batch {label!r}: {note}", stacklevel=2)
            per[label] = b
        if batch_key is None:
            adata.uns["scrublet"] = {**per.get("all", {}), "parameters": params}
        else:
            adata.uns["scrublet"] = {"batches": per, "batched_by": batch_key,
                                     "parameters": params}

    def calculate_qc_metrics(
        self,
        adata: AnnData,
        *,
        mito_prefix: str = "MT-",
    ) -> None:
        """Compute per-cell and per-gene quality control metrics.

        Adds to ``adata.obs``:
            - ``n_counts``: total UMI counts per cell
            - ``n_genes``: number of detected genes per cell
            - ``pct_counts_mt``: percentage of counts from mitochondrial genes

        Adds to ``adata.var``:
            - ``n_cells``: number of cells expressing each gene
            - ``total_counts``: total counts per gene
            - ``mean_counts``: mean expression per gene

        Args:
            adata: The data container (mutated in place).
            mito_prefix: Prefix for mitochondrial gene names (default "MT-").
        """
        adata = self._ensure_materialized(adata)

        var_names = list(adata.var_names) if adata.var_names else []
        mt_mask = None
        if var_names and mito_prefix:
            mt_mask = np.array(
                [str(n).startswith(mito_prefix) for n in var_names], dtype=bool
            )
            if not mt_mask.any():
                mt_mask = None

        mt_counts = None
        if _is_sparse(adata.X) or _is_disk(adata.X):
            # One fused pass in Rust: cell totals, detected genes, mito counts,
            # per-gene cell counts and totals.
            qc = adata.X.qc_metrics(mt_mask)
            n_counts = qc["n_counts"]
            n_genes = qc["n_genes"]
            n_cells_per_gene = qc["gene_n_cells"]
            total_counts_gene = qc["gene_total_counts"]
            mt_counts = qc["mt_counts"]
        else:
            n_counts = adata.X.sum(axis=1)
            n_genes = (adata.X > 0).sum(axis=1)
            n_cells_per_gene = (adata.X > 0).sum(axis=0)
            total_counts_gene = adata.X.sum(axis=0)
            if mt_mask is not None:
                mt_counts = adata.X[:, mt_mask].sum(axis=1)

        adata.obs["n_counts"] = np.asarray(n_counts, dtype=np.float32)
        adata.obs["n_genes"] = np.asarray(n_genes, dtype=np.int32)

        if mt_counts is not None:
            total = np.asarray(n_counts, dtype=np.float64).copy()
            total[total == 0] = 1
            adata.obs["pct_counts_mt"] = (np.asarray(mt_counts) / total * 100).astype(np.float32)

        adata.var["n_cells"] = np.asarray(n_cells_per_gene, dtype=np.int32)
        total_counts_gene = np.asarray(total_counts_gene, dtype=np.float64)
        adata.var["total_counts"] = total_counts_gene.astype(np.float32)
        adata.var["mean_counts"] = (total_counts_gene / adata.n_obs).astype(np.float32)

    def filter_cells(
        self,
        adata: AnnData,
        *,
        min_counts: float = 0,
        max_counts: float = np.inf,
        min_genes: int = 0,
        max_genes: int = np.inf,
        max_pct_mt: float = np.inf,
        inplace: bool = True,
    ) -> AnnData | None:
        """Filter cells based on QC metrics.

        Requires :meth:`calculate_qc_metrics` to have been run first.

        Args:
            adata: The data container.
            min_counts: Minimum total counts per cell.
            max_counts: Maximum total counts per cell.
            min_genes: Minimum detected genes per cell.
            max_genes: Maximum detected genes per cell.
            max_pct_mt: Maximum percentage of counts from mitochondrial genes
                (``obs['pct_counts_mt']``; needs mitochondrial genes in the data).
            inplace: If True (default), modify in place. Else return filtered copy.

        Returns:
            None if inplace, else a new filtered AnnData.
        """
        adata = self._ensure_materialized(adata)

        # Count and gene thresholds need the per-cell totals. Without them this
        # used to keep every cell silently, which a Scanpy user (whose
        # filter_cells computes them itself) would never expect.
        wants = min_counts > 0 or max_counts < np.inf or min_genes > 0 or max_genes < np.inf
        if wants and ("n_counts" not in adata.obs or "n_genes" not in adata.obs):
            self.calculate_qc_metrics(adata)

        mask = np.ones(adata.n_obs, dtype=bool)
        if "n_counts" in adata.obs:
            nc = np.asarray(adata.obs["n_counts"])
            mask &= (nc >= min_counts) & (nc <= max_counts)
        if "n_genes" in adata.obs:
            ng = np.asarray(adata.obs["n_genes"])
            mask &= (ng >= min_genes) & (ng <= max_genes)
        if max_pct_mt < np.inf:
            if "pct_counts_mt" not in adata.obs:
                raise KeyError(
                    "max_pct_mt needs obs['pct_counts_mt']: run calculate_qc_metrics "
                    "on data with mitochondrial genes first"
                )
            mask &= np.asarray(adata.obs["pct_counts_mt"]) <= max_pct_mt

        n_removed = (~mask).sum()
        if inplace:
            self._subset_cells(adata, mask)
            print(f"Filtered out {n_removed} cells ({adata.n_obs:,} remaining)")
            return None
        else:
            new = self._copy_subset_cells(adata, mask)
            print(f"Filtered out {n_removed} cells ({new.n_obs:,} remaining)")
            return new

    def filter_genes(
        self,
        adata: AnnData,
        *,
        min_cells: int = 0,
        min_counts: float = 0,
        inplace: bool = True,
    ) -> AnnData | None:
        """Filter genes based on detection across cells.

        Args:
            adata: The data container.
            min_cells: Minimum number of cells expressing the gene.
            min_counts: Minimum total counts for the gene.
            inplace: If True (default), modify in place.

        Returns:
            None if inplace, else a new filtered AnnData.
        """
        adata = self._ensure_materialized(adata)

        # Count on the matrix as it is now, as Scanpy does. var['n_cells'] from
        # an earlier calculate_qc_metrics still counts the cells filter_cells
        # has since removed, which kept genes Scanpy would drop.
        if _is_sparse(adata.X) or _is_disk(adata.X):
            qc = adata.X.qc_metrics(None)
            n_cells = np.asarray(qc["gene_n_cells"])
            total = np.asarray(qc["gene_total_counts"], dtype=np.float64)
        else:
            x = np.asarray(adata.X)
            n_cells = (x > 0).sum(axis=0)
            total = x.sum(axis=0, dtype=np.float64)
        adata.var["n_cells"] = n_cells.astype(np.int32)
        adata.var["total_counts"] = total.astype(np.float32)
        if "mean_counts" in adata.var:
            adata.var["mean_counts"] = (total / max(adata.n_obs, 1)).astype(np.float32)

        mask = np.ones(adata.n_vars, dtype=bool)
        if min_cells > 0:
            mask &= n_cells >= min_cells
        if min_counts > 0:
            mask &= total >= min_counts

        n_removed = (~mask).sum()
        if inplace:
            self._subset_genes(adata, mask)
            print(f"Filtered out {n_removed} genes ({adata.n_vars:,} remaining)")
            return None
        else:
            new = self._copy_subset_genes(adata, mask)
            print(f"Filtered out {n_removed} genes ({new.n_vars:,} remaining)")
            return new

    @staticmethod
    def _subset_cells(adata: AnnData, mask: np.ndarray) -> None:
        """Subset cells in place."""
        if _is_disk(adata.X):
            # Out of core the cell filter is a view: the store is not rewritten.
            adata.X.retain_rows_mask(np.ascontiguousarray(mask, dtype=np.bool_))
        elif _is_sparse(adata.X):
            if isinstance(adata.X, SparseMatrix):
                # In-place compaction: peak memory stays at ~1x the matrix
                # instead of the ~1.5x of a masked copy.
                adata.X.retain_rows_mask(np.ascontiguousarray(mask, dtype=np.bool_))
            else:
                adata.X = adata.X[mask, :]
        else:
            adata.X = adata.X[mask]
        adata.obs = {k: _take(v, mask) for k, v in adata.obs.items()}
        adata._rows = _keep_positions(getattr(adata, "_rows", None), np.flatnonzero(mask))
        adata.obsm = {k: v[mask] for k, v in adata.obsm.items()}
        if adata.obsp:
            for k, v in adata.obsp.items():
                if hasattr(v, "tocsr"):
                    adata.obsp[k] = v[mask, :][:, mask]
        if adata.obs_names:
            adata.obs_names = [n for n, m in zip(adata.obs_names, mask, strict=True) if m]
        # Update cached dims (LazyAnnData stores _n_obs).
        if hasattr(adata, "_n_obs"):
            adata._n_obs = int(mask.sum())

    @staticmethod
    def _copy_subset_cells(adata: AnnData, mask: np.ndarray) -> AnnData:
        """Return a new AnnData with subset cells."""
        return _subset(adata, np.flatnonzero(mask), None)

    @staticmethod
    def _subset_genes(adata: AnnData, mask: np.ndarray) -> None:
        """Subset genes in place."""
        if _is_disk(adata.X):
            adata.X.retain_cols_mask(np.ascontiguousarray(mask, dtype=np.bool_))
        else:
            adata.X = adata.X[:, mask]
        adata.var = {k: _take(v, mask) for k, v in adata.var.items()}
        adata._cols = _keep_positions(getattr(adata, "_cols", None), np.flatnonzero(mask))
        if adata.var_names:
            adata.var_names = [n for n, m in zip(adata.var_names, mask, strict=True) if m]
        # Update cached dims (LazyAnnData stores _n_vars).
        if hasattr(adata, "_n_vars"):
            adata._n_vars = int(mask.sum())

    @staticmethod
    def _copy_subset_genes(adata: AnnData, mask: np.ndarray) -> AnnData:
        """Return a new AnnData with subset genes."""
        return _subset(adata, None, np.flatnonzero(mask))

    def normalize_total(
        self,
        adata: AnnData,
        target_sum: float = 1e4,
    ) -> None:
        """Scale each cell's total counts to ``target_sum`` (scanpy ``pp.normalize_total``).

        Operates in place on ``adata.X``. Sparse-aware — keeps CSR format.

        Args:
            adata: The data container (mutated in place).
            target_sum: Target sum per cell (default 1e4).
        """
        adata = self._ensure_materialized(adata)
        if _is_sparse(adata.X) or _is_disk(adata.X):
            totals = adata.X.normalize_total(float(target_sum))  # in place (a view out of core)
            if _is_sparse(adata.X):
                # each cell's total, so that renormalize can turn values back into counts in place
                adata.uns.setdefault("normalize", {})["cell_totals"] = np.asarray(totals, dtype=np.float32)
        else:
            row_sums = adata.X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            adata.X = (adata.X * (target_sum / row_sums)).astype(np.float32)
        adata.uns.setdefault("normalize", {})["target_sum"] = target_sum

    def renormalize(
        self,
        adata: AnnData,
        target_sum: float = 1000.0,
        *,
        log1p: bool = True,
        counts: AnnData | None = None,
    ) -> None:
        """Normalize the current genes' raw counts again, over those genes only.

        Scarf's normalization (``renormalize_subset`` in its ``make_graph``):
        after the highly variable genes are selected (with ``subset=True``),
        each cell is scaled to ``target_sum`` by its total over the selected
        genes, then log-transformed. Scarf scales to 1,000 and then
        standardizes each gene, ``ps.pp.pca(adata, scale=True)``. On the human
        fetal atlas, Scarf's genes with this normalization gave clusters that
        matched the published cell types as well as Scarf's own.

        Out of core the normalization is a view over the stored counts, so it
        is simply replaced. In memory the counts were rewritten when they were
        normalized, but ``normalize_total`` records each cell's total, so the
        values are turned back into counts in place, without a copy. Pass
        ``counts`` (an object holding the raw counts of the same cells; genes
        matched by name) only when the cells changed after normalizing.

        Args:
            adata: The data container (mutated in place).
            target_sum: Total each cell is scaled to.
            log1p: Log-transform after scaling.
            counts: In memory, the raw counts when ``adata`` is already
                normalized.
        """
        adata = self._ensure_materialized(adata)
        if _is_disk(adata.X):
            # totals from the copy of the current genes that PCA reuses, not from the whole store
            adata.X.reset_normalization()
            adata.X.normalize_total_subset(float(target_sum), _subset_prefix(adata, adata.X))
            adata.uns.pop("normalize", None)
            adata.uns.pop("log1p", None)
            adata.uns["normalize"] = {"target_sum": target_sum, "over": "current genes"}
            if log1p:
                self.log1p(adata)
            return
        elif counts is None and _is_sparse(adata.X) and "cell_totals" in adata.uns.get("normalize", {}) \
                and len(adata.uns["normalize"]["cell_totals"]) == adata.n_obs:
            # in place, without a copy: values back to counts with each cell's original total
            old = adata.uns["normalize"]
            back = np.asarray(old["cell_totals"], dtype=np.float32) / np.float32(old["target_sum"])
            adata.X.renormalize_inplace(back, "log1p" in adata.uns, float(target_sum), bool(log1p))
            adata.uns.pop("log1p", None)
            adata.uns["normalize"] = {"target_sum": target_sum, "over": "current genes"}
            if log1p:
                adata.uns["log1p"] = {"base": None}
            return
        elif "normalize" in adata.uns or "log1p" in adata.uns:
            if counts is None:
                raise ValueError(
                    "the counts were rewritten by normalize_total and their totals are not recorded "
                    "(the cells changed after normalizing): pass counts=, an object with the raw "
                    "counts of the same cells")
            counts = self._ensure_materialized(counts)
            if counts.n_obs != adata.n_obs:
                raise ValueError(f"counts has {counts.n_obs:,} cells, adata {adata.n_obs:,}")
            adata.X = counts[:, list(map(str, adata.var_names))].X
        adata.uns.pop("normalize", None)
        adata.uns.pop("log1p", None)
        self.normalize_total(adata, target_sum=target_sum)
        if log1p:
            self.log1p(adata)
        adata.uns["normalize"]["over"] = "current genes"

    def log1p(self, adata: AnnData | LazyAnnData) -> None:
        """Apply ``log1p`` (natural log of 1+x) to ``adata.X`` in place.

        Preserves sparsity since ``log1p(0) = 0``.
        """
        adata = self._ensure_materialized(adata)
        if _is_sparse(adata.X) or _is_disk(adata.X):
            adata.X.log1p()  # in place (a view out of core)
        else:
            adata.X = np.log1p(adata.X).astype(np.float32)
        adata.uns["log1p"] = {"base": None}  # as Scanpy records it

    def highly_variable_genes(
        self,
        adata: AnnData,
        n_top_genes: int = 2000,
        *,
        flavor: str | None = None,
        subset: bool = False,
        min_cells: int | None = None,
        n_bins: int | None = None,
        lowess_frac: float = 0.1,
    ) -> None:
        """Select highly-variable genes (scanpy ``pp.highly_variable_genes``).

        ``flavor='seurat'`` implements scanpy's method step for step
        (statistics on ``expm1(X)``, log dispersion z-scored within 20 mean
        bins), so the selected set agrees with scanpy on the same
        log-normalized input. It stores ``highly_variable``, ``means``,
        ``dispersions`` and ``dispersions_norm`` in ``adata.var``.

        ``flavor='scarf'`` selects genes as Scarf does (Dhapola et al. 2022):
        genes are binned by mean expression (200 bins), a LOWESS curve is fitted
        through each bin's least variable gene, and genes are ranked by their
        variance over that floor, among genes detected in more than
        ``min_cells`` cells (default: 1% of the cells). It skips the rarely
        detected genes that the seurat ranking can favour, and stores
        ``highly_variable``, ``log_mean``, ``log_var``, ``corrected_var`` and
        ``detected`` in ``adata.var``.

        Args:
            adata: The data container (mutated in place).
            n_top_genes: Number of top HVGs to select (exact count).
            flavor: ``'seurat'`` or ``'scarf'``. Left out, the workflow decides
                (:func:`set_workflow`): ``'polariseq'`` ranks as ``'seurat'``
                among the genes detected in more than 1% of the cells,
                ``'scanpy'`` is ``'seurat'`` over every gene.
            subset: If True, subset the matrix to HVGs (updates X, var, var_names).
            min_cells: a gene must be detected in more cells than this. For
                ``'scarf'`` the default is ``int(0.01 * n_obs)``; for
                ``'seurat'`` the default is no floor (scanpy's behaviour), and a
                floor ranks only the genes above it, as if the others were not
                in the data.
            n_bins: Mean-expression bins (default 20 for ``'seurat'``, 200 for
                ``'scarf'``).
            lowess_frac: ``'scarf'`` only: the share of floor genes in each
                local fit of the trend.
        """
        if flavor is None:
            flavor = "seurat"
            if _WORKFLOW == "polariseq" and min_cells is None:
                min_cells = int(0.01 * adata.n_obs)
        if flavor not in ("seurat", "scarf"):
            raise NotImplementedError(
                f"flavor={flavor!r} is not implemented; 'seurat' and 'scarf' are available"
            )
        adata = self._ensure_materialized(adata)
        X = adata.X if (_is_sparse(adata.X) or _is_disk(adata.X)) else SparseMatrix.from_dense(
            np.ascontiguousarray(adata.X, dtype=np.float32)
        )
        if flavor == "scarf":
            floor = int(0.01 * adata.n_obs) if min_cells is None else int(min_cells)
            r = X.highly_variable_genes(int(n_top_genes), int(n_bins or 200), True, flavor="scarf",
                                        min_cells=floor, lowess_frac=float(lowess_frac))
            hvg_mask = np.asarray(r["mask"], dtype=bool)
            for key in ("log_mean", "log_var", "corrected_var", "detected"):
                adata.var[key] = np.asarray(r[key])
            adata.var["highly_variable"] = hvg_mask
            adata.uns["hvg"] = {"n_top_genes": int(r["n_hvg"]), "flavor": "scarf", "min_cells": floor,
                                "n_bins": int(n_bins or 200), "lowess_frac": float(lowess_frac)}
        else:
            floor = 0 if min_cells is None else int(min_cells)
            r = X.highly_variable_genes(int(n_top_genes), int(n_bins or 20), True, min_cells=floor)
            hvg_mask = np.asarray(r["mask"], dtype=bool)
            adata.var["means"] = np.asarray(r["means"])
            adata.var["dispersions"] = np.asarray(r["dispersions"])
            adata.var["dispersions_norm"] = np.asarray(r["dispersions_norm"])
            adata.var["highly_variable"] = hvg_mask
            adata.uns["hvg"] = {"n_top_genes": int(r["n_hvg"]), "flavor": "seurat", "min_cells": floor}

        # Auto-subset if requested (updates X, var, and var_names together).
        if subset:
            if _is_disk(adata.X):
                adata.X.retain_cols_mask(np.ascontiguousarray(hvg_mask, dtype=np.bool_))
            else:
                adata.X = adata.X[:, hvg_mask]
            adata.var = {k: np.asarray(v)[hvg_mask] for k, v in adata.var.items()}
            if adata.var_names:
                adata.var_names = [n for n, m in zip(adata.var_names, hvg_mask, strict=True) if m]
            if hasattr(adata, "_n_vars"):
                adata._n_vars = int(hvg_mask.sum())

    def pca(
        self,
        adata: AnnData,
        n_components: int = 50,
        *,
        center: bool = True,
        scale: bool = False,
        max_value: float | None = None,
        seed: int = 0,
    ) -> None:
        """Run PCA on ``adata.X``, storing the embedding in ``adata.obsm['X_pca']`.

        The Rust core adaptively chooses the algorithm: exact thin SVD for
        small matrices or when many components are requested, randomized SVD
        for large matrices with few components (7× faster than scanpy).

        Args:
            adata: The data container (mutated in place).
            n_components: Number of principal components.
            center: Whether to center columns (scanpy zero_center default).
            scale: Standardize each gene first, as ``sc.pp.scale`` before
                ``sc.tl.pca``. The scaled matrix is never formed, so a sparse
                ``X`` stays sparse and unchanged. Implies centering.
            max_value: With ``scale``, clip scaled values to
                ``±max_value`` (scanpy's ``max_value``).
            seed: Random seed for the randomized path.
        """
        if max_value is not None and not scale:
            raise ValueError("max_value only applies with scale=True")
        if scale and not center:
            raise ValueError("scale=True always centers; drop center=False")
        adata = self._ensure_materialized(adata)
        # The embedding is written by the core straight into its result
        # array: a memory-mapped file in the project when there is one.
        out = _results.new(adata, "pca", "X_pca", (adata.n_obs, n_components), np.float32)
        if _is_disk(adata.X):
            result = adata.X.pca(n_components, _subset_prefix(adata, adata.X), center or scale, 10, 4,
                                 seed, out=out, scale=bool(scale), max_value=max_value)
        elif scale and _is_sparse(adata.X):
            result = pca_sparse_scaled(adata.X, n_components, max_value, seed, out=out)
        elif scale:
            x = np.asarray(adata.X, dtype=np.float64)
            sd = x.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            x = (x - x.mean(axis=0)) / sd
            if max_value is not None:
                np.clip(x, -max_value, max_value, out=x)
            result = pca_dense(np.ascontiguousarray(x, dtype=np.float32), n_components, True, seed)
        elif _is_sparse(adata.X):
            # Randomized SVD on the Rust-owned matrix, by reference, with
            # implicit centering — the matrix is never densified. (Tiny
            # matrices fall back to an exact thin SVD inside the core.)
            result = pca_sparse(
                adata.X,
                n_components,
                center,
                10,  # oversample
                4,  # subspace iterations: matches arpack to < 1° on real data
                seed,
                out=out,
            )
        else:
            result = pca_dense(
                np.ascontiguousarray(adata.X, dtype=np.float32),
                n_components,
                center,
                seed,
            )
        if result["embedding"] is not None:  # the dense paths return theirs
            out[...] = result["embedding"]
        adata.obsm["X_pca"] = _results.view(out)
        adata.uns["pca"] = {
            "variance_ratio": result["variance_ratio"],
            "n_components": int(result["n_components"]),
        }

    def neighbors(
        self,
        adata: AnnData,
        n_neighbors: int = 15,
        *,
        use_rep: str = "X_pca",
        n_pcs: int | None = None,
        metric: str = "euclidean",
        approximate: bool | None = None,
        seed: int = 42,
    ) -> None:
        """Build a kNN graph from an embedding, storing it in ``adata.obsp``.

        For datasets with > 10,000 cells, automatically switches to NN-Descent
        (approximate kNN) which is O(n^1.2) instead of O(n²). Set
        ``approximate=False`` to force exact kNN, or ``True`` to force approximate.

        Args:
            adata: The data container (mutated in place).
            n_neighbors: Number of nearest neighbors per cell.
            use_rep: Which ``obsm`` key to use as the embedding.
            n_pcs: Number of PCs to use (default: all available).
            metric: Distance metric ('euclidean' or 'cosine').
            approximate: Force exact (False) or approximate (True) kNN.
                Default: auto (exact ≤ 10K cells, approximate above).
            seed: Random seed for NN-Descent.
        """
        adata = self._ensure_materialized(adata)
        embedding = adata.obsm[use_rep]
        if n_pcs is not None:
            embedding = embedding[:, :n_pcs]
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        n_obs = emb.shape[0]

        # Auto-switch: exact for small data, NN-Descent for large.
        use_approx = approximate if approximate is not None else (n_obs > 10_000)

        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine', not {metric!r}")

        if use_approx:
            # 0 iterations = auto (max(5, log2 n)); NN-Descent stops early once
            # fewer than 0.1% of neighbour entries change.
            # n_neighbors counts the cell itself (scanpy's convention, and
            # the exact path's), so the search asks for one fewer. Until
            # 2026-09-20 this path returned one neighbour more than scanpy.
            #
            # NN-Descent searches in Euclidean distance. For cosine, the rows
            # are scaled to unit length first — on unit vectors Euclidean
            # order is cosine order — and the distances are converted back to
            # the exact path's 1 - cos, since |a-b|^2 = 2 - 2cos. Until
            # 2026-09-26 this path ignored `metric` and still recorded it.
            search = emb
            if metric == "cosine":
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                search = np.ascontiguousarray(emb / np.maximum(norms, 1e-12),
                                              dtype=np.float32)
            # The graph is written by the core into its result arrays
            # (memory-mapped files in a project), reserved at k per cell.
            k = max(1, n_neighbors - 1)
            cap = n_obs * max(1, min(k, n_obs - 1))
            d_ip = _results.new(adata, "neighbors", "distances_indptr", (n_obs + 1,), np.uint32)
            d_ix = _results.new(adata, "neighbors", "distances_indices", (cap,), np.uint32)
            d_dx = _results.new(adata, "neighbors", "distances_data", (cap,), np.float32)
            nnz = nndescent_dense(search, k, 0, seed, out=(d_ip, d_ix, d_dx))["nnz"]
            if metric == "cosine":
                d_dx[:nnz] = (d_dx[:nnz].astype(np.float64) ** 2 / 2.0).astype(np.float32)
            method = "nndescent"
        else:
            result = neighbors_dense(emb, n_neighbors, metric)
            nnz = len(result["indices"])
            d_ip = _results.new(adata, "neighbors", "distances_indptr", (n_obs + 1,), np.uint32)
            d_ix = _results.new(adata, "neighbors", "distances_indices", (nnz,), np.uint32)
            d_dx = _results.new(adata, "neighbors", "distances_data", (nnz,), np.float32)
            d_ip[...], d_ix[...], d_dx[...] = result["indptr"], result["indices"], result["data"]
            method = "exact"
        d_ip, d_ix, d_dx = (_results.view(d_ip), _results.view(d_ix, nnz),
                            _results.view(d_dx, nnz))
        adata.obsp["distances"] = _csr_from_arrays(d_ip, d_ix, d_dx, n_obs)

        # scanpy parity: the symmetrised fuzzy connectivities (the graph
        # UMAP and clustering share), counted then written by the core
        # straight into their result arrays.
        conn: dict[str, np.ndarray] = {}

        def alloc(n_rows: int, n_entries: int):
            conn["indptr"] = _results.new(adata, "neighbors", "connectivities_indptr",
                                          (n_rows + 1,), np.uint32)
            conn["indices"] = _results.new(adata, "neighbors", "connectivities_indices",
                                           (n_entries,), np.uint32)
            conn["data"] = _results.new(adata, "neighbors", "connectivities_data",
                                        (n_entries,), np.float32)
            return conn["indptr"], conn["indices"], conn["data"]

        fuzzy_connectivities_into(_as_u32(d_ip), _as_u32(d_ix), d_dx, n_obs, alloc)
        adata.obsp["connectivities"] = _csr_from_arrays(
            _results.view(conn["indptr"]), _results.view(conn["indices"]),
            _results.view(conn["data"]), n_obs
        )
        adata.uns["neighbors"] = {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "use_rep": use_rep,
            "method": method,
            "connectivities": True,
        }

    def harmony_integrate(
        self,
        adata: AnnData,
        batch_key: str | list[str],
        *,
        use_rep: str = "X_pca",
        key_added: str = "X_pca_harmony",
        theta: float | list[float] = 2.0,
        sigma: float = 0.1,
        lamb: float | list[float] | None = None,
        n_clusters: int | None = None,
        tau: float = 0.0,
        block_size: float = 0.05,
        max_iter_harmony: int = 10,
        max_iter_kmeans: int = 4,
        epsilon_cluster: float = 1e-3,
        epsilon_harmony: float = 1e-2,
        alpha: float = 0.2,
        seed: int = 0,
    ) -> None:
        """Correct the PCA embedding for batch effects with Harmony.

        Harmony (Korsunsky et al. 2019) as R harmony 2 and harmonypy 2 run
        it, computed in Rust. Cells are assigned softly to clusters that are
        pushed to hold every batch in proportion; within each cluster a ridge
        regression estimates each batch's offset, and cells move back by their
        clusters' offsets. Clustering and correction alternate until the
        objective stops falling. The defaults are harmonypy's.

        The corrected embedding goes to ``adata.obsm[key_added]``
        (``X_pca_harmony``, as in Scanpy) and the uncorrected one stays in
        ``use_rep``, so mixing can be compared before and after. Build the
        graph on the corrected one::

            ps.pp.harmony_integrate(adata, "sample")
            ps.pp.neighbors(adata, use_rep="X_pca_harmony")

        With a project the result is a memory-mapped file, and the embedding
        is read where it lies.

        Args:
            adata: The data container (mutated in place).
            batch_key: Column of ``adata.obs`` naming each cell's batch, or a
                list of columns to correct for together (e.g. ``["donor",
                "lane"]``).
            use_rep: The ``obsm`` embedding to correct.
            key_added: The ``obsm`` key of the corrected embedding.
            theta: Diversity penalty, one for all batch variables or one each.
                Larger values push harder for clusters that mix the batches.
            sigma: Width of the soft clustering; smaller is closer to hard
                k-means.
            lamb: Ridge penalty, one for all variables or one each. ``None``
                estimates it per cluster from the expected number of cells
                (``alpha`` times it), as harmonypy 2 does.
            n_clusters: Number of clusters; default ``min(n_cells / 30, 100)``.
            tau: Protection of small batches against overcorrection (0 = off).
            block_size: Fraction of the cells reassigned at a time.
            max_iter_harmony: Maximum rounds of clustering and correction.
            max_iter_kmeans: Assignment rounds per clustering.
            epsilon_cluster: Convergence threshold of the assignment rounds.
            epsilon_harmony: Convergence threshold of the rounds of clustering
                and correction.
            alpha: Scale of the estimated ridge penalty.
            seed: Random seed.
        """
        from polariseq._compare import categorical

        keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
        if not keys:
            raise ValueError("batch_key names no column")
        codes, n_levels = [], []
        for key in keys:
            if key not in adata.obs:
                raise KeyError(f"batch_key '{key}' not in adata.obs")
            cat = categorical(adata.obs[key]).remove_unused_categories()
            c = np.asarray(cat.codes)
            if (c < 0).any():
                raise ValueError(f"adata.obs['{key}'] has missing values; give every cell a batch")
            codes.append(c.astype(np.uint32))
            n_levels.append(len(cat.categories))
        if all(n < 2 for n in n_levels):
            raise ValueError("Need >= 2 batches for correction")
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not in adata.obsm; run ps.pp.pca first")
        # A float32 C-ordered embedding (a memory-mapped result included) is
        # read as it is; anything else is converted once.
        embedding = np.ascontiguousarray(adata.obsm[use_rep], dtype=np.float32)
        if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs:
            raise ValueError(f"obsm['{use_rep}'] is not a cells x dimensions embedding")
        per_key = lambda v: [float(x) for x in np.atleast_1d(v)]  # noqa: E731
        out = _results.new(adata, "harmony", key_added, embedding.shape, np.float32)
        report = harmony_correct(
            embedding, codes, n_levels, out,
            theta=per_key(theta), sigma=float(sigma),
            lamb=None if lamb is None else per_key(lamb),
            n_clusters=n_clusters, tau=float(tau), block_size=float(block_size),
            max_iter_harmony=int(max_iter_harmony), max_iter_kmeans=int(max_iter_kmeans),
            epsilon_cluster=float(epsilon_cluster), epsilon_harmony=float(epsilon_harmony),
            alpha=float(alpha), batch_prop_cutoff=1e-5, seed=int(seed),
        )
        adata.obsm[key_added] = _results.view(out)
        adata.uns["harmony"] = {
            "batch_key": batch_key,
            "n_batches": n_levels[0] if len(n_levels) == 1 else n_levels,
            "use_rep": use_rep,
            "key_added": key_added,
            "n_clusters": int(report["n_clusters"]),
            "iterations": len(report["kmeans_rounds"]),
            "converged": bool(report["converged"]),
            "objective": list(report["objective"]),
            "kmeans_rounds": list(report["kmeans_rounds"]),
            "params": {"theta": theta, "sigma": sigma, "lamb": lamb, "tau": tau,
                       "block_size": block_size, "max_iter_harmony": max_iter_harmony,
                       "max_iter_kmeans": max_iter_kmeans, "alpha": alpha, "seed": seed},
        }


def _read_any(path: str) -> AnnData | LazyAnnData:
    """Read any supported format, auto-detecting .h5ad vs 10x .h5 vs .mtx.

    Detection is cheap: extension first, then (for bare .h5) a lazy metadata
    scan that fails fast on the wrong format.
    """
    import os
    if os.path.isdir(path) or path.endswith(".mtx") or path.endswith(".mtx.gz"):
        return read_10x_mtx(path)
    if path.endswith(".h5ad"):
        return read_h5ad(path)
    try:
        return read_h5ad(path)  # lazy metadata scan — raises fast for 10x files
    except (RuntimeError, ValueError):
        return read_10x_h5(path)


def merge_datasets(
    paths: list[str],
    *,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    batch_key: str = "dataset",
    seed: int = 42,
) -> AnnData:
    """Process multiple .h5ad datasets independently, merge PCA embeddings.

    This is the memory-efficient multi-study workflow: each dataset is loaded,
    preprocessed (normalize → log1p → HVG → PCA) one at a time, then ONLY the
    PCA embeddings are concatenated. Memory usage is bounded by the largest
    single dataset, not the combined total.

    Args:
        paths: Dataset paths — any mix of .h5ad, 10x .h5, and MatrixMarket
            .mtx directories (auto-detected per file).
        n_hvg: Number of HVGs per dataset.
        n_pcs: Number of PCs per dataset.
        batch_key: Column name for the batch/study identifier.
        seed: Random seed.

    Returns:
        An :class:`AnnData` with merged PCA embeddings and batch labels.
    """
    import warnings

    warnings.warn(
        "merge_datasets is deprecated: it fits a separate PCA to each file, so the "
        "embeddings it joins lie in different bases. Read the files as one dataset, "
        "ps.read_h5ad([...]), and correct with ps.pp.harmony_integrate(adata, 'dataset').",
        DeprecationWarning, stacklevel=2)
    embeddings = []
    cell_types_all = []
    batch_labels_all = []

    for i, path in enumerate(paths):
        print(f"  Processing {path} (dataset {i})...", flush=True)
        adata = _read_any(path)
        pp.normalize_total(adata)
        pp.log1p(adata)
        pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata.X = adata.X[:, adata.var["highly_variable"]]
        pp.pca(adata, n_components=n_pcs, seed=seed)

        # Read cell_type annotations from the original file via backed obs.
        ct_values = ["unknown"] * adata.n_obs
        try:
            import anndata as ad_tmp
            backed = ad_tmp.read_h5ad(path, backed="r")
            ct_col = None
            for col in ["cell_type", "celltype"]:
                if col in backed.obs.columns:
                    ct_col = col
                    break
            if ct_col:
                vals = backed.obs[ct_col].astype(str).tolist()
                if len(vals) == adata.n_obs:
                    ct_values = vals
            del backed
        except Exception:
            pass
        cell_types_all.extend(ct_values[:adata.n_obs])  # truncate to match

        emb = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        embeddings.append(emb)
        batch_labels_all.extend([i] * adata.n_obs)

        # Free the raw data, keep only embedding
        del adata
        import gc

        gc.collect()

    merged_emb = np.vstack(embeddings)
    del embeddings

    result = AnnData(
        X=np.zeros((merged_emb.shape[0], 1), dtype=np.float32),  # placeholder
        obs={
            batch_key: np.array(batch_labels_all, dtype=np.int32),
            "cell_type": np.array(cell_types_all),
        },
    )
    result.obsm["X_pca"] = merged_emb
    print(f"  Merged: {merged_emb.shape[0]:,} cells × {merged_emb.shape[1]} PCs "
          f"from {len(paths)} datasets")
    return result


class _Tools:
    """The ``ps.tl`` namespace: analysis tools."""

    def rank_genes_groups(
        self,
        adata: AnnData,
        groupby: str = "leiden",
        *,
        groups: list[str] | None = None,
        reference: str = "rest",
        n_genes: int = 50,
    ) -> None:
        """Rank genes by differential expression per group.

        Each group is compared with all other cells (``reference="rest"``)
        or with one group (``reference="control"``) by Welch's t-test, as
        Scanpy's ``rank_genes_groups(method="t-test")``. Results go to
        ``adata.uns['rank_genes_groups']``, one entry per group. To compare
        conditions within cell types with replicates, use
        :meth:`compare_conditions`.

        Args:
            adata: The data container (mutated in place).
            groupby: ``obs`` column of the groups: a clustering, or any
                labels (conditions, cell types).
            groups: Groups to rank (default: all but the reference).
            reference: ``"rest"``, or the group the others are compared with.
            n_genes: Number of top genes to store per group.
        """
        from polariseq._compare import categorical

        src = adata.uns.get("disk_backed")
        if src is not None:
            self._rank_genes_groups_file(adata, groupby, n_genes, src)
            return

        pp = _Preprocessing()
        pp._ensure_materialized(adata)

        if groupby not in adata.obs:
            raise KeyError(f"groupby '{groupby}' not in obs. Run clustering first.")
        cat = categorical(adata.obs[groupby])
        names = [str(c) for c in cat.categories]
        codes = np.asarray(cat.codes, dtype=np.int64)
        wanted = set(names if groups is None else map(str, groups))
        var_names = list(adata.var_names) if adata.var_names else [f"gene_{i}" for i in range(adata.n_vars)]
        X = adata.X if (_is_sparse(adata.X) or _is_disk(adata.X)) else SparseMatrix.from_dense(
            np.ascontiguousarray(adata.X, dtype=np.float32))

        if reference != "rest":
            if reference not in names:
                raise KeyError(f"reference {reference!r} is not a value of obs[{groupby!r}]")
            from polariseq._polariseq import compare_cells

            ref = names.index(reference)
            use = np.array([(n in wanted) or i == ref for i, n in enumerate(names)])
            label = np.where((codes >= 0) & use[np.maximum(codes, 0)], codes, 2**32 - 1)
            rows, _ = compare_cells(X, label.astype(np.uint32), 1, len(names), ref, min_cells=2)
            out = {}
            for code in np.unique(rows["condition"]):
                sel = np.flatnonzero(rows["condition"] == code)
                order = sel[np.argsort(-rows["t"][sel], kind="stable")][:n_genes]
                out[names[int(code)]] = {
                    "names": [var_names[int(j)] for j in rows["gene"][order]],
                    "scores": np.asarray(rows["t"][order]),
                    "logfoldchanges": np.asarray(rows["log_fc"][order]),
                    "pvals": np.asarray(rows["p_value"][order]),
                    "pvals_adj": np.asarray(rows["adj_p_value"][order]),
                    "pts": np.asarray(rows["pct_condition"][order]),
                }
            adata.uns["rank_genes_groups"] = out
            adata.uns["rank_genes_groups_params"] = {"groupby": groupby, "reference": reference}
            return

        labels = np.where(codes >= 0, codes, 2**32 - 1).astype(np.uint32)
        if _is_disk(X):
            result = X.rank_genes_groups(labels, len(names))
        else:
            from polariseq._polariseq import rank_genes_groups as _rgg
            result = _rgg(X, labels, len(names))
        result = {"groups": [dict(g, group_id=names[g["group_id"]]) for g in result["groups"]
                             if names[g["group_id"]] in wanted]}
        self._store_rank_genes(adata, result, var_names, n_genes)
        adata.uns["rank_genes_groups_params"] = {"groupby": groupby, "reference": reference}

    def _rank_genes_groups_file(self, adata: AnnData, groupby: str, n_genes: int,
                                src: dict) -> None:
        """``rank_genes_groups`` for a :func:`process_diskbacked` result.

        The matrix never entered memory, so the per-cluster statistics are
        streamed from the file, normalized and log-transformed as the analysis
        did, over the genes it kept; cells removed in quality control take no
        part. The test and the result are those of the in-memory function.
        """
        from polariseq._polariseq import rank_genes_groups_file

        if groupby not in adata.obs:
            raise KeyError(f"groupby '{groupby}' not in obs. Run clustering first.")
        from polariseq._compare import categorical
        cat = categorical(adata.obs[groupby])
        names = [str(c) for c in cat.categories]
        n_groups = len(names)
        codes = np.asarray(cat.codes, dtype=np.int64)
        labels = np.where(codes >= 0, codes, n_groups).astype(np.uint32)
        full = np.full(int(src["n_obs_file"]), n_groups, dtype=np.uint32)  # n_groups: left out
        full[np.asarray(adata.obs["file_row"], dtype=np.int64)] = labels
        keep = np.asarray(adata.var["kept"], dtype=bool) if "kept" in adata.var else None
        result = rank_genes_groups_file(
            src["path"], full, n_groups, float(src["target_sum"]),
            None if keep is None or keep.all() else keep,
        )
        gene_names = adata.var.get("name")
        var_names = ([str(g) for g in gene_names] if gene_names is not None
                     else [f"gene_{i}" for i in range(int(src["n_vars_file"]))])
        result = {"groups": [dict(g, group_id=names[g["group_id"]]) for g in result["groups"]
                             if g["group_id"] < n_groups]}
        self._store_rank_genes(adata, result, var_names, n_genes)
        adata.uns["rank_genes_groups_params"] = {"groupby": groupby, "reference": "rest"}

    @staticmethod
    def _store_rank_genes(adata: AnnData, result: dict, var_names: list, n_genes: int) -> None:
        """Store a ranking in ``uns['rank_genes_groups']``, Scanpy's layout."""
        uns_result = {}
        for g in result["groups"]:
            gid = g["group_id"]
            top_n = min(n_genes, len(g["gene_indices"]))
            gene_names = [var_names[i] for i in g["gene_indices"][:top_n]]
            uns_result[str(gid)] = {
                "names": gene_names,
                "scores": np.asarray(g["scores"][:top_n]),
                "logfoldchanges": np.asarray(g["logfoldchanges"][:top_n]),
                "pvals": np.asarray(g["pvals"][:top_n]),
                "pvals_adj": np.asarray(g["pvals_adj"][:top_n]),
                "pts": np.asarray(g["pts"][:top_n]),
            }
        adata.uns["rank_genes_groups"] = uns_result

    def compare_conditions(
        self,
        adata: AnnData | LazyAnnData,
        condition_key: str,
        reference: str,
        *,
        groupby: str | None = None,
        sample_key: str | None = None,
        conditions: list[str] | None = None,
        groups: list[str] | None = None,
        min_cells: int = 10,
        key_added: str = "compare_conditions",
    ) -> Any:
        """Compare conditions within each cell type: which genes change, and how much.

        With ``sample_key`` (recommended), cells are summed per replicate
        (for example the donor), condition and cell type, and the sums are
        tested as bulk samples, as edgeR and limma test them
        (``filterByExpr``, TMM, ``limma-trend``; reproduced in the Rust core
        and checked against edgeR 4.10 and limma 3.68). The replicates are
        the unit of evidence, so the p-values do not grow with the number of
        cells. A replicate measured under several conditions (a donor's cells
        before and after treatment) is compared with itself: the design
        blocks on it. The raw counts are read wherever they are: the matrix
        while it is raw, the on-disk copy, or, once normalized in memory, the
        files the data came from.

        Without ``sample_key``, each condition's cells are compared with the
        reference's by Welch's t-test on the normalized values, as Scanpy's
        ``rank_genes_groups(method="t-test")``. Cells of one donor are not
        independent, so these p-values overstate the evidence; use them to
        explore, and replicates to conclude.

        Args:
            adata: The data. ``uns[key_added]`` receives the table, one
                summary per cell type and the parameters.
            condition_key: ``obs`` column of the conditions.
            reference: The condition the others are compared with.
            groupby: ``obs`` column of the cell types (default: all cells as
                one group).
            sample_key: ``obs`` column of the biological replicates.
            conditions: Conditions to compare (default: all but the
                reference).
            groups: Cell types to compare (default: all).
            min_cells: A replicate's cells in a cell type below which it
                takes no part (with ``sample_key``); the cells each condition
                needs (without).
            key_added: Where in ``uns`` the results go.

        Returns:
            A ``pandas.DataFrame``, one row per cell type, condition and gene
            tested: ``log2_fold_change``, the mean expression of each side
            (log2 CPM for replicates, the normalized value for cells), the
            share of cells expressing the gene on each side, ``t``,
            ``p_value`` and ``adj_p_value`` (Benjamini-Hochberg within each
            cell type and condition), most significant first.
        """
        from polariseq._compare import compare_conditions as _compare_conditions

        adata = _Preprocessing._ensure_materialized(adata)
        return _compare_conditions(adata, condition_key, reference, groupby=groupby,
                                   sample_key=sample_key, conditions=conditions, groups=groups,
                                   min_cells=min_cells, key_added=key_added)

    def score_genes(
        self,
        adata: AnnData,
        gene_list: list[str],
        score_name: str | None = None,
        *,
        seed: int = 42,
    ) -> None:
        """Score cells by mean expression of a gene set.

        Computes per-cell mean expression of the genes in ``gene_list``,
        normalized by the mean expression of a random control gene set
        (Seurat's AddModuleScore approach).

        Args:
            adata: The data container.
            gene_list: List of gene names to score.
            score_name: Column name for the score (default: auto-generated).
            seed: Random seed for control set selection.
        """
        _in_memory_only(adata, "score_genes")
        pp = _Preprocessing()
        pp._ensure_materialized(adata)

        if score_name is None:
            score_name = f"score_{gene_list[0][:10]}"

        var_names = list(adata.var_names) if adata.var_names else []
        gene_idx = [var_names.index(g) for g in gene_list if g in var_names]
        if not gene_idx:
            raise ValueError(f"None of {gene_list[:3]}... found in var_names")

        # Mean expression of the gene set.
        if _is_sparse(adata.X):
            subset = adata.X[:, gene_idx]
            set_mean = np.asarray(subset.mean(axis=1)).flatten()
        else:
            set_mean = adata.X[:, gene_idx].mean(axis=1)

        # Control set: random genes matched by expression level.
        rng = np.random.default_rng(seed)
        n_control = max(50, len(gene_idx) * 5)
        all_idx = np.arange(adata.n_vars)
        control_idx = rng.choice(all_idx[~np.isin(all_idx, gene_idx)], size=min(n_control, adata.n_vars - len(gene_idx)), replace=False)

        if _is_sparse(adata.X):
            control_subset = adata.X[:, control_idx]
            control_mean = np.asarray(control_subset.mean(axis=1)).flatten()
        else:
            control_mean = adata.X[:, control_idx].mean(axis=1)

        adata.obs[score_name] = (set_mean - control_mean).astype(np.float32)

    def auto_annotate(
        self,
        adata: AnnData,
        groupby: str = "leiden",
        *,
        tissue: str | None = None,
        species: str = "human",
        method: str = "panglaodb",
    ) -> None:
        """Automatically assign cell type labels to clusters.

        Three annotation methods:
        - ``'panglaodb'`` (default): Score against PanglaoDB markers
          (178 cell types, 30 organs, human + mouse)
        - ``'markers'``: Score against our curated markers (~50 types)
        - ``'celltypist'``: ML-based via CellTypist (requires `pip install celltypist`)

        Args:
            adata: The data container.
            groupby: Column in obs identifying clusters.
            tissue: Restrict to specific tissue/organ markers.
            species: "human" or "mouse" (used by PanglaoDB).
            method: Annotation method ('panglaodb', 'markers', or 'celltypist').
        """
        if groupby not in adata.obs:
            raise KeyError(f"groupby '{groupby}' not in obs")

        if method == "celltypist":
            self._annotate_celltypist(adata, tissue)
            return

        if method == "panglaodb":
            from polariseq.panglaodb import get_panglaodb_markers
            # Map our tissue aliases to PanglaoDB organ names.
            organ_map = {
                "immune": "Immune system", "pbmc": "Immune system",
                "blood": "Immune system", "brain": "Brain", "neural": "Brain",
                "heart": "Heart", "cardiac": "Heart",
                "lung": "Lungs", "kidney": "Kidney", "pancreas": "Pancreas",
                "liver": "Liver", "skin": "Skin", "gut": "GI tract",
                "intestine": "GI tract", "reproductive": "Reproductive",
            }
            organ = organ_map.get(tissue.lower() if tissue else "")
            markers = get_panglaodb_markers(organ=organ, species=species)
        else:
            from polariseq.markers import get_all_markers, get_tissue_markers
            markers = get_tissue_markers(tissue) if tissue else get_all_markers()

        if not markers:
            raise ValueError(f"no markers found for tissue: {tissue}, method: {method}")

        _in_memory_only(adata, "auto_annotate")
        var_names = set(adata.var_names) if adata.var_names else set()
        cluster_labels = np.asarray(adata.obs[groupby])
        clusters = np.unique(cluster_labels)
        all_var_names = list(adata.var_names) if adata.var_names else []
        cluster_means: dict[int, dict[str, float]] = {}

        # LAZY PATH: If not yet materialized, compute cluster means via streaming
        # (never loads the full matrix — O(chunk) memory).
        if (isinstance(adata, LazyAnnData) and not adata.is_materialized
                and not isinstance(adata, MergedLazyAnnData)):
            from polariseq._polariseq import streaming_cluster_means as _stream_means
            n_clusters = int(clusters.max()) + 1
            # Map cluster labels to 0-indexed contiguous ids.
            cluster_to_idx = {int(c): i for i, c in enumerate(clusters)}
            labels_arr = np.array(
                [cluster_to_idx[int(c)] for c in cluster_labels], dtype=np.uint32
            )
            result = _stream_means(adata.source_path, labels_arr, n_clusters, 50_000)
            means_matrix = result["means"]  # (n_genes × n_clusters)

            for ci, c in enumerate(clusters):
                gene_scores = {}
                for gi, name in enumerate(all_var_names):
                    if gi < means_matrix.shape[0]:
                        gene_scores[name] = float(means_matrix[gi, ci])
                cluster_means[int(c)] = gene_scores
        else:
            # MATERIALIZED PATH: direct computation.
            for c in clusters:
                mask = cluster_labels == c
                if _is_sparse(adata.X):
                    mean_expr = np.asarray(adata.X[mask, :].mean(axis=0)).flatten()
                else:
                    mean_expr = adata.X[mask, :].mean(axis=0)
                gene_scores = {}
                for i, name in enumerate(all_var_names):
                    if i < len(mean_expr):
                        gene_scores[name] = float(mean_expr[i])
                cluster_means[int(c)] = gene_scores

        # Score each cluster against each cell type.
        # Markers are scored with per-gene z-scores ACROSS clusters (relative
        # specificity), not raw means: raw means let one broadly
        # highly-expressed marker set win for every cluster.
        cell_types = list(markers.keys())
        gene_names = all_var_names
        n_genes = len(gene_names)
        n_clusters = len(clusters)
        means_mat = np.zeros((n_genes, n_clusters), dtype=np.float64)
        for j, c in enumerate(clusters):
            cm = cluster_means[int(c)]
            for i, name in enumerate(gene_names):
                v = cm.get(name)
                if v is not None:
                    means_mat[i, j] = v
        mu = means_mat.mean(axis=1, keepdims=True)
        sd = means_mat.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        zmat = (means_mat - mu) / sd
        gene_pos = {name: i for i, name in enumerate(gene_names)}

        annotation_scores = {}
        assignments = {}
        for j, c in enumerate(clusters):
            scores = {}
            for ct in cell_types:
                ct_markers = [m for m in markers[ct] if m in var_names]
                if not ct_markers:
                    scores[ct] = 0.0
                    continue
                z_vals = [zmat[gene_pos[m], j] for m in ct_markers
                          if m in gene_pos]
                if not z_vals:
                    scores[ct] = 0.0
                    continue
                coverage = len(ct_markers) / len(markers[ct])
                scores[ct] = float(np.mean(z_vals)) * coverage
            best = max(scores, key=scores.get)
            assignments[int(c)] = best
            annotation_scores[int(c)] = scores

        cell_type_labels = np.array(
            [assignments[int(c)] for c in cluster_labels], dtype=object
        )
        adata.obs["cell_type"] = cell_type_labels
        adata.uns["annotation"] = {
            "assignments": assignments,
            "scores": annotation_scores,
            "tissue": tissue,
            "method": method,
        }

    def _annotate_celltypist(
        self,
        adata: AnnData,
        tissue: str | None = None,
    ) -> None:
        """Annotate using CellTypist (requires celltypist package)."""
        try:
            import celltypist
        except ImportError:
            raise ImportError(
                "CellTypist requires: pip install celltypist"
            ) from None

        _in_memory_only(adata, "CellTypist annotation")
        pp = _Preprocessing()
        pp._ensure_materialized(adata)

        # Convert our AnnData to a real anndata.AnnData for CellTypist.
        import anndata as ad
        import pandas as pd

        obs_index = adata.obs_names or [f"cell_{i}" for i in range(adata.n_obs)]
        var_index = adata.var_names or [f"gene_{i}" for i in range(adata.n_vars)]
        obs_df = pd.DataFrame(adata.obs, index=obs_index)
        var_df = pd.DataFrame(adata.var, index=var_index)
        X = adata.X.to_scipy() if _is_sparse(adata.X) else adata.X
        real_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        # Select model based on tissue.
        model_map = {
            "immune": "Immune_All_Low.pkl",
            "pbmc": "Immune_All_Low.pkl",
            "blood": "Immune_All_Low.pkl",
            "brain": "Adult_Human_MTG.pkl",
            "heart": "Healthy_Adult_Heart.pkl",
            "lung": "Human_Lung_Atlas.pkl",
            "gut": "Cells_Intestinal_Tract.pkl",
            "skin": "Adult_Human_Skin.pkl",
            "liver": "Healthy_Human_Liver.pkl",
            "pancreas": "Adult_Human_PancreaticIslet.pkl",
        }
        model_name = model_map.get(tissue.lower() if tissue else "", "Immune_All_Low.pkl")

        # Run annotation.
        predictions = celltypist.annotate(
            real_adata, model=model_name, majority_voting=True
        )
        predictions.to_adata(real_adata)

        # Copy results back to our AnnData.
        label_col = "majority_voting" if "majority_voting" in real_adata.obs else "predicted_labels"
        if label_col in real_adata.obs:
            adata.obs["cell_type"] = np.asarray(
                real_adata.obs[label_col].astype(str).values, dtype=object
            )

        adata.uns["annotation"] = {
            "method": "celltypist",
            "model": model_name,
            "tissue": tissue,
        }

    def annotate(self, adata: AnnData, mapping: dict, groupby: str = "leiden", *,
                 key_added: str = "cell_type", unassigned: str | None = None) -> None:
        """Name clusters, ``{"0": "CD4 T cell"}`` or ``{"B cell": ["4", "9"]}``; clusters given one
        name are joined. See :func:`polariseq._curation.annotate`."""
        from polariseq._curation import annotate
        annotate(adata, mapping, groupby, key_added=key_added, unassigned=unassigned)

    def merge_clusters(self, adata: AnnData, groups: list, groupby: str = "leiden", *,
                       key_added: str | None = None, names: list[str] | None = None) -> None:
        """Join clusters into one, ``[["3", "7"]]`` becomes ``"3+7"``.
        See :func:`polariseq._curation.merge_clusters`."""
        from polariseq._curation import merge_clusters
        merge_clusters(adata, groups, groupby, key_added=key_added, names=names)

    def subcluster(self, adata: AnnData, cluster: Any, groupby: str = "leiden", *,
                   resolution: float = 1.0, seed: int = 0, key_added: str | None = None) -> int:
        """Split one cluster by Leiden on its own cells, ``"5"`` becomes ``"5.0"``, ``"5.1"``, ...;
        returns the number of parts. See :func:`polariseq._curation.subcluster`."""
        from polariseq._curation import subcluster
        return subcluster(adata, cluster, groupby, resolution=resolution, seed=seed, key_added=key_added)

    def paris(
        self,
        adata: AnnData,
        n_clusters: int | None = None,
        *,
        balanced: bool = False,
        min_size: int | None = None,
        max_size: int | None = None,
        max_distance_fc: float = 2.0,
        key_added: str = "paris",
        recompute: bool = False,
    ) -> None:
        """Paris hierarchical clustering of the neighbour graph.

        Paris (Bonald et al. 2018) builds the whole cluster hierarchy of
        ``obsp['connectivities']`` in one pass: a dendrogram in which every
        cut is a clustering, from one cluster down to single cells. Coarse
        types and their sub-types are then cuts of the same tree, without
        clustering again. The dendrogram (``(n - 1) x 4``: children, height,
        size, as scipy's ``linkage``) is stored in
        ``uns[key_added]['dendrogram']`` and reused while the graph is
        unchanged, so clusterings at many resolutions cost one hierarchy.

        With ``n_clusters``, the labels of a straight cut into (at least)
        that many clusters go to ``obs[key_added]`` (0-based, by decreasing
        size). With ``balanced=True``, Scarf's balanced cut: clusters of at
        most ``max_size`` cells, whose two halves are joined only while their
        heights stay within a factor ``max_distance_fc`` once both exceed
        ``min_size`` cells.

        The dendrogram is identical to scikit-network's ``Paris`` and the
        balanced cut to Scarf's (``tests/test_paris.py``).

        Args:
            adata: The data container, after ``ps.pp.neighbors()``.
            n_clusters: Straight cut into this many clusters (or more, where
                merge heights tie).
            balanced: Use Scarf's balanced cut instead.
            min_size: Balanced cut: size above which heights are compared.
            max_size: Balanced cut: largest cluster.
            max_distance_fc: Balanced cut: largest height ratio to join.
            key_added: ``obs`` / ``uns`` key.
            recompute: Rebuild the dendrogram even if one is stored.
        """
        if "connectivities" not in adata.obsp:
            raise KeyError("Run ps.pp.neighbors() first")
        from polariseq._polariseq import (
            dendrogram_cut_balanced,
            dendrogram_cut_straight,
            paris_dendrogram,
        )

        conn = adata.obsp["connectivities"]
        if hasattr(conn, "has_canonical_format") and not conn.has_canonical_format:
            conn = conn.copy()
            conn.sum_duplicates()  # sorts rows too
        data = np.asarray(conn.data, dtype=np.float64)
        if data.size and not (np.all(data > 0) and np.all(np.isfinite(data))):
            raise ValueError("Paris needs positive, finite edge weights")
        graph = [int(adata.n_obs), int(data.size), float(np.sum(data))]
        stored = adata.uns.get(key_added) or {}
        if recompute or stored.get("graph") != graph or "dendrogram" not in stored:
            d = paris_dendrogram(_as_u32(conn.indptr), _as_u32(conn.indices), data, adata.n_obs)
            stored = {"dendrogram": d, "graph": graph}
            adata.uns[key_added] = stored
        d = stored["dendrogram"]
        if balanced:
            if min_size is None or max_size is None:
                raise ValueError("balanced=True needs min_size and max_size")
            labels = dendrogram_cut_balanced(d, int(max_size), int(min_size), float(max_distance_fc))
            adata.obs[key_added] = (labels - 1).astype(np.int32)
            stored["cut"] = {"balanced": True, "min_size": int(min_size), "max_size": int(max_size),
                             "max_distance_fc": float(max_distance_fc)}
        elif n_clusters is not None:
            labels = dendrogram_cut_straight(d, int(n_clusters))
            adata.obs[key_added] = labels.astype(np.int32)
            stored["cut"] = {"n_clusters": int(n_clusters)}
        if key_added in adata.obs:
            stored["n_clusters"] = int(np.asarray(adata.obs[key_added]).max()) + 1

    def dpt(
        self,
        adata: AnnData,
        root: int = 0,
    ) -> None:
        """Compute diffusion pseudotime from a root cell.

        Uses Dijkstra shortest path on the kNN graph. Stores per-cell
        pseudotime in ``adata.obs['dpt_pseudotime']``.

        Args:
            adata: The data container.
            root: Index of the root cell (pseudotime origin).
        """
        if "distances" not in adata.obsp:
            raise KeyError("Run ps.pp.neighbors() first")

        from polariseq._polariseq import pseudotime_dijkstra as _dpt
        dist = adata.obsp["distances"]
        result = _dpt(
            dist.indptr.astype(np.uint32),
            dist.indices.astype(np.uint32),
            dist.data.astype(np.float32),
            adata.n_obs,
            root,
        )
        adata.obs["dpt_pseudotime"] = np.asarray(result["pseudotime"], dtype=np.float32)
        adata.uns["dpt"] = {"root": root, "n_reachable": result["n_reachable"]}

    def paga(
        self,
        adata: AnnData,
        groupby: str = "leiden",
    ) -> None:
        """Compute PAGA (partition-based graph abstraction).

        Creates a cluster-level connectivity graph showing relationships
        between cell populations. Stores in ``adata.uns['paga']``.

        Args:
            adata: The data container.
            groupby: Column in obs identifying clusters.
        """
        if "distances" not in adata.obsp:
            raise KeyError("Run ps.pp.neighbors() first")
        if groupby not in adata.obs:
            raise KeyError(f"groupby '{groupby}' not in obs")

        from polariseq._polariseq import paga_connectivities as _paga
        dist = adata.obsp["distances"]
        labels = np.asarray(adata.obs[groupby]).astype(np.uint32)
        n_clusters = int(labels.max()) + 1

        result = _paga(
            dist.indptr.astype(np.uint32),
            dist.indices.astype(np.uint32),
            dist.data.astype(np.float32),
            adata.n_obs,
            labels,
            n_clusters,
        )
        conn = np.asarray(result["connectivities"]).reshape(n_clusters, n_clusters)
        adata.uns["paga"] = {
            "connectivities": conn,
            "cluster_sizes": result["cluster_sizes"],
            "groupby": groupby,
        }

    def umap(
        self,
        adata: AnnData,
        *,
        min_dist: float | None = None,
        spread: float = 1.0,
        n_epochs: int = 0,
        seed: int = 42,
        deterministic: bool = False,
        init: str | np.ndarray | None = None,
        init_smooth: int = 30,
    ) -> None:
        """Run UMAP on the neighbor graph, storing the 2D embedding in
        ``adata.obsm['X_umap']``.

        Requires :meth:`pp.neighbors` to have been run first.

        Args:
            adata: The data container (mutated in place).
            min_dist: Minimum distance between embedded points (left out:
                0.5 in the ``'polariseq'`` workflow, 0.1 in ``'scanpy'``).
            spread: Scale of embedded point spread.
            n_epochs: SGD epochs (0 = auto: 500 up to 10,000 cells, else 100
                in the ``'polariseq'`` workflow and 200 in ``'scanpy'``).
            seed: Random seed.
            deterministic: Run single-threaded, bit-reproducible SGD instead
                of the default parallel (Hogwild) mode. The parallel layout
                differs run to run by construction, as umap-learn's does.
            init: Where the cells start. ``'spectral'`` (the graph's leading
                eigenvectors, each axis in [-10, 10]), ``'random'`` (uniform in
                [-10, 10]), ``'pca'`` (the first two of ``obsm['X_pca']``),
                ``'kmeans'`` (Scarf's start: 1,000 k-means centroids of
                ``obsm['X_pca']`` laid out by their own PCA, each cell at its
                centroid), ``'multilevel'`` (Polariseq's multilevel spectral
                start: the exact spectral layout of the graph between k-means
                groups of ``obsm['X_pca']`` (about 20 cells per group, 50 to
                1,000 groups), lifted to the cells and
                refined by ``init_smooth`` smoothing steps on the cell graph),
                ``'leiden'`` (the same through the graph between the Leiden
                clusters in ``obs['leiden']``), or an n × 2 array. Left out:
                ``'multilevel'`` in the ``'polariseq'`` workflow, ``'spectral'``
                in ``'scanpy'``. Any start
                but the first two is scaled, axis by axis, to [0, 10],
                umap-learn's convention for every start.
            init_smooth: Smoothing steps of the multilevel starts.
        """
        polariseq_defaults = _WORKFLOW == "polariseq"
        if init is None:
            init = "multilevel" if polariseq_defaults else "spectral"
        if min_dist is None:
            min_dist = 0.5 if polariseq_defaults else 0.1
        if n_epochs == 0 and polariseq_defaults and adata.n_obs > 10_000:
            n_epochs = 100
        # The graph `pp.neighbors` stored, read in place: its connectivities
        # (the fuzzy set itself, as Scanpy lays out), else its distances.
        use_conn = "connectivities" in adata.obsp
        if not use_conn and "distances" not in adata.obsp:
            raise KeyError("Run ps.pp.neighbors() first")
        graph = adata.obsp["connectivities" if use_conn else "distances"]
        if use_conn and hasattr(graph, "has_sorted_indices") and not graph.has_sorted_indices:
            graph.sort_indices()  # the core expects rows sorted by column
        out = _results.new(adata, "umap", "X_umap", (adata.n_obs, 2), np.float32)
        mode = init if isinstance(init, str) else "given"
        if isinstance(init, str) and init in ("pca", "kmeans"):
            if "X_pca" not in adata.obsm:
                raise KeyError(f"init={init!r} needs obsm['X_pca']; run ps.pp.pca() first")
            if init == "pca":
                start = np.asarray(adata.obsm["X_pca"][:, :2], dtype=np.float64)
            else:
                pcs = np.ascontiguousarray(adata.obsm["X_pca"], dtype=np.float32)
                start = np.asarray(umap_kmeans_start(pcs, 1000, int(seed)), dtype=np.float64)
            mode = "given"
        elif isinstance(init, str) and init in ("multilevel", "leiden"):
            if use_conn:
                c_ip, c_ix, c_d = _as_u32(graph.indptr), _as_u32(graph.indices), np.asarray(graph.data, np.float32)
            else:
                fz = fuzzy_connectivities(_as_u32(graph.indptr), _as_u32(graph.indices),
                                          np.asarray(graph.data, dtype=np.float32), adata.n_obs)
                c_ip, c_ix, c_d = fz["indptr"], fz["indices"], fz["data"]
            if init == "leiden":
                if "leiden" not in adata.obs:
                    raise KeyError("init='leiden' needs obs['leiden']; run ps.tl.leiden() first")
                cats, codes = np.unique(np.asarray(adata.obs["leiden"]), return_inverse=True)
                labels, n_groups = codes.astype(np.uint32), len(cats)
            else:
                if "X_pca" not in adata.obsm:
                    raise KeyError("init='multilevel' needs obsm['X_pca']; run ps.pp.pca() first")
                pcs = np.ascontiguousarray(adata.obsm["X_pca"], dtype=np.float32)
                # about 20 cells per group, 50 to 1,000 groups: 500 or more groups brought the start within
                # 5% of the eigenvectors' energy on every graph measured
                n_want = int(min(1000, max(50, adata.n_obs // 20)))
                labels = np.asarray(umap_kmeans_labels(pcs, n_want, int(seed)), dtype=np.uint32)
                n_groups = int(labels.max()) + 1
            start = np.asarray(umap_multilevel_layout(c_ip, c_ix, c_d, adata.n_obs, labels, n_groups,
                                                      int(init_smooth)), dtype=np.float64)
            mode = "given"
        elif not isinstance(init, str):
            start = np.asarray(init, dtype=np.float64)
            if start.shape != (adata.n_obs, 2):
                raise ValueError(f"init must be n_obs × 2, not {start.shape}")
        elif init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral', 'random', 'pca', 'kmeans', 'multilevel', 'leiden' "
                             f"or an array, not {init!r}")
        if mode == "given":
            lo, hi = start.min(axis=0), start.max(axis=0)
            out[:] = 10.0 * (start - lo) / np.where(hi > lo, hi - lo, 1.0)
        umap_embed(
            _as_u32(graph.indptr),
            _as_u32(graph.indices),
            np.asarray(graph.data, dtype=np.float32),
            adata.n_obs,
            min_dist,
            spread,
            n_epochs,
            seed,
            deterministic,
            connectivities=use_conn,
            out=out,
            init=mode,
        )
        adata.obsm["X_umap"] = _results.view(out)
        adata.uns["umap"] = {"min_dist": min_dist, "spread": spread}

    def leiden(
        self,
        adata: AnnData,
        resolution: float = 1.0,
        *,
        seed: int = 0,
        key_added: str = "leiden",
        method: str = "auto",
    ) -> None:
        """Run community detection on the neighbor graph.

        Algorithms:
        - ``'leiden'``: full Leiden (Traag et al. 2019), igraph-class quality,
          deterministic per seed
        - ``'fast'``: multi-level Louvain on the same engine (opt-in)
        - ``'igraph'``: igraph's C Leiden on the same weighted graph
          (requires the optional python-igraph package); the reference
        - ``'auto'`` (default): Leiden at every scale

        All methods run on the fuzzy connectivity graph
        (``obsp['connectivities']``, computed by ``pp.neighbors``), so their
        reported modularities are directly comparable.

        Requires :meth:`pp.neighbors` to have been run first.

        Args:
        """
        # The graph every method runs on: the symmetrised fuzzy
        # connectivities (scanpy semantics; the graph UMAP shares), used as
        # given. Same weights for ours, fast, and igraph, so quality numbers
        # are comparable across implementations. The arrays are handed over
        # as views, not copies: at 4M cells the graph is ~1 GB, and each
        # `astype` here used to duplicate it.
        if "connectivities" in adata.obsp:
            conn = adata.obsp["connectivities"]
            if hasattr(conn, "has_sorted_indices") and not conn.has_sorted_indices:
                conn.sort_indices()  # in place; the core expects sorted rows
            c_indptr = _as_u32(conn.indptr)
            c_indices = _as_u32(conn.indices)
            c_data = np.asarray(conn.data, dtype=np.float32)
        else:
            dist = adata.obsp["distances"]
            conn = fuzzy_connectivities(_as_u32(dist.indptr), _as_u32(dist.indices),
                                        np.asarray(dist.data, dtype=np.float32), adata.n_obs)
            c_indptr = conn["indptr"]
            c_indices = conn["indices"]
            c_data = conn["data"]

        if method == "igraph":
            # Optional backend delegating to the C igraph library (requires
            # python-igraph). Measured on real atlases: igraph's Leiden is the
            # strongest mid-scale reference implementation, so it is offered
            # as an explicit choice rather than a silent default.
            try:
                import igraph as _ig
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "method='igraph' requires the python-igraph package "
                    "(pip install igraph)"
                ) from e
            n = adata.n_obs
            rows = np.repeat(
                np.arange(n, dtype=np.int64), np.diff(c_indptr.astype(np.int64))
            )
            cols = c_indices.astype(np.int64)
            mask = cols > rows  # dedupe symmetric storage
            g = _ig.Graph(
                n=n,
                edges=list(zip(rows[mask].tolist(), cols[mask].tolist(), strict=True)),
                directed=False,
            )
            g.es["weight"] = c_data[mask].astype(float).tolist()
            # igraph draws from Python's `random` module unless given its own
            # generator; without one the result depends on whatever ran before.
            import random as _random
            _ig.set_random_number_generator(_random.Random(seed))
            try:
                part = g.community_leiden(
                    weights="weight", resolution=resolution, n_iterations=2,
                    objective_function="modularity",
                )
            finally:
                _ig.set_random_number_generator(_random)
            labels = np.asarray(part.membership, dtype=np.int32)
            adata.obs[key_added] = labels
            adata.uns["leiden"] = {
                "n_communities": int(len(part)),
                "quality": float(g.modularity(part.membership, weights="weight")),
                "resolution": resolution,
                "method": "igraph_leiden",
                "params": {"seed": seed, "key_added": key_added},
            }
            return

        # The Rust Leiden is igraph-class at every scale we can measure, so
        # 'auto' is always Leiden; 'fast' (multi-level Louvain, same engine)
        # stays available as an explicit opt-in.
        use_fast = method == "fast"

        # c_indptr/c_indices/c_data: the fuzzy connectivity graph. An earlier
        # version passed the *distances* arrays here while the igraph branch
        # below used the connectivities -- so the default methods clustered a
        # different graph than documented, than igraph, and than UMAP shares.
        # It also happened to be pathologically slow on some inputs: on the
        # spill arm's heart_50k kNN graph, Leiden's local moves oscillated on
        # the unsymmetrised distance weights for minutes where the
        # connectivity graph clusters in ~2 s (found while running the
        # telemetry ladder; chapter 09).
        if use_fast:
            result = fast_community(c_indptr, c_indices, c_data, adata.n_obs, resolution, seed)
            algo = "fast_label_propagation"
        else:
            result = leiden_cluster(c_indptr, c_indices, c_data, adata.n_obs, resolution, seed)
            algo = "leiden"

        adata.obs[key_added] = result["labels"].astype(np.int32)
        adata.uns["leiden"] = {
            "n_communities": int(result["n_communities"]),
            "quality": float(result["quality"]),
            "resolution": resolution,
            "method": algo,
            "n_edges": int(result["n_edges"]),
            "total_weight": float(result["total_weight"]),
            "params": {"seed": seed, "key_added": key_added},
        }


class _Get:
    """The ``ps.get`` namespace: data taken out of an analysis."""

    def markers(self, adata: AnnData, n: int = 10, *, groups: list | None = None,
                wide: bool = False) -> Any:
        """Top marker genes per cluster from ``tl.rank_genes_groups``, as a table (``wide=True``: one
        column of gene names per cluster). See :func:`polariseq._curation.markers`."""
        from polariseq._curation import markers
        return markers(adata, n, groups=groups, wide=wide)

    def pseudobulk(self, adata: AnnData | LazyAnnData, sample_key: str, *,
                   groupby: str | None = None, condition_key: str | None = None,
                   min_cells: int = 0) -> AnnData:
        """Raw counts summed per replicate (and per cell type and condition).

        The table edgeR, DESeq2 or limma test: one row per combination of
        ``groupby``, ``condition_key`` and ``sample_key`` with at least
        ``min_cells`` cells, the gene counts summed, and the number of cells
        in ``obs['n_cells']``. The counts are summed in the Rust core from
        wherever the raw counts are held (see :meth:`tl.compare_conditions`);
        with a project folder the result is a memory-mapped file. Pass the
        result's ``to_anndata()`` to pydeseq2, or write its ``X`` for R.
        """
        from polariseq._compare import pseudobulk as _pseudobulk

        adata = _Preprocessing._ensure_materialized(adata)
        return _pseudobulk(adata, sample_key, groupby=groupby, condition_key=condition_key,
                           min_cells=min_cells)


pp = _Preprocessing()
tl = _Tools()
get = _Get()

# Plotting namespace
from polariseq import plotting  # noqa: E402,F401  (late: plotting imports back)
from polariseq import plotting as _plotting_module  # noqa: E402
from polariseq.plotting import (  # noqa: E402
    dotplot,
    heatmap,
    highly_variable_genes,
    pca_variance_ratio,
    scatter_qc,
    stacked_bar,
    violin,
)
from polariseq.plotting import (  # noqa: E402  (late: plotting imports back)
    pca as pca_plot,
)
from polariseq.plotting import (  # noqa: E402  (late: plotting imports back)
    umap as umap_plot,
)


class _Plotting:
    """The ``ps.pl`` namespace: scanpy-compatible static plots."""

    umap = staticmethod(umap_plot)
    pca = staticmethod(pca_plot)
    pca_variance_ratio = staticmethod(pca_variance_ratio)
    violin = staticmethod(violin)
    highly_variable_genes = staticmethod(highly_variable_genes)
    dotplot = staticmethod(dotplot)
    heatmap = staticmethod(heatmap)
    stacked_bar = staticmethod(stacked_bar)
    scatter_qc = staticmethod(scatter_qc)
    qc_distributions = staticmethod(_plotting_module.qc_distributions)
    cluster_sizes = staticmethod(_plotting_module.cluster_sizes)
    overview = staticmethod(_plotting_module.overview)
    embedding = staticmethod(_plotting_module.embedding)
    rank_genes_groups = staticmethod(_plotting_module.rank_genes_groups)
    scrublet_score_distribution = staticmethod(_plotting_module.scrublet_score_distribution)
    volcano = staticmethod(_plotting_module.volcano)
    fold_changes = staticmethod(_plotting_module.fold_changes)
    # theme control
    set_theme = staticmethod(_plotting_module.set_theme)
    get_theme = staticmethod(_plotting_module.get_theme)
    theme = staticmethod(_plotting_module.theme)
    THEMES = _plotting_module.THEMES
    categorical_palette = staticmethod(_plotting_module.categorical_palette)

    def __getattr__(self, name: str):
        """Fall through to the plotting module.

        The list above is explicit so that tab-completion shows the supported
        surface, but a hand-maintained list silently omits every new plotting
        function until someone remembers to add it here. This makes the
        omission a non-event rather than an AttributeError.
        """
        try:
            return getattr(_plotting_module, name)
        except AttributeError:
            raise AttributeError(
                f"ps.pl has no attribute {name!r}. Available: "
                f"{sorted(k for k in vars(type(self)) if not k.startswith('_'))}"
            ) from None


pl = _Plotting()


def integrate_datasets(
    paths: list[str],
    *,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    chunk_size: int = 50_000,
    reference_size: int = 100_000,
    harmony_clusters: int = 30,
    seed: int = 42,
) -> dict:
    """Stream multiple datasets, integrate with batch correction — low memory.

    The full pipeline without loading full expression matrices:
    1. Scan metadata (instant) → find shared genes across datasets
    2. Streaming HVG across all datasets (chunked reads)
    3. Reference-based PCA (subsample from all, then project each chunk)
    4. Harmony batch correction on the embedding

    Peak memory: O(chunk) for reads + O(n_total × n_pcs) for the embedding.
    For 3M cells × 50 PCs: ~1.4 GB total (vs 24+ GB materialized).

    Args:
        paths: List of .h5ad file paths.
        n_hvg: Number of highly variable genes (across all datasets).
        n_pcs: Number of PCA components.
        chunk_size: Rows per streaming chunk.
        reference_size: Subsample size for reference PCA.
        harmony_clusters: Number of Harmony clusters.
        seed: Random seed.

    Returns:
        Dict with 'embedding' (n_total × n_pcs), 'batch_labels', 'stats'.
    """
    import warnings

    warnings.warn(
        "integrate_datasets is deprecated: read the files as one dataset, "
        "ps.read_h5ad([...]) (in memory or out of core, as the budget allows), then run "
        "ps.pp.pca and ps.pp.harmony_integrate(adata, 'dataset').",
        DeprecationWarning, stacklevel=2)
    from polariseq._polariseq import integrate_datasets as _integrate_raw
    return _integrate_raw(
        paths, n_hvg, n_pcs, chunk_size, reference_size, harmony_clusters, seed
    )


class BudgetWarning(UserWarning):
    """An in-memory load is predicted to exceed the active budget.

    The budget bounds the out-of-core mode only; a matrix held in memory is
    as large as the data. This warning is how the in-memory mode says so —
    it never stops the run. Silence it with
    ``warnings.filterwarnings("ignore", category=ps.BudgetWarning)``.
    """


def _warn_if_over_budget(n_cells: int, n_genes: int, nnz: int, path: str) -> None:
    """Warn when holding this matrix in memory is predicted to exceed the budget.

    Costed with the same model and defaults as :func:`plan`, from shape alone,
    so it reads nothing and the numbers match what ``ps.plan(path)`` prints.
    """
    b = budget()
    ram = _estimate(n_cells, n_genes, nnz, b, storage="ram")
    if ram.fits:
        return
    import os
    import warnings

    name = os.path.basename(path)
    rec = _estimate(n_cells, n_genes, nnz, b)
    if not rec.fits:
        advice = "No mode fits this budget; see ps.plan() for the floor."
    elif rec.storage == "spill":
        advice = (f"ps.plan() recommends out-of-core for this file: "
                  f'ps.process_diskbacked("{path}").')
    else:
        advice = "ps.plan() recommends the packed in-memory form for this file."
    warnings.warn(
        f"in-memory analysis of {name} is predicted to peak at "
        f"{ram.peak_gb:.1f} GB, above the {b.ram_gb:.1f} GB budget "
        f"(limiting step: {ram.limiting_step}). The budget does not bound the "
        f"in-memory mode. {advice}",
        BudgetWarning,
        stacklevel=3,
    )


def estimate_dataset_size_gb(path: str) -> float:
    """Estimate the in-memory size of a .h5ad dataset's CSR matrix in GB."""
    import os
    return os.path.getsize(path) / 1e9


def needs_disk_backing(path: str) -> bool:
    """Check if a dataset exceeds the RAM limit and needs disk-backed processing."""
    return estimate_dataset_size_gb(path) * 1e9 > budget().ram_bytes



def _windows_pid_alive(pid: int) -> bool:
    """Whether a process with this id exists (Windows; os.kill(pid, 0) means Ctrl+C there, not a check).

    A process that cannot be opened for another reason (access denied) counts as alive.
    """
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
    try:
        code = wintypes.DWORD()
        if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == 259  # STILL_ACTIVE
        return True
    finally:
        k32.CloseHandle(handle)


def _sweep_orphan_spills(spill_dir: str) -> int:
    """Delete spill files left by processes that no longer exist.

    Spills are named ``raw_<pid>.*`` / ``hvg_<pid>.*``. A run that is killed
    or crashes cannot delete its own (and before 2026-09-29 no run deleted
    its raw spill at all), so each out-of-core run starts by removing those
    of dead processes. A pid that is alive, or that belongs to another user,
    is left alone. Returns the bytes freed.
    """
    import os
    import re

    pat = re.compile(r"^(?:raw|hvg)_(\d+)\.(?:data|indices|indptr)$")
    freed = 0
    try:
        names = os.listdir(spill_dir)
    except OSError:
        return 0
    for name in names:
        m = pat.match(name)
        if not m:
            continue
        pid = int(m.group(1))
        if pid == os.getpid():
            continue
        if os.name == "nt":
            try:
                if _windows_pid_alive(pid):
                    continue
            except (OSError, AttributeError):
                continue  # cannot tell: leave the files alone
        else:
            try:
                os.kill(pid, 0)
                continue  # alive
            except PermissionError:
                continue  # alive, another user's
            except ProcessLookupError:
                pass
            except OSError:
                continue
        path = os.path.join(spill_dir, name)
        try:
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
        except OSError:
            pass
    return freed


def _read_h5ad_names(path: str, keys: tuple[str, ...] = ("obs", "var")) -> dict[str, np.ndarray | None]:
    """Cell and gene names from a `.h5ad`, without touching the matrix.

    The index lives under the name in the group's `_index` attribute, but
    its *encoding* varies by AnnData version: older files store a plain
    string dataset, newer ones a `nullable-string-array` group holding
    `values` and a `mask`. Both are handled. Anything unreadable yields
    None and the caller falls back to positional identifiers rather than
    failing a whole pipeline over labels. ``keys`` limits the read to cells
    or genes: millions of cell names are a few hundred megabytes, which an
    out-of-core run must not hold while it streams the matrix.
    """
    out: dict[str, np.ndarray | None] = {"obs": None, "var": None}
    try:
        import h5py
    except ImportError:
        return out
    try:
        with h5py.File(path, "r") as f:
            for key in keys:
                grp = f.get(key)
                if grp is None:
                    continue
                name = grp.attrs.get("_index", "_index")
                if isinstance(name, bytes):
                    name = name.decode()
                node = grp.get(str(name))
                if isinstance(node, h5py.Group):
                    # nullable-string-array: values + a validity mask.
                    node = node.get("values")
                if not isinstance(node, h5py.Dataset):
                    continue
                out[key] = np.asarray(
                    [v.decode() if isinstance(v, bytes) else str(v) for v in node[:]]
                )
    except (OSError, KeyError, ValueError):
        pass
    return out


def process_diskbacked(
    path: str,
    *,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    target_sum: float = 1e4,
    resolution: float = 1.0,
    seed: int = 42,
    min_genes: int = 0,
    max_genes: float = np.inf,
    min_counts: float = 0,
    max_counts: float = np.inf,
    max_pct_mt: float = np.inf,
    mito_prefix: str = "MT-",
    min_cells: int = 0,
) -> AnnData:
    """Run the pipeline on a dataset that exceeds RAM, using SSD.

    The matrix is streamed from the file straight onto SSD (never held in
    memory), memory-mapped, and then processed in place: QC metrics,
    ``filter_cells(min_genes)``, ``normalize_total``, ``log1p``,
    highly-variable genes, PCA. Only per-gene vectors and the
    ``n_obs × n_pcs`` embedding live in RAM. Neighbours and clustering then
    run on that embedding.

    These are the same kernels the in-memory pipeline uses, on a different
    block source, so the results match the in-memory pipeline given the same
    ``min_genes``. The workflow (:func:`set_workflow`) decides the gene
    selection, normalization and scaling: ``'polariseq'`` ranks genes among
    those detected in more than 1% of the cells, normalizes each cell again
    over the selected genes and standardizes each gene before PCA;
    ``'scanpy'`` follows Scanpy's steps. Before ``min_genes`` existed this path silently kept every
    cell, answering a different question than ``ps.pp`` on the same file.

    Args:
        path: Path to the ``.h5ad`` file.
        n_hvg: Number of highly-variable genes.
        n_pcs: Number of principal components.
        n_neighbors: Neighbours per cell for the kNN graph.
        target_sum: Target counts per cell for normalisation.
        resolution: Leiden resolution.
        seed: Random seed.
        min_genes: Minimum detected genes for a cell to be kept (0 keeps
            all cells), matching ``ps.pp.filter_cells``.
        max_genes: Maximum detected genes for a cell to be kept.
        min_counts: Minimum total counts for a cell to be kept.
        max_counts: Maximum total counts for a cell to be kept.
        max_pct_mt: Maximum percentage of a cell's counts from mitochondrial
            genes (``obs['pct_counts_mt']``, reported whenever such genes
            exist).
        mito_prefix: Prefix of the mitochondrial gene names, as in
            ``ps.pp.calculate_qc_metrics``.
        min_cells: Minimum number of kept cells in which a gene is detected
            for it to be kept, matching ``ps.pp.filter_genes`` after
            ``filter_cells``; removed genes take no part in normalization,
            gene selection or PCA (``var['kept']``).

    The cell and gene filters are those of the in-memory functions, applied
    in the same order (quality control, ``filter_cells``, ``filter_genes``),
    so both modes analyze the same cells and genes.

    Returns:
        An :class:`AnnData` carrying ``obsm['X_pca']``, the neighbour graph,
        cluster labels, and the QC/HVG results in ``obs``/``var``.
    """
    import os
    import time

    file_gb = estimate_dataset_size_gb(path)
    spill_dir = budget().spill_dir
    os.makedirs(spill_dir, exist_ok=True)
    freed = _sweep_orphan_spills(spill_dir)
    if freed:
        print(f"[disk-backed] removed {freed / 1e9:.1f} GB of spill files left by "
              "processes that no longer exist")
    print(f"[disk-backed] {path}: {file_gb:.1f} GB file, spilling to {spill_dir}/")

    # Size the streamed blocks from the budget rather than a constant: the
    # working set of this path is a small multiple of one block, so this is
    # what keeps a big-but-sparse file inside a small machine.
    sized = plan(
        path, n_hvg=n_hvg, n_pcs=n_pcs, storage="spill", verbose=False
    )
    # The working set has a floor — file residency plus the process plus one
    # block in flight — that block sizing cannot shrink below. A budget under
    # it is refused rather than honoured with ever-smaller blocks: measured
    # on the 800K-cell sweep, sub-floor requests demanded *more* memory than
    # requests at the floor, because the churn of the extra blocks outweighed
    # their size. Refusal is the budget meaning what it says.
    if budget().ram_bytes < sized.min_budget_bytes:
        raise ValueError(
            f"the {budget().ram_gb:.1f} GB budget is below the out-of-core"
            f" working-set floor of {sized.min_budget_gb:.1f} GB for this file"
            " (spill-file residency + process + one block in flight); no block"
            " size meets it. Raise the budget (ps.set_budget) or analyse a"
            " subset. This refusal is deliberate: sub-floor runs were measured"
            " to demand more than runs at the floor."
        )
    print(
        f"[disk-backed] plan: {sized.rows_per_block:,}-row blocks, "
        f"predicted peak {sized.peak_gb:.2f} GB of {budget().ram_gb:.1f} GB budget"
        f" (floor {sized.min_budget_gb:.2f} GB)"
    )
    if sized.peak_bytes > budget().ram_bytes:
        # The streamed steps honour the budget by construction; the graph
        # steps hold per-cell results (neighbours, connectivities) for every
        # cell and cannot be streamed, so say so before running rather than
        # after.
        import warnings

        warnings.warn(
            f"the '{sized.limiting_step}' step is predicted to need "
            f"{sized.peak_gb:.1f} GB, above the {budget().ram_gb:.1f} GB budget: the "
            "per-cell graph steps scale with the number of cells and are not "
            "streamed. Raise the budget or analyse fewer cells.",
            BudgetWarning,
            stacklevel=2,
        )

    # Mitochondrial genes are marked exactly as calculate_qc_metrics marks
    # them. Only the gene names are read here, a small string read; the cell
    # names wait until the matrix pass is over (measured: holding a million
    # cell names through it raised the peak by 0.3 GB).
    var_names = _read_h5ad_names(path, keys=("var",))["var"]
    mito = None
    if var_names is not None and mito_prefix:
        mito = np.array([str(n).startswith(mito_prefix) for n in var_names], dtype=bool)
        if not mito.any():
            mito = None
    if max_pct_mt < np.inf and mito is None:
        raise ValueError(
            f"max_pct_mt needs mitochondrial genes: no gene name starts with {mito_prefix!r}"
        )

    t0 = time.perf_counter()
    r = _disk_backed_preprocess(
        path, spill_dir, n_hvg, n_pcs, target_sum, seed, sized.rows_per_block, min_genes,
        max_genes=None if max_genes == np.inf else int(max_genes),
        min_counts=float(min_counts),
        max_counts=None if max_counts == np.inf else float(max_counts),
        max_pct_mt=None if max_pct_mt == np.inf else float(max_pct_mt),
        mito=mito,
        min_cells=int(min_cells),
        # Polariseq's workflow (set_workflow): genes ranked among those detected in more than 1% of
        # the cells, each cell normalized again over them, genes standardized before PCA
        hvg_floor_frac=0.01 if _WORKFLOW == "polariseq" else 0.0,
        renormalize=1000.0 if _WORKFLOW == "polariseq" else None,
        scale_pcs=_WORKFLOW == "polariseq",
    )
    prep_t = time.perf_counter() - t0
    filtered = int(r.get("n_obs_input", r["n_obs"])) - int(r["n_obs"])
    gene_keep = np.asarray(r["gene_keep"], dtype=bool)
    print(
        f"[disk-backed] spill + QC + normalize + log1p + HVG + PCA: {prep_t:.1f}s "
        f"({r['n_obs']:,} of {r.get('n_obs_input', r['n_obs']):,} cells"
        + (f", {filtered:,} removed by quality control" if filtered else "")
        + f" × {int(gene_keep.sum()):,} of {r['n_vars']:,} genes, {r['nnz']:,} nnz)"
    )

    # The embedding and the annotations are small; the matrix stays on disk.
    adata = AnnData(X=np.zeros((r["n_obs"], 1), dtype=np.float32))
    # Carry the file's cell barcodes and gene names through. The disk path
    # filters cells, so row i of the embedding is not input row i, and
    # anything comparing this output to another implementation's would
    # otherwise have to assume the two orders match — the exact assumption
    # that hides a filtering bug. `cell_keep` is the mask that makes the
    # alignment explicit; names are read from the file's metadata, so this
    # costs a small string read and no matrix I/O.
    keep = np.asarray(r.get("cell_keep", []), dtype=bool)
    names = {"obs": _read_h5ad_names(path, keys=("obs",))["obs"], "var": var_names}
    if names["obs"] is not None and keep.size == names["obs"].size:
        adata.obs_names = list(names["obs"][keep])
    hv = np.asarray(r["highly_variable"], dtype=bool)
    if names["var"] is not None and hv.size == names["var"].size:
        adata.uns["hvg_names"] = list(names["var"][hv])
        adata.var["name"] = np.asarray(names["var"], dtype=object)
    adata.obsm["X_pca"] = np.asarray(r["embedding"], dtype=np.float32)
    adata.obs["n_counts"] = np.asarray(r["n_counts"], dtype=np.float32)
    adata.obs["n_genes"] = np.asarray(r["n_genes"], dtype=np.int32)
    if "pct_counts_mt" in r:
        adata.obs["pct_counts_mt"] = np.asarray(r["pct_counts_mt"], dtype=np.float32)
    # Which row of the file each cell is, for anything that streams the
    # matrix again (marker genes) or lines results up with the file.
    adata.obs["file_row"] = np.flatnonzero(keep).astype(np.int64)
    adata.var["kept"] = gene_keep
    adata.var["n_cells"] = np.asarray(r["gene_n_cells"], dtype=np.int32)
    adata.var["total_counts"] = np.asarray(r["gene_total_counts"], dtype=np.float64)
    adata.var["highly_variable"] = np.asarray(r["highly_variable"], dtype=bool)
    adata.var["means"] = np.asarray(r["hvg_means"])
    adata.var["dispersions_norm"] = np.asarray(r["hvg_dispersions_norm"])
    adata.uns["pca"] = {
        "variance_ratio": r["variance_ratio"],
        "n_components": int(r["n_components"]),
        "disk_backed": True,
    }
    adata.uns["hvg"] = {"n_top_genes": int(adata.var["highly_variable"].sum()),
                        "flavor": "seurat"}
    # Where the matrix is, and how it was normalized, so steps that need the
    # matrix again (ps.tl.rank_genes_groups) stream it with the same
    # transform and the same gene filter.
    adata.uns["disk_backed"] = {
        "path": os.path.abspath(path),
        "target_sum": float(target_sum),
        "n_obs_file": int(r["n_obs_input"]),
        "n_vars_file": int(r["n_vars"]),
        "qc": {"min_genes": int(min_genes), "max_genes": float(max_genes),
               "min_counts": float(min_counts), "max_counts": float(max_counts),
               "max_pct_mt": float(max_pct_mt), "mito_prefix": mito_prefix,
               "min_cells": int(min_cells)},
    }

    print("[disk-backed] neighbors + leiden on the embedding...")
    pp.neighbors(adata, n_neighbors=n_neighbors, seed=seed)
    tl.leiden(adata, resolution=resolution, seed=seed)
    print(f"[disk-backed] done: {adata.uns['leiden']['n_communities']} clusters")
    return adata


def _positions(key: Any, n: int, names: Any, what: str) -> np.ndarray | None:
    """The positions ``key`` selects along an axis of ``n`` ``what``
    (``None``: all of them): a mask, positions, names or a slice."""
    import pandas as pd

    if isinstance(key, slice):
        return None if key == slice(None) else np.arange(n)[key]
    if isinstance(key, (str, bytes)) or np.isscalar(key):
        key = [key]
    if isinstance(key, (pd.Series, pd.Index)):
        key = key.to_numpy()
    arr = np.asarray(key)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise IndexError(f"a mask over {what} needs {n:,} entries, not {arr.size:,}")
        return np.flatnonzero(arr)
    if arr.dtype.kind in "iu":
        pos = arr.astype(np.int64).ravel()
        pos = np.where(pos < 0, pos + n, pos)
        if pos.size and (pos.min() < 0 or pos.max() >= n):
            raise IndexError(f"a position is out of range for {n:,} {what}")
        return pos
    lookup = {str(v): i for i, v in enumerate(names)}
    try:
        return np.fromiter((lookup[str(k)] for k in arr.ravel()), dtype=np.int64,
                           count=arr.size)
    except KeyError as e:
        raise KeyError(f"no {what} named {e.args[0]!r}") from None


def _keep_positions(held: np.ndarray | None, pos: np.ndarray | None) -> np.ndarray | None:
    """Which source rows (or genes) remain after keeping the positions
    ``pos`` of those ``held`` (``None``: all, in order)."""
    if pos is None:
        return held
    pos = np.asarray(pos, dtype=np.int64)
    return pos if held is None else np.asarray(held)[pos]


def _subset_by_key(adata: Any, key: Any) -> AnnData:
    if isinstance(key, tuple):
        if len(key) != 2:
            raise IndexError("index with adata[cells] or adata[cells, genes]")
        rows, cols = key
    else:
        rows, cols = key, slice(None)
    return _subset(adata, _positions(rows, adata.n_obs, adata.obs_names, "cells"),
                   _positions(cols, adata.n_vars, adata.var_names, "genes"))


def _subset(adata: Any, rows: np.ndarray | None, cols: np.ndarray | None) -> AnnData:
    """A new, independent object with the cells at ``rows`` and the genes at
    ``cols`` (``None``: all)."""
    import copy as _copy

    X = adata.X
    n, g = X.shape
    if _is_disk(X):
        def mask(pos: np.ndarray | None, size: int) -> np.ndarray | None:
            if pos is None:
                return None
            if pos.size > 1 and np.any(np.diff(pos) <= 0):
                raise IndexError("out of core, cells and genes keep their stored order: select "
                                 "them with a mask or with increasing positions")
            m = np.zeros(size, dtype=bool)
            m[pos] = True
            return m
        X = X.subset(mask(rows, n), mask(cols, g))
    elif _is_sparse(X):
        X = X.copy() if rows is None and cols is None else X[
            slice(None) if rows is None else rows, slice(None) if cols is None else cols]
    else:
        X = np.asarray(X)
        X = X[rows] if rows is not None else X.copy()
        if cols is not None:
            X = X[:, cols]

    def pick(values: Any, pos: np.ndarray | None) -> Any:
        if pos is not None:
            return _take(values, pos)
        return values.copy() if hasattr(values, "copy") else _take(values, slice(None))

    names, genes = list(adata.obs_names), list(adata.var_names)
    new = AnnData(X=X,
                  obs={k: pick(v, rows) for k, v in adata.obs.items()},
                  var={k: pick(v, cols) for k, v in adata.var.items()},
                  obs_names=names if rows is None else [names[i] for i in rows],
                  var_names=genes if cols is None else [genes[i] for i in cols])
    new.obsm = {k: (np.array(v) if rows is None else np.asarray(v)[rows])
                for k, v in adata.obsm.items()}
    new.obsp = {k: (v.copy() if rows is None else v[rows][:, rows])
                for k, v in adata.obsp.items()}
    new.uns = _copy.copy(adata.uns)
    new._source = getattr(adata, "_source", None)
    new._rows = _keep_positions(getattr(adata, "_rows", None), rows)
    new._cols = _keep_positions(getattr(adata, "_cols", None), cols)
    for attr in ("name", "batch_key", "labels"):
        if hasattr(adata, attr):
            setattr(new, attr, getattr(adata, attr))
    return new


def _is_sparse(x: object) -> bool:
    """Whether ``x`` is the Rust-owned :class:`SparseMatrix`."""
    return isinstance(x, SparseMatrix)


def _is_disk(x: object) -> bool:
    """Whether ``x`` is the matrix of an out-of-core analysis (:class:`DiskMatrix`)."""
    return isinstance(x, DiskMatrix)


def _x_if_loaded(adata: Any) -> Any:
    """The matrix of ``adata`` if it has one in hand, without loading it."""
    if isinstance(adata, LazyAnnData):
        return adata._X if adata._materialized else None
    return adata.X


MODES = ("auto", "memory", "out-of-core")
#: Stored values the in-memory matrix can hold: its row pointers are 32-bit.
#: The out-of-core store's are 64-bit, so larger matrices are analyzed there.
MAX_IN_MEMORY_VALUES = 2**32 - 1


def _too_large_for_memory(nnz: int, path: str) -> str:
    return (f"{os.path.basename(path)} has {nnz:,} stored values, more than the "
            f"{MAX_IN_MEMORY_VALUES:,} the in-memory mode holds; analyze it out of core "
            "(mode='out-of-core', or ps.process_diskbacked)")


def _choose_mode(mode: str, n_obs: int, n_vars: int, nnz: int, fmt: str, path: str) -> str:
    """The mode an analysis of this matrix runs in: the planning step's
    choice for the active budget, unless the caller fixed one."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if nnz > MAX_IN_MEMORY_VALUES:
        # Only the out-of-core store can address this many values.
        if mode == "memory":
            raise ValueError(_too_large_for_memory(nnz, path))
        if fmt != "h5ad":
            raise NotImplementedError("the out-of-core mode, which this matrix needs, reads .h5ad "
                                      "files; convert it to .h5ad")
        return "out-of-core"
    if mode == "auto":
        b = budget()
        mode = "memory"
        if _estimate(n_obs, n_vars, nnz, b).storage == "spill":
            # Out of core only when that mode can meet the budget. Below its
            # floor nothing fits, and the in-memory mode runs with a warning,
            # the budget being advisory there.
            ooc = _estimate(n_obs, n_vars, nnz, b, storage="spill")
            if b.ram_bytes >= ooc.min_budget_bytes:
                mode = "out-of-core"
        if mode == "out-of-core" and fmt != "h5ad":
            import warnings

            warnings.warn(
                f"{os.path.basename(path)} would be analyzed out of core, which reads .h5ad "
                "files; it is analyzed in memory instead (convert it to .h5ad to stay "
                "within the budget)",
                BudgetWarning,
                stacklevel=4,
            )
            mode = "memory"
    elif mode == "out-of-core" and fmt != "h5ad":
        raise NotImplementedError("the out-of-core mode reads .h5ad files")
    return mode


def _default_name(path: str) -> str:
    """The folder name of a dataset in the project: its file name without
    the extension (the folder name for a 10x directory)."""
    base = os.path.basename(os.path.normpath(path))
    for ext in (".h5ad", ".h5", ".loom"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


#: A gene outside the merged set in a file's gene map (`merge::DROPPED`).
_DROPPED = 2**32 - 1


def _sources(paths: Any) -> list[tuple[str, str]]:
    """(label, path) of each dataset read as one: a dict's keys, or else each
    file's name without its extension."""
    if isinstance(paths, dict):
        pairs = [(str(k), os.fspath(v)) for k, v in paths.items()]
    else:
        pairs = [(_default_name(os.fspath(p)), os.fspath(p)) for p in paths]
    if not pairs:
        raise ValueError("no datasets given")
    labels = [k for k, _ in pairs]
    for k in labels:
        if labels.count(k) > 1:
            raise ValueError(f"two datasets are labelled {k!r}; pass a dict "
                             "{label: path} to name them")
    return pairs


def _merged_genes(var_lists: list[list[str]], genes: str,
                  paths: list[str]) -> tuple[list[str], list[np.ndarray]]:
    """The genes of files read as one, and each file's map into them (its
    gene ``j`` is merged column ``map[j]``, or ``_DROPPED``), in the order of
    ``anndata.concat``: an intersection keeps the first file's order; a union
    is sorted by name unless every file lists the same genes."""
    import functools

    import pandas as pd

    if genes not in ("intersection", "union"):
        raise ValueError(f"genes must be 'intersection' or 'union', not {genes!r}")
    indexes = []
    for names, path in zip(var_lists, paths, strict=True):
        ix = pd.Index(names)
        if len(ix) == 0:
            raise ValueError(f"{path}: no gene names to match the files by")
        if not ix.is_unique:
            raise ValueError(f"{path}: gene names repeat ({ix[ix.duplicated()][0]!r}), so its "
                             "genes cannot be matched to the other files' by name")
        indexes.append(ix)
    join = pd.Index.intersection if genes == "intersection" else pd.Index.union
    merged = functools.reduce(join, indexes)
    if len(merged) == 0:
        raise ValueError("the datasets share no genes")
    maps = [np.where(m < 0, _DROPPED, m).astype(np.uint32)
            for m in (merged.get_indexer(ix) for ix in indexes)]
    return [str(g) for g in merged], maps


def _plan_datasets(paths: Any, genes: str) -> tuple[list[dict[str, Any]], int]:
    """Each dataset's shape and genes kept when read as one, and the merged
    gene count: from the files' metadata and gene names, not their cells."""
    sources = _sources(paths)
    scans = [scan_h5ad_genes(p) for _, p in sources]
    names, maps = _merged_genes([s["var_names"] for s in scans], genes, [p for _, p in sources])
    return [{"label": k, "path": p, "n_cells": int(s["n_obs"]), "n_genes": int(s["n_vars"]),
             "nnz": int(s["nnz"]), "genes_kept": int((m != _DROPPED).sum())}
            for (k, p), s, m in zip(sources, scans, maps, strict=True)], len(names)


def _merged_name(labels: list[str]) -> str:
    """The project folder of datasets read as one: their labels joined, or,
    past 64 characters, the first label, the count and a digest."""
    import hashlib
    import re

    name = "+".join(labels)
    if len(name) > 64:
        digest = hashlib.sha1("\0".join(labels).encode()).hexdigest()[:8]
        name = f"{labels[0]}+{len(labels) - 1}_more_{digest}"
    return re.sub(r"[^\w.+-]", "_", name)


def _remap_rows(rows: dict, gmap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A file's rows, as :func:`read_h5ad_rows` returns them, in merged
    columns: values, indices and row pointers, dropped genes removed and each
    row sorted (as the Rust reader writes them)."""
    ip = np.asarray(rows["indptr"], dtype=np.int64)
    m = gmap[np.asarray(rows["indices"], dtype=np.int64)]
    keep = m != _DROPPED
    ip = np.concatenate([[0], np.cumsum(keep)])[ip]
    m, data = m[keep], np.asarray(rows["data"])[keep]
    order = np.lexsort((m, np.repeat(np.arange(len(ip) - 1), np.diff(ip))))
    return data[order], m[order], ip


def _file_record(path: str) -> dict[str, Any]:
    """What identifies a file's contents: its path, size and modification time."""
    st = os.stat(path)
    return {"path": os.path.abspath(path), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def _open_out_of_core(lazy: LazyAnnData) -> DiskMatrix:
    """The on-disk copy of a dataset's counts, in the project folder.

    Written once, streaming, in the blocks the planning step chose for the
    budget; reused, without reading the files again, when they have not
    changed since (size, time and shape match what was recorded). Several
    files read as one are copied into one store, their genes remapped.
    """
    import json
    import shutil
    import time

    proj = _require_project(f"Analyzing {lazy.name} out of core")
    if isinstance(lazy, MergedLazyAnnData):
        sized = plan(dict(zip(lazy.labels, lazy.sources, strict=True)), genes=lazy.genes, storage="spill",
                     verbose=False)
    else:
        sized = plan(lazy.source_path, storage="spill", verbose=False)
    if budget().ram_bytes < sized.min_budget_bytes:
        raise ValueError(
            f"the {budget().ram_gb:.1f} GB budget is below the out-of-core working-set floor "
            f"of {sized.min_budget_gb:.1f} GB for {lazy.name}; no block size meets it. Raise "
            "the budget (ps.set_budget) or analyse a subset."
        )
    matrix = proj.run_dir(lazy.name) / "matrix"
    prefix = matrix / "counts"
    source = lazy._source_record()
    record = matrix / "source.json"
    t0 = time.perf_counter()
    if (record.exists() and json.loads(record.read_text()) == source
            and prefix.with_suffix(".indptr").exists()):
        dm = DiskMatrix.open(str(prefix), lazy._n_obs, lazy._n_vars, sized.rows_per_block)
        how = "reused"
    else:
        if matrix.exists():
            shutil.rmtree(matrix)
        matrix.mkdir(parents=True)
        dm = lazy._copy_counts(str(prefix), sized.rows_per_block)
        record.write_text(json.dumps(source, indent=1))
        how = "copied"
    # Blocks follow the store's own count of entries. Files merged over their
    # shared genes were planned from their entries summed, an upper bound;
    # with the count itself, the blocks, and so every result, are those of
    # one merged file. For a single file the count is the planned one.
    lazy._nnz = int(dm.nnz)
    sized = _estimate(lazy._n_obs, lazy._n_vars, lazy._nnz, budget(), storage="spill",
                      densest_nnz_per_row=_densest_nnz_per_row(sized.datasets))
    if sized.rows_per_block != dm.block_rows:
        dm = DiskMatrix.open(str(prefix), lazy._n_obs, lazy._n_vars, sized.rows_per_block)
    gb = sum(f.stat().st_size for f in matrix.glob("counts.*")) / 1e9
    print(f"[polariseq] {lazy.name}: out of core (planned peak {sized.peak_gb:.1f} GB of "
          f"{budget().ram_gb:.1f} GB); counts {how} in {matrix} ({gb:.1f} GB, "
          f"{time.perf_counter() - t0:.1f} s), {sized.rows_per_block:,} cells per block")
    return dm


def _in_memory_only(adata: Any, what: str) -> None:
    """Refuse, with the reason, a step that needs the whole matrix in memory."""
    if getattr(adata, "mode", "memory") == "out-of-core" or _is_disk(_x_if_loaded(adata)):
        raise NotImplementedError(
            f"{what} needs the whole matrix in memory and is not available out of core "
            "yet; read the data with mode='memory' if it fits")


def _subset_prefix(adata: Any, dm: DiskMatrix) -> str:
    """Where the counts of the current genes are copied for PCA: the
    dataset's latents folder, named by the gene set so a copy is reused only
    for the same genes."""
    import hashlib

    genes = hashlib.sha1(np.packbits(np.asarray(dm.col_mask(), dtype=bool)).tobytes()).hexdigest()
    proj = _require_project("PCA out of core")
    name = getattr(adata, "name", None) or "dataset"
    return str(proj.run_dir(name) / "latents" / f"pca_genes_{genes[:12]}" / "counts")


def _as_u32(a) -> np.ndarray:
    """``a`` as uint32 without a copy where possible: SciPy stores indices
    as int32, whose non-negative values have the same bits as uint32."""
    a = np.asarray(a)
    if a.dtype == np.uint32:
        return a
    if a.dtype == np.int32:
        return a.view(np.uint32)
    return a.astype(np.uint32)


def _as_i32(a, max_value: int) -> np.ndarray:
    """The reverse of :func:`_as_u32`, for handing Rust's arrays to SciPy
    without the int32 copy it would otherwise make. ``max_value`` bounds the
    values (the last row pointer, or the column count for indices)."""
    a = np.asarray(a)
    if a.dtype == np.int32:
        return a
    if a.dtype == np.uint32 and max_value < 2**31:
        return a.view(np.int32)
    return a.astype(np.int64)


def _csr_from_arrays(indptr, indices, data, n_obs):
    """Construct a scipy CSR matrix from its arrays, reusing their buffers."""
    try:
        import scipy.sparse as sp

        indptr = np.asarray(indptr)
        nnz = int(indptr[-1]) if indptr.size else 0
        if max(nnz, n_obs) < 2**31:
            ip, ix = _as_i32(indptr, nnz), _as_i32(indices, n_obs)
        else:
            ip, ix = indptr.astype(np.int64), np.asarray(indices).astype(np.int64)
        return sp.csr_matrix((np.asarray(data), ix, ip), shape=(n_obs, n_obs), copy=False)
    except ImportError:
        # No scipy: return a lightweight namedtuple-like.
        from types import SimpleNamespace

        return SimpleNamespace(
            indptr=np.asarray(indptr),
            indices=np.asarray(indices),
            data=np.asarray(data),
            shape=(n_obs, n_obs),
        )
