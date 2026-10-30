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
    H5Writer,  # noqa: F401  (the writer of write_h5ad, polariseq._h5ad)
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
from polariseq._state import SourceMovedError
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
    "load",
    "marker_table",
    "SourceMovedError",
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

__version__ = "0.2.2"

# The active budget: what Polariseq may use. Detected from the machine on
# first use and overridable with `set_budget`. Detection is lazy so that
# importing the package never shells out or touches the filesystem.
_BUDGET: Budget | None = None


def _apply_threads(b: Budget, *, asked: bool) -> Budget:
    """Size the worker pool from the budget, and never fatally; returns the budget
    with the thread count the pool really has.

    Rayon's global pool is built on first use and cannot be resized. When the
    count was asked for (``set_budget(threads=...)``) and parallel work has
    already sized the pool otherwise, that is worth a warning: silently
    ignoring it would leave the user believing a setting they do not have.
    When nobody asked (the budget detected on first use, or ``set_budget``
    without ``threads``), there is nothing to warn about: the budget takes the
    pool's count, so the memory estimates that depend on it stay true. Asking
    again for the count the pool already has is a no-op, so this runs every
    time a budget is set rather than only once: an earlier "only once" flag let
    `set_budget(threads=4)` return quietly after the detected count had been
    applied first (measured 2026-09-19: 12 threads on a 12-thread laptop, 48
    on a cluster node).
    """
    try:
        _set_threads(b.threads)
    except RuntimeError as exc:
        running = _thread_count()
        if asked:
            import warnings

            warnings.warn(
                f"could not set the worker pool to {b.threads} threads ({exc}); "
                f"continuing with {running}",
                RuntimeWarning,
                stacklevel=3,
            )
        return b.with_overrides(threads=running)
    return b


#: Which defaults the steps use when an argument is left out (see :func:`set_workflow`).
WORKFLOWS = ("polariseq", "scanpy")
#: POLARISEQ_WORKFLOW=scanpy starts a process in the 'scanpy' workflow (benchmark harnesses, tests).
_WORKFLOW = os.environ.get("POLARISEQ_WORKFLOW", "polariseq")
if _WORKFLOW not in WORKFLOWS:
    raise ValueError(f"POLARISEQ_WORKFLOW must be one of {WORKFLOWS}, not {_WORKFLOW!r}")


def set_workflow(name: str) -> None:
    """Choose the default settings of the analysis steps.

    ``'polariseq'`` (the default): genes are ranked as Scanpy's seurat flavor ranks them, among the
    genes detected in more than 1% of the cells. In the one-call pipeline, each cell is normalized
    again over the selected genes (to 1,000, then log, as Scarf does) and each gene is standardized
    before PCA. UMAP starts from the multilevel spectral start, with a minimum distance of 0.5 and 100
    rounds above 10,000 cells. On the human fetal atlas these defaults gave clusters closer to the
    published cell types (normalized mutual information 0.80 against 0.55) and one coherent UMAP of
    the 4 million cells.

    ``'scanpy'``: the steps as Scanpy defines them: the seurat ranking over all genes, no second
    normalization, and UMAP from the spectral start with a minimum distance of 0.1 and 500 rounds up
    to 10,000 cells, otherwise 200. Use it to compare with Scanpy step by step.

    An argument you give explicitly always wins over the workflow.

    Args:
        name: ``'polariseq'`` or ``'scanpy'``.

    Examples:
        ```python
        ps.set_workflow("scanpy")
        ps.workflow()          # "scanpy"
        ```

    See also:
        ``ps.workflow``.
    """
    global _WORKFLOW
    if name not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {WORKFLOWS}, not {name!r}")
    _WORKFLOW = name


def workflow() -> str:
    """The workflow in force, set with ``set_workflow``.

    Returns:
        ``"polariseq"`` or ``"scanpy"``.

    See also:
        ``ps.set_workflow``.

    Examples:
        ```python
        ps.workflow()          # "polariseq" unless changed
        ```
    """
    return _WORKFLOW


def budget() -> Budget:
    """The budget in force: the memory and threads Polariseq may use.

    Detected the first time it is needed (three quarters of the available memory, and the machine's
    performance cores), unless ``set_budget`` set it.

    Returns:
        The budget.

    Examples:
        ```python
        print(ps.budget())
        ```

    See also:
        ``ps.set_budget``, ``ps.resources``.
    """
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = _apply_threads(Budget.detect(), asked=False)
    return _BUDGET


def resources() -> Resources:
    """What this machine has: its memory (total and available), cores and scratch space.

    ``available_source`` says how the available memory was found, since a measured figure and an
    estimate deserve different trust.

    Returns:
        The machine's resources.

    Examples:
        ```python
        >>> ps.resources()
        Resources(ram=8.6 GB total, 3.1 GB available [vm_stat], cores=4P/8, ...)
        ```

    See also:
        ``ps.budget``, ``ps.set_budget``.
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
    """Set the memory and threads Polariseq may use.

    Set it before the first step. Anything you leave out stays as detected. A pinned ``ram_gb`` also
    makes two runs of a script identical out of core, where the block size follows the budget.

    Args:
        ram_gb: The most memory, in gigabytes, Polariseq may hold.
        threads: The worker threads of every parallel step. They are set once per Python session.
        spill_dir: The folder for out-of-core scratch files.
        spill_gb: The most space those files may take.
        allow_lossy: Let the planner trade accuracy for fit. Off by default: Polariseq picks a slower
            exact plan over a faster approximate one unless asked, and reports any approximation it
            makes.

    Returns:
        The new budget.

    Examples:
        ```python
        ps.set_budget(ram_gb=6, threads=4)
        ```

    See also:
        ``ps.budget``, ``ps.plan``, ``ps.resources``.
    """
    global _BUDGET
    from polariseq import _checks

    for arg, value, whole in (("ram_gb", ram_gb, False), ("threads", threads, True),
                              ("spill_gb", spill_gb, False)):
        _checks.positive(arg, value, what="set_budget", integer=whole)
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
    _BUDGET = _apply_threads(_BUDGET, asked=threads is not None)
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
    """Predict what an analysis will need, before reading the data.

    Reads only the file's description (its numbers of cells, genes and stored values), then predicts
    the peak memory and disk of each step under the budget, names the step that needs the most, and
    chooses where the analysis would run: in memory when the matrix takes at most half the budget,
    otherwise out of core. It returns at once, whatever the size of the file.

    Args:
        path: The file; or several, as a list of paths or a dict from each dataset's label to its
            path, planned as the one dataset ``read_h5ad`` reads from them.
        genes: For several files, the genes kept: ``"intersection"`` (default) or ``"union"``.
        n_hvg, n_pcs, n_neighbors: The analysis settings that change the cost.
        storage: Plan for a given mode: ``"ram"`` (in memory) or ``"spill"`` (out of core).
        with_umap: Include the UMAP step.
        verbose: Print the report as well as returning the plan.

    Returns:
        The plan: ``fits`` (whether it fits the budget), ``peak_gb``, ``limiting_step``,
        ``rows_per_block``, the cost of every step, and for several files each one's shape and the
        genes it keeps (``datasets``).

    Examples:
        ```python
        plan = ps.plan("atlas.h5ad")
        print(plan.fits, plan.peak_gb, plan.limiting_step)
        ```

    See also:
        ``ps.set_budget``, ``ps.max_cells``, ``ps.read_h5ad``.
    """
    from polariseq import _checks

    if isinstance(path, (str, os.PathLike)):
        meta = scan_h5ad_meta(_checks.input_file(path, what="plan"))
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
        network_project=_project_on_network(),
    )
    p.datasets = datasets
    if verbose:
        print(p.report())
    return p


def _project_on_network() -> bool:
    """Whether the current project folder is on a network file system."""
    from polariseq.resources import is_network_filesystem

    proj = _current_project()
    return proj is not None and is_network_filesystem(str(proj.path))


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
    """The largest number of cells the budget can analyze.

    The question ``plan`` answers, the other way round: how large a dataset of a given density this
    machine can take, in memory or out of core.

    Args:
        n_genes: The genes in the dataset.
        nnz_per_cell: Stored (nonzero) values per cell, the density of the data.
        storage: ``None`` (the planner chooses), ``'ram'`` (in memory) or ``'spill'`` (out of core).
        **kwargs: Analysis settings that change the cost, as for ``plan`` (``n_hvg``, ``n_pcs``,
            ``n_neighbors``, ``with_umap``).

    Returns:
        The largest number of cells.

    Examples:
        ```python
        ps.max_cells()                    # the planner's choice of mode
        ps.max_cells(storage="spill")     # out of core
        ```

    See also:
        ``ps.plan``, ``ps.set_budget``.
    """
    return _max_cells(
        budget(), n_genes=n_genes, nnz_per_cell=nnz_per_cell, storage=storage, **kwargs
    )


def get_build_info() -> dict[str, Any]:
    """The versions of the Python package and of the compiled core it runs.

    Include it when you report a problem.

    Returns:
        ``python_version`` (the package), ``rust_core_version`` (the compiled core) and ``matched``:
        whether their release numbers agree.

    Examples:
        ```python
        print(ps.get_build_info())
        ```

    See also:
        ``ps.resources``.
    """
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


def _export_parts(adata: Any) -> tuple[Any, Any, Any, dict, dict, dict]:
    """What an export holds, with Scanpy's conventions: the matrix (as held:
    ``SparseMatrix``, ``DiskMatrix``, an array, or ``None``), ``obs`` and ``var``
    as data frames, and ``obsm``, ``obsp`` and ``uns`` with no memory-mapped
    arrays. Shared by :meth:`AnnData.to_anndata` and :meth:`AnnData.write_h5ad`."""
    import pandas as pd

    disk = bool((adata.uns.get("pca") or {}).get("disk_backed"))
    var_cols = {k: _as_column(v) for k, v in adata.var.items() if k != "name"}
    if disk:
        x = None
        names = adata.var.get("name")
        n = len(next(iter(var_cols.values()))) if var_cols else 0
        var_index = [str(g) for g in names] if names is not None else [str(i) for i in range(n)]
    else:
        x = adata.X
        var_index = [str(g) for g in adata.var_names]
    obs = pd.DataFrame({k: _as_column(v) for k, v in adata.obs.items()},
                       index=pd.Index([str(o) for o in adata.obs_names]))
    var = pd.DataFrame(var_cols, index=pd.Index(var_index))
    obsm = {k: np.asarray(v) for k, v in adata.obsm.items()}
    # Results a project keeps on disk are memory-mapped; the file formats cannot write those.
    obsp = {k: _in_memory(v) for k, v in adata.obsp.items()}
    uns = _in_memory(dict(adata.uns))
    batch_key = getattr(adata, "batch_key", None)
    if batch_key in obs:
        # As anndata.concat labels datasets: categorical, in file order.
        obs[batch_key] = pd.Categorical(obs[batch_key], categories=adata.labels)
    # Scanpy's conventions, so its tools (PAGA, plots by cluster) run on
    # the result as they would on Scanpy's own: clusterings as
    # categorical strings, and the neighbor graph's keys where Scanpy
    # looks for them.
    clusterings = {"louvain"}
    leiden = adata.uns.get("leiden")
    if isinstance(leiden, dict):
        clusterings.add((leiden.get("params") or {}).get("key_added", "leiden"))
    clusterings |= {k for k, v in adata.uns.items() if isinstance(v, dict) and "dendrogram" in v}
    for k in clusterings:
        if k in obs and np.issubdtype(obs[k].dtype, np.integer):
            v = obs[k].to_numpy()
            obs[k] = pd.Categorical(v.astype(str), categories=[str(c) for c in np.unique(v)])
    nb = adata.uns.get("neighbors")
    if isinstance(nb, dict) and "connectivities" in obsp:
        params = {k: v for k, v in nb.items()
                  if k not in ("connectivities", "distances", "params",
                               "connectivities_key", "distances_key")}
        scanpy_nb = {"connectivities_key": "connectivities",
                     "params": {"method": "umap", **params}}
        if "distances" in obsp:
            scanpy_nb["distances_key"] = "distances"
        uns["neighbors"] = scanpy_nb
    return x, obs, var, obsm, obsp, uns


def _in_memory(value: Any) -> Any:
    """``value`` with every memory-mapped array (also inside dicts and lists) as a plain array, so that
    an exported object can be written by anndata, h5py or pickle."""
    if isinstance(value, np.memmap):
        return np.asarray(value)
    if isinstance(value, dict):
        return {k: _in_memory(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_in_memory(v) for v in value)
    return value


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


def _scipy_sparse(what: str) -> Any:
    """``scipy.sparse``, which only the hand-over of results to other tools needs (the ``scipy`` extra)."""
    try:
        import scipy.sparse as sp
    except ImportError:
        raise ImportError(
            f"{what} returns a SciPy sparse matrix, and SciPy is not installed. Polariseq's own analysis "
            'does not need it; to hand results to other tools, install it: pip install scipy (or the extra, '
            'pip install "polariseq[scipy] @ git+https://github.com/DarkOxygen123/Polariseq.git")'
        ) from None
    return sp


def _sparse_to_scipy(m: SparseMatrix) -> Any:
    """Copy a :class:`SparseMatrix` out to a ``scipy.sparse.csr_matrix``."""
    sp = _scipy_sparse("SparseMatrix.to_scipy()")

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
    #: The steps that rewrote the values of an in-memory sparse matrix read
    #: from those files (normalize, log, renormalize, each with the genes it
    #: saw), so that a saved analysis rebuilds the matrix from the counts
    #: instead of storing it (``save``). ``None``: the values have another
    #: origin, and a save stores them.
    _xlog: list[dict[str, Any]] | None = None
    #: The analysis's name within its dataset (``from_counts``): its results go
    #: to ``runs/<dataset>/analyses/<analysis>/``. ``None``: the dataset's own.
    analysis: str | None = None
    #: The ``obs`` columns the analysis derived (clusters, scores, labels):
    #: ``from_counts`` leaves them out, keeping the cells' own annotations.
    _derived: set[str] | None = None

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
        self._xlog = None  # values of unknown origin; the steps' own subsets keep it (_set_x)

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
        """An independent copy of the analysis.

        Out of core the copy is a new view of the same counts on disk; the counts themselves are not
        copied.

        Returns:
            The copy.

        Examples:
            ```python
            backup = adata.copy()
            ```

        See also:
            ``adata.from_counts``.
        """
        return _subset(self, None, None)

    def from_counts(self, name: str, *, genes: str = "all") -> AnnData:
        """A new analysis of the same cells, starting again from their raw counts.

        For trying other settings on the same data (another number of genes, normalization or
        resolution) and comparing the results with ``tl.compare_analyses``. Out of core nothing is
        copied: the new analysis reads the same counts on disk. In memory the cells' counts are read again
        from the files the data came from.

        The cells are this analysis's current cells, after quality control and any doublet removal. Their
        own annotations come along: the files' columns, quality metrics, doublet scores and any column you
        added. What this analysis derived from its matrix does not: clusters, scores, labels, embeddings
        and graphs.

        Each analysis keeps its results in its own folder of the project
        (``runs/<dataset>/analyses/<name>/``), so analyses of one dataset never replace each other's
        files.

        Args:
            name: The new analysis's name (letters, digits, ``. _ + -``).
            genes: ``"all"`` (default): every gene of the source files, to filter and select again;
                ``"current"``: this analysis's genes.

        Returns:
            The new analysis; this one is unchanged.

        Examples:
            ```python
            b = adata.from_counts("hvg3000")
            ps.pp.normalize_total(b, target_sum=1e4)
            ps.pp.log1p(b)
            ps.pp.highly_variable_genes(b, n_top_genes=3000, subset=True)
            ```

        See also:
            ``ps.tl.compare_analyses``, ``adata.save``.
        """
        return _from_counts(self, name, genes)

    def save(self, name: str, *, overwrite: bool = False) -> Any:
        """Save the analysis in the project folder, to continue it later with ``ps.load``.

        The counts are not copied: the saved analysis points to the files they came from, and links the
        results the project folder already holds (embeddings, graphs, scores). Loading rebuilds the matrix
        exactly as the steps left it. If a source file is moved, loading stops and says where it was
        expected. The analysis is saved in ``<project>/states/<name>/``.

        To open the analysis in other tools, export it with ``write_h5ad`` or ``to_anndata``.

        Args:
            name: The saved analysis's name (letters, digits, ``. _ + -``).
            overwrite: Replace a saved analysis of that name.

        Returns:
            The folder written.

        Examples:
            ```python
            adata.save("clustered")
            adata = ps.load("my_analysis", "clustered")    # later, in a new session
            ```

        See also:
            ``ps.load``, ``adata.write_h5ad``.
        """
        from polariseq._state import save

        return save(self, name, overwrite=overwrite)

    def to_anndata(self) -> Any:
        """Convert to an ``anndata.AnnData``, for Scanpy and the tools built on it.

        ``X`` becomes a SciPy sparse matrix (or stays a NumPy array), and ``obs``, ``var``, ``obsm``,
        ``obsp`` and ``uns`` are carried over. The result of ``process_diskbacked`` never held the
        matrix in memory, so it converts without ``X``, with its per-gene statistics in ``var`` and every
        per-cell result. Needs the ``anndata`` package.

        Returns:
            An ``anndata.AnnData``.

        Examples:
            ```python
            import scanpy as sc

            ad = adata.to_anndata()
            sc.pl.umap(ad, color="leiden")
            ```

        See also:
            ``adata.write_h5ad``.
        """
        import anndata as ad

        x, obs, var, obsm, obsp, uns = _export_parts(self)
        if _is_disk(x):
            x = None  # an out-of-core matrix stays on disk (write_h5ad streams it into a file)
        elif _is_sparse(x):
            x = x.to_scipy()
        out = ad.AnnData(X=x, obs=obs, var=var)
        for k, v in obsm.items():
            out.obsm[k] = v
        for k, v in obsp.items():
            out.obsp[k] = v
        out.uns.update(uns)
        return out

    def write_h5ad(self, path: str | os.PathLike, *, counts: bool = False) -> Any:
        """Write the analysis as an ``.h5ad`` file for other tools (Scanpy, cellxgene, R).

        ``X`` holds the matrix as the analysis left it: the normalized (and log-transformed) values of the
        current cells and genes. Out of core it is written a few blocks at a time, so an atlas is written
        within the memory budget. Annotations, embeddings, graphs and results follow, with Scanpy's
        conventions.

        ``uns['polariseq']`` records where the raw counts came from (the source files and which of their
        cells and genes the file holds) and the steps that made ``X``; other tools ignore it. With
        ``counts=True`` the raw counts are written too, as ``layers['counts']``, for someone without the
        source files; the file grows accordingly.

        Needs no ``anndata`` package. The file is written under a temporary name and renamed when
        complete, so an interrupted write never leaves a broken file at ``path``.

        Args:
            path: The ``.h5ad`` file to write (replaced if it exists).
            counts: Also write the raw counts, as ``layers['counts']``.

        Returns:
            The path written.

        Examples:
            ```python
            adata.write_h5ad("analysis.h5ad", counts=True)
            ```

        See also:
            ``adata.to_anndata``, ``adata.save``.
        """
        from polariseq._h5ad import write_h5ad

        return write_h5ad(self, path, counts=counts)


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
        self._xlog: list[dict[str, Any]] | None = []  # the counts as read (see AnnData._xlog)
        self.analysis: str | None = None  # see AnnData.analysis
        self._derived: set[str] | None = None

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
        """The raw values of the first ``n`` cells, read without opening the whole matrix.

        For an ``.h5ad`` file only those cells are read. 10x files cannot be read in part, so they are
        read whole first; such files are usually small.

        Args:
            n: Number of cells, from the first.

        Returns:
            A dict with ``data``, ``indices``, ``indptr``, ``n_cells`` and ``n_genes``: the cells' raw
            values as compressed sparse rows.

        Examples:
            ```python
            first = adata.head(50)
            ```

        See also:
            ``adata.sample``.
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
        """The raw values of ``n`` cells drawn at random, read without opening the whole matrix.

        Args:
            n: Number of cells.
            seed: Random seed.

        Returns:
            A SciPy sparse (CSR) matrix of the cells' raw values.

        Examples:
            ```python
            cells = adata.sample(1000)
            ```

        See also:
            ``adata.head``.
        """
        sp = _scipy_sparse("sample()")
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
        self._xlog = None

    @property
    def out_dir(self) -> Any:
        """This dataset's folder in the current project (``None`` without one):
        its on-disk copy of the counts and its intermediates."""
        proj = _current_project()
        return proj.run_dir(self.name) if proj is not None else None

    def to_anndata(self) -> Any:
        """Convert to an ``anndata.AnnData`` (see :meth:`AnnData.to_anndata`)."""
        return AnnData.to_anndata(self)

    def write_h5ad(self, path: str | os.PathLike, *, counts: bool = False) -> Any:
        """Write the analysis as an ``.h5ad`` file (see :meth:`AnnData.write_h5ad`)."""
        from polariseq._h5ad import write_h5ad

        self._materialize()
        return write_h5ad(self, path, counts=counts)

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

    def from_counts(self, name: str, *, genes: str = "all") -> Any:
        """A new analysis of the same cells from their raw counts (see :meth:`AnnData.from_counts`)."""
        self._materialize()
        return _from_counts(self, name, genes)

    def save(self, name: str, *, overwrite: bool = False) -> Any:
        """Save the analysis in the project folder (see :meth:`AnnData.save`)."""
        from polariseq._state import save

        self._materialize()
        return save(self, name, overwrite=overwrite)

    def close(self) -> None:
        """Free the memory and temporary files the matrix holds.

        Afterwards the analysis keeps only the file's description; the matrix is read again when a step
        next needs it.

        Examples:
            ```python
            adata.close()
            ```

        See also:
            ``adata.head``.
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
        """Check the file without reading its matrix.

        Returns:
            A dict: the file's ``path``, ``format``, whether it ``exists``, its ``size_mb``, the shape
            (``n_obs``, ``n_vars``), the stored values (``nnz``, ``sparsity``, ``encoding``), the
            estimated memory (``est_ram_gb``) and how long the check took (``scan_time_s``).

        Examples:
            ```python
            print(adata.validate())
            ```

        See also:
            ``adata.head``, ``ps.plan``.
        """
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
    """Open a 10x Genomics ``.h5`` file, as Cell Ranger writes it.

    10x files are analyzed in memory. To analyze one out of core, write it to an ``.h5ad`` file
    first (``ps.read_10x_h5(path).write_h5ad("data.h5ad")``) and open that.

    Args:
        path: The ``filtered_feature_bc_matrix.h5`` file.
        lazy: Read the matrix when a step first needs it (default); ``False`` reads it at once.

    Returns:
        The analysis: a ``LazyAnnData`` (default), or an ``AnnData`` with ``lazy=False``.

    Examples:
        ```python
        adata = ps.read_10x_h5("filtered_feature_bc_matrix.h5")
        ```

    See also:
        ``ps.read_10x_mtx``, ``ps.read_h5ad``.
    """
    from polariseq import _checks

    path = _checks.input_file(path, what="read_10x_h5", kind="h5")
    if lazy:
        return LazyAnnData(path, "10x_h5")
    result = _read_10x_h5_raw(path)
    return _tenx_dict_to_anndata(result)


def read_10x_mtx(dir: str, *, lazy: bool = True) -> AnnData | LazyAnnData:
    """Open a 10x Genomics matrix folder.

    The folder holds ``matrix.mtx``, ``barcodes.tsv`` and ``features.tsv`` (or ``genes.tsv``), plain
    or gzipped. It is analyzed in memory.

    Args:
        dir: The folder.
        lazy: Read the matrix when a step first needs it (default); ``False`` reads it at once.

    Returns:
        The analysis: a ``LazyAnnData`` (default), or an ``AnnData`` with ``lazy=False``.

    Examples:
        ```python
        adata = ps.read_10x_mtx("filtered_gene_bc_matrices/hg19/")
        ```

    See also:
        ``ps.read_10x_h5``, ``ps.read_h5ad``.
    """
    from polariseq import _checks

    dir = _checks.input_file(dir, what="read_10x_mtx", kind="mtx")
    if lazy:
        return LazyAnnData(dir, "10x_mtx")
    result = _read_10x_mtx_raw(dir)
    return _tenx_dict_to_anndata(result)


def read_h5ad(path: str | list[str] | dict[str, str], *, sparse: bool = True,
              lazy: bool = True, mode: str = "auto", name: str | None = None,
              batch_key: str = "dataset",
              genes: str = "intersection",
              obs_columns: list[str] | None = None) -> AnnData | LazyAnnData:
    """Open an ``.h5ad`` file, or several files as one dataset.

    Opening reads only the file's description: its cells, genes and annotations. The matrix is read
    when a step first needs it. The planning step then chooses where the analysis runs from the size
    of the matrix and the budget: in memory when it fits, otherwise out of core, with the counts copied
    once to the project folder and read in blocks. ``adata.mode`` says which; the steps and their
    results are the same either way.

    Several files are analyzed as one dataset: their cells one after another over one set of genes,
    matched by name, each cell labelled with its file (the file name, or the dict's key) in
    ``obs[batch_key]``. The matrix is the one ``anndata.concat`` would build, without the files being
    merged first.

    Args:
        path: The ``.h5ad`` file; or several, as a list of paths or a dict from each dataset's label
            to its path.
        sparse: Keep a sparse matrix sparse (default); ``False`` reads it as a dense array.
        lazy: Read the matrix when a step first needs it (default); ``False`` reads it at once.
        mode: Where the analysis runs: ``"auto"`` (default) lets the planning step choose,
            ``"memory"`` or ``"out-of-core"`` fixes it.
        name: The dataset's folder in the project (default: the file name without its extension; for
            several files, their labels joined by ``+``).
        batch_key: For several files, the ``obs`` column of each cell's file.
        genes: For several files, the genes kept: ``"intersection"`` (default; the genes every file
            has, in the first file's order) or ``"union"`` (every gene, zero where a file lacks it),
            ordered as ``anndata.concat`` orders them.
        obs_columns: The file's cell annotations to read into ``obs`` (default: all of them). For
            several files, a column one file lacks is missing for its cells.

    Returns:
        The analysis: a ``LazyAnnData`` (default), or an ``AnnData`` with ``lazy=False``.

    Examples:
        ```python
        ps.project("my_analysis")
        adata = ps.read_h5ad("data.h5ad")
        adata = ps.read_h5ad({"control": "ctrl.h5ad", "treated": "stim.h5ad"},
                             batch_key="condition")
        ```

    See also:
        ``ps.plan``, ``ps.read_10x_h5``, ``ps.read_10x_mtx``, ``ps.load``.
    """
    from polariseq import _checks

    if not isinstance(path, (str, os.PathLike)):
        for _, p in _sources(path):
            _checks.input_file(p, what="read_h5ad")
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
        out._xlog = []
        return out
    path = _checks.input_file(path, what="read_h5ad")
    if lazy:
        return LazyAnnData(path, "h5ad", mode=mode, name=name, obs_columns=obs_columns)
    if mode == "out-of-core":
        raise ValueError("an out-of-core analysis is lazy: drop lazy=False")
    shape = scan_h5ad(path)
    if shape["n_elements"] > MAX_IN_MEMORY_VALUES:
        raise ValueError(_too_large_for_memory(shape["n_elements"], path))
    _warn_if_over_budget(shape["n_obs"], shape["n_vars"], shape["n_elements"], path)
    return _read_h5ad_eager(path, sparse, obs_columns)


def load(project: str | os.PathLike, name: str, *, relink: dict[str, str] | None = None) -> Any:
    """Open an analysis saved with ``adata.save``, to continue where it stopped.

    Declares ``project`` the project folder, as ``project`` does, and returns the analysis as it was
    saved: its cells and genes, the matrix as the steps left it, and every annotation, embedding,
    graph and result. Loading is quick, because large results are read only when they are used.

    Args:
        project: The project folder the analysis was saved in.
        name: The saved analysis's name.
        relink: New places of source files that have moved, ``{old path: new path}``.

    Returns:
        The saved analysis, the same kind of object ``read_h5ad`` returns.

    Raises:
        SourceMovedError: A source file is no longer where it was; the message says where it was
            expected.

    Examples:
        ```python
        adata = ps.load("my_analysis", "clustered")
        ```

    See also:
        ``adata.save``, ``ps.project``.
    """
    from polariseq._state import load as _load

    return _load(project, name, relink=relink)


def marker_table(source: str | list[str] | tuple[str, ...] | None = ("curated", "hpa"), *,
                 tissue: str | list[str] | None = None,
                 species: str = "human", markers: Any = None) -> Any:
    """The marker genes ``tl.auto_annotate`` would score with, as a table.

    Use it to look at a source before using it, or to start a table of your own: filter it, add rows,
    and pass it back as ``markers=`` with ``source=None``.

    Args:
        source: A marker source or a list of them: ``"curated"``, ``"hpa"``, ``"msigdb_c8"``
            (shipped), ``"accordion"``, ``"cellmarker"``, ``"panglaodb"``, ``"sctype"`` (downloaded
            once on first use). ``None`` with ``markers``: only yours.
        tissue: Keep this tissue's types, a name or a list. Immune types and types found in many
            tissues are kept with any tissue. ``None``: all.
        species: ``"human"`` or ``"mouse"``; human-only sources are mapped to mouse genes.
        markers: Your own markers, in any form ``auto_annotate`` takes.

    Returns:
        A ``pandas.DataFrame`` with one row per marker: ``cell_type``, ``gene``, ``sign`` (+1
        expressed, -1 not expressed), ``evidence`` (how many records name it), ``strength`` (the
        source's own ranking value, 0 when it has none), ``tissue``, ``cl_id`` (the Cell Ontology
        term), ``species`` and ``source``.

    Examples:
        ```python
        table = ps.marker_table(tissue="immune")
        table[table["cell_type"] == "B cell"]
        ```

    See also:
        ``ps.tl.auto_annotate``.
    """
    from polariseq._markers import marker_table as _marker_table

    return _marker_table(source, tissue=tissue, species=species, markers=markers)


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
    out._xlog = []
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
        """Find droplets that held two cells (doublets) with Scrublet.

        Doublets are simulated by adding up the counts of random pairs of cells, and each cell is scored
        by the share of simulated doublets among its nearest neighbors. Cells scoring above a threshold
        are called doublets; the threshold is found from the simulated doublets' scores unless you give
        one.

        Doublets form within one capture (one 10x channel), so score each capture on its own by passing
        the ``obs`` column that names it as ``batch_key``. Scrublet works on raw counts: in memory, run it
        before ``normalize_total``; out of core it reads the counts from the copy on disk, so it can run at
        any point.

        Writes ``obs['doublet_score']``, ``obs['predicted_doublet']`` and ``uns['scrublet']``: for each
        capture, the threshold, the detected and expected doublet rates, and the simulated doublets'
        scores, which ``ps.pl.scrublet_score_distribution`` draws. The names are those of Scanpy's
        ``pp.scrublet``.

        The method is Scrublet (Wolock, Lopez and Klein 2019), as its reference implementation computes
        it.

        Args:
            adata: The analysis, on raw counts.
            batch_key: The ``obs`` column of the capture each cell came from.
            sim_doublet_ratio: Doublets simulated per cell.
            expected_doublet_rate: The share of doublets you expect from the loading density.
            stdev_doublet_rate: The uncertainty of that share.
            n_neighbors: Neighbors per cell (default ``round(0.5 * sqrt(n))``).
            n_prin_comps: Principal components used to find the neighbors.
            min_counts, min_cells: A gene is used if it reaches ``min_counts`` normalized counts in at
                least ``min_cells`` cells.
            min_gene_variability_pctl: Genes are used above this percentile of variability.
            threshold: The score above which a cell is called a doublet.
            random_state: Random seed of the simulation.

        Examples:
            ```python
            ps.pp.scrublet(adata, batch_key="sample")
            ps.pl.scrublet_score_distribution(adata)
            adata = adata[~np.asarray(adata.obs["predicted_doublet"])]
            ```

        See also:
            ``ps.pl.scrublet_score_distribution``, ``ps.pp.filter_cells``.
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
            from polariseq import _checks

            _checks.column(adata, batch_key, what="scrublet batch_key")
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
        """Quality metrics for every cell and gene.

        Counts each cell's total counts, its detected genes and the share of its counts from
        mitochondrial genes (gene names starting with ``mito_prefix``), and each gene's number of cells,
        total and mean counts. Use them to choose the thresholds of ``filter_cells`` and
        ``filter_genes``.

        Writes ``obs['n_counts']``, ``obs['n_genes']`` and ``obs['pct_counts_mt']``, and
        ``var['n_cells']``, ``var['total_counts']`` and ``var['mean_counts']``.

        Args:
            adata: The analysis, before normalization.
            mito_prefix: The start of mitochondrial gene names (``"MT-"`` for human, ``"mt-"`` for
                mouse).

        Examples:
            ```python
            ps.pp.calculate_qc_metrics(adata)
            ps.pl.violin(adata, keys=["n_genes", "n_counts", "pct_counts_mt"])
            ```

        See also:
            ``ps.pp.filter_cells``, ``ps.pp.filter_genes``, ``ps.pl.qc_distributions``.
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
        """Remove cells outside quality thresholds.

        Keeps the cells whose total counts, detected genes and mitochondrial share lie within the limits
        given; a limit left out removes nothing. Needs ``calculate_qc_metrics`` first.

        Args:
            adata: The analysis, after ``calculate_qc_metrics``.
            min_counts: The fewest total counts a cell may have.
            max_counts: The most total counts a cell may have.
            min_genes: The fewest detected genes a cell may have.
            max_genes: The most detected genes a cell may have.
            max_pct_mt: The largest share of mitochondrial counts, in percent (needs mitochondrial genes
                in the data).
            inplace: Remove the cells from ``adata`` (default); ``False`` returns a filtered copy and
                leaves ``adata`` as it is.

        Returns:
            ``None``, or the filtered copy with ``inplace=False``.

        Examples:
            ```python
            ps.pp.calculate_qc_metrics(adata)
            ps.pp.filter_cells(adata, min_genes=200, max_pct_mt=20)
            ```

        See also:
            ``ps.pp.calculate_qc_metrics``, ``ps.pp.filter_genes``, ``ps.pl.qc_distributions``.
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
                from polariseq import _checks

                raise _checks.InputError(
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
        """Remove genes detected in too few cells.

        Keeps the genes detected in at least ``min_cells`` cells and with at least ``min_counts`` counts in
        total, and writes the remaining genes' ``var['n_cells']``, ``var['total_counts']`` and
        ``var['mean_counts']``.

        Args:
            adata: The analysis.
            min_cells: The fewest cells a gene must be detected in.
            min_counts: The fewest total counts a gene must have.
            inplace: Remove the genes from ``adata`` (default); ``False`` returns a filtered copy and
                leaves ``adata`` as it is.

        Returns:
            ``None``, or the filtered copy with ``inplace=False``.

        Examples:
            ```python
            ps.pp.filter_genes(adata, min_cells=3)
            ```

        See also:
            ``ps.pp.filter_cells``, ``ps.pp.highly_variable_genes``.
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
                _set_x(adata, adata.X[mask, :])
        else:
            _set_x(adata, adata.X[mask])
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
            _set_x(adata, adata.X[:, mask])
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
        """Scale each cell to the same total count.

        Divides each cell's counts by its total and multiplies them by ``target_sum``, so that cells
        sequenced to different depths can be compared. Usually followed by ``log1p``. Out of core the
        counts on disk are left unchanged: the scale is applied as they are read.

        Args:
            adata: The analysis, on raw counts.
            target_sum: The total each cell is scaled to.

        Examples:
            ```python
            ps.pp.normalize_total(adata, target_sum=1e4)
            ps.pp.log1p(adata)
            ```

        See also:
            ``ps.pp.log1p``, ``ps.pp.renormalize``.
        """
        adata = self._ensure_materialized(adata)
        if _is_sparse(adata.X) or _is_disk(adata.X):
            totals = adata.X.normalize_total(float(target_sum))  # in place (a view out of core)
            if _is_sparse(adata.X):
                # each cell's total, so that renormalize can turn values back into counts in place
                adata.uns.setdefault("normalize", {})["cell_totals"] = np.asarray(totals, dtype=np.float32)
                _log_x(adata, {"op": "normalize_total", "target_sum": target_sum})
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
        """Normalize each cell again over the selected genes only.

        After gene selection with ``subset=True``, scales each cell to ``target_sum`` by its total over the
        selected genes, then takes the logarithm. This is Scarf's normalization (Dhapola et al. 2022), and
        the default recipe's step before ``pca(scale=True)``.

        In memory the values are turned back into counts and normalized again, using the totals that
        ``normalize_total`` recorded; pass ``counts`` only when cells were removed after normalizing. Out
        of core the normalization is a view on the counts on disk and is simply replaced.

        Writes ``uns['normalize']`` and ``uns['log1p']``.

        Args:
            adata: The analysis, after ``highly_variable_genes(..., subset=True)``.
            target_sum: The total each cell is scaled to.
            log1p: Take the logarithm after scaling.
            counts: In memory, the raw counts of the same cells (genes matched by name), when cells were
                removed after ``normalize_total``.

        Examples:
            ```python
            ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
            ps.pp.renormalize(adata)
            ps.pp.pca(adata, n_components=50, scale=True)
            ```

        See also:
            ``ps.pp.highly_variable_genes``, ``ps.pp.pca``.
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
            _log_x(adata, {"op": "renormalize", "from_log": "log1p" in adata.uns,
                           "target_sum": target_sum, "log1p": bool(log1p)})
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
        """Take the logarithm of the expression values, ``log(1 + x)``.

        Compresses the range of the values, so that a few very high counts do not dominate. Zeros stay
        zero, so a sparse matrix stays sparse. Writes ``uns['log1p']``.

        Args:
            adata: The analysis, after ``normalize_total``.

        Examples:
            ```python
            ps.pp.normalize_total(adata, target_sum=1e4)
            ps.pp.log1p(adata)
            ```

        See also:
            ``ps.pp.normalize_total``, ``ps.pp.highly_variable_genes``.
        """
        adata = self._ensure_materialized(adata)
        if _is_sparse(adata.X) or _is_disk(adata.X):
            adata.X.log1p()  # in place (a view out of core)
            if _is_sparse(adata.X):
                _log_x(adata, {"op": "log1p"})
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
        include: list[str] | None = None,
    ) -> None:
        """Select the genes that vary most between cells.

        Ranks the genes by how much more variable they are than genes of similar mean expression and
        marks the top ``n_top_genes`` in ``var['highly_variable']``. With ``subset=True`` only those genes
        are kept, as most analyses do before PCA. Run it on log-normalized data.

        Two rankings:

        - ``flavor='seurat'``: Seurat's dispersion method (Satija et al. 2015), computed step for step as
          Scanpy computes it, so the selected genes agree with Scanpy's on the same data. Writes
          ``var['means']``, ``var['dispersions']`` and ``var['dispersions_norm']``.
        - ``flavor='scarf'``: Scarf's ranking (Dhapola et al. 2022). Genes are binned by mean expression,
          a smooth curve (LOWESS) is fitted through each bin's least variable gene, and genes are ranked
          by their variance above that curve. Writes ``var['log_mean']``, ``var['log_var']``,
          ``var['corrected_var']`` and ``var['detected']``.

        In the default ``'polariseq'`` workflow, genes are ranked as ``'seurat'`` among those detected in
        more than 1% of the cells, so that genes seen in a handful of cells do not crowd out the
        informative ones. The settings are kept in ``uns['hvg']``.

        Args:
            adata: The analysis, after ``log1p``.
            n_top_genes: The number of genes to select.
            flavor: ``'seurat'`` or ``'scarf'``. Left out, the workflow decides (``ps.set_workflow``):
                ``'polariseq'`` ranks as ``'seurat'`` among the genes detected in more than 1% of the
                cells, ``'scanpy'`` as ``'seurat'`` over every gene.
            subset: Keep only the selected genes.
            min_cells: A gene must be detected in more cells than this to be ranked. For ``'scarf'`` the
                default is 1% of the cells; for ``'seurat'`` it is no floor, as in Scanpy.
            n_bins: The number of mean-expression bins (default 20 for ``'seurat'``, 200 for
                ``'scarf'``).
            lowess_frac: ``'scarf'`` only: the share of the bins' genes in each local fit of the curve.
            include: Genes to keep whatever the ranking says, by name. They are added to the top
                ``n_top_genes`` and marked in ``var['included']``. A name that is not among the current
                genes is an error that says why and suggests the closest names.

        Examples:
            ```python
            ps.pp.highly_variable_genes(adata, n_top_genes=2000)
            ps.pl.highly_variable_genes(adata)
            ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, include=["CD8A"])
            ```

        See also:
            ``ps.pp.renormalize`` and ``ps.pp.pca``, the steps that follow. ``ps.pl.highly_variable_genes``
            plots the genes this selects.
        """
        from polariseq import _checks

        if int(n_top_genes) <= 0:
            raise ValueError(f"highly_variable_genes: n_top_genes must be positive, not {n_top_genes}")
        if int(n_top_genes) >= adata.n_vars:
            import warnings

            warnings.warn(f"highly_variable_genes: n_top_genes={n_top_genes:,}, but the data hold "
                          f"{adata.n_vars:,} genes; every gene is selected", stacklevel=2)
        if flavor is None:
            flavor = "seurat"
            if _WORKFLOW == "polariseq" and min_cells is None:
                min_cells = int(0.01 * adata.n_obs)
        if flavor not in ("seurat", "scarf"):
            raise NotImplementedError(
                f"flavor={flavor!r} is not implemented; 'seurat' and 'scarf' are available"
            )
        adata = self._ensure_materialized(adata)
        forced = None
        if include is not None:
            forced = _checks.gene_positions(adata.var_names, include, what="include",
                                            source_names=_source_gene_names(adata))
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
        if forced is not None:
            hvg_mask = _add_included(adata, X, hvg_mask, forced, floor)

        # Auto-subset if requested (updates X, var, and var_names together).
        if subset:
            if _is_disk(adata.X):
                adata.X.retain_cols_mask(np.ascontiguousarray(hvg_mask, dtype=np.bool_))
            else:
                _set_x(adata, adata.X[:, hvg_mask])
            adata.var = {k: np.asarray(v)[hvg_mask] for k, v in adata.var.items()}
            adata._cols = _keep_positions(getattr(adata, "_cols", None), np.flatnonzero(hvg_mask))
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
        """Principal component analysis of the expression matrix.

        Finds the directions along which the cells vary most and gives each cell its position along the
        first ``n_components`` of them: the embedding the neighbor graph and Harmony work on. Run it after
        gene selection. With ``scale=True`` each gene is standardized first, so that highly expressed
        genes do not dominate the components.

        Writes ``obsm['X_pca']`` (cells by components) and ``uns['pca']``: ``variance_ratio``, the share
        of the variance each component explains, and ``n_components``.

        The components are found with randomized SVD (Halko et al. 2011) for large matrices and with an
        exact decomposition for smaller ones.

        Args:
            adata: The analysis.
            n_components: Number of components.
            center: Subtract each gene's mean first, as PCA usually does.
            scale: Standardize each gene to unit variance first, as ``sc.pp.scale`` does; implies
                ``center``. The matrix itself is left unchanged.
            max_value: With ``scale``, the largest absolute scaled value kept; larger ones are clipped.
            seed: Random seed of the randomized method.

        Examples:
            ```python
            ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
            ps.pp.pca(adata, n_components=50, scale=True)
            ps.pl.pca_variance_ratio(adata)
            ```

        See also:
            ``ps.pl.pca_variance_ratio``, ``ps.pp.harmony_integrate``, ``ps.pp.neighbors``.
        """
        if max_value is not None and not scale:
            raise ValueError("max_value only applies with scale=True")
        if scale and not center:
            raise ValueError("scale=True always centers; drop center=False")
        adata = self._ensure_materialized(adata)
        most = min(adata.n_obs, adata.n_vars)
        if not 0 < int(n_components) <= most:
            raise ValueError(f"pca: n_components={n_components}, but {adata.n_obs:,} cells x "
                             f"{adata.n_vars:,} genes give at most {most:,} components")
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
        """Build the neighbor graph: each cell's nearest cells in an embedding.

        Finds each cell's ``n_neighbors`` nearest cells in ``obsm[use_rep]`` (the PCA embedding by default,
        or a corrected one such as ``X_pca_harmony``) and turns them into the weighted graph that
        clustering and UMAP work on. Above 10,000 cells the neighbors are found approximately with
        NN-Descent (Dong, Moses and Li 2011), which is far faster; below, exactly.

        Writes ``obsp['distances']`` (each cell's neighbors and their distances), ``obsp['connectivities']``
        (the weighted graph, built as UMAP builds it; McInnes et al. 2018) and ``uns['neighbors']``.

        Args:
            adata: The analysis, after ``pca``.
            n_neighbors: Neighbors per cell.
            use_rep: The ``obsm`` embedding to search.
            n_pcs: How many of its columns to use (default: all).
            metric: ``'euclidean'`` or ``'cosine'``.
            approximate: ``True`` always uses NN-Descent, ``False`` always the exact search; left out,
                the exact search is used up to 10,000 cells.
            seed: Random seed of NN-Descent.

        Examples:
            ```python
            ps.pp.neighbors(adata, n_neighbors=15)
            ps.pp.neighbors(adata, use_rep="X_pca_harmony")      # after batch correction
            ```

        See also:
            ``ps.tl.leiden``, ``ps.tl.umap``, ``ps.pp.harmony_integrate``.
        """
        from polariseq import _checks

        adata = self._ensure_materialized(adata)
        _checks.result(adata, use_rep, what="neighbors", where="obsm", made_by=_checks.made_by(use_rep))
        embedding = adata.obsm[use_rep]
        if n_pcs is not None:
            embedding = embedding[:, :n_pcs]
        n_obs = embedding.shape[0]
        # The embedding is read into memory when it fits beside NN-Descent's
        # working memory, else in place (`_results.graph_inputs`).
        from polariseq.planning import neighbors_working_bytes

        k = max(1, n_neighbors - 1)
        held = _held_bytes(adata)
        (embedding,), emb_inputs, emb_cache = _results.graph_inputs(
            [embedding], held, neighbors_working_bytes(n_obs, n_neighbors, budget().threads))
        with emb_cache:   # removes a local copy of the embedding, if one was made, at the end
            emb = np.ascontiguousarray(embedding, dtype=np.float32)

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

            emb = search = embedding = None  # the embedding (or its copy) is not needed further
        # The distance graph, read in random order for the symmetric union:
        # into memory when it fits, else in place.
        (f_ip, f_ix, f_dx), dist_inputs, dist_cache = _results.graph_inputs(
            [d_ip, d_ix, d_dx], held, 0)
        with dist_cache:
            fuzzy_connectivities_into(_as_u32(f_ip), _as_u32(f_ix), f_dx, n_obs, alloc)
        del f_ip, f_ix, f_dx
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
            "inputs": {"embedding": emb_inputs, "distances": dist_inputs},
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

        Moves the cells in the principal components so that batches mix while cell types stay apart.
        Cells are grouped softly into clusters that are pushed to hold every batch in proportion; within
        each cluster each batch's offset is estimated, and cells move back by their clusters' offsets.
        Grouping and correction alternate until the result stops improving.

        Writes the corrected embedding to ``obsm[key_added]`` (``X_pca_harmony``, as in Scanpy) and the
        settings to ``uns['harmony']``; the uncorrected embedding stays in ``use_rep``, so mixing can be
        compared before and after. Build the graph on the corrected one.

        The method is Harmony (Korsunsky et al. 2019), as R harmony 2 and harmonypy 2 run it; the
        defaults are harmonypy's.

        Args:
            adata: The analysis, after ``pca``.
            batch_key: The ``obs`` column of each cell's batch, or a list of columns to correct for
                together (for example ``["donor", "lane"]``).
            use_rep: The ``obsm`` embedding to correct.
            key_added: The ``obsm`` key of the corrected embedding.
            theta: How strongly to mix the batches, one value for all batch variables or one each.
            sigma: Width of the soft clustering; smaller is closer to hard clusters.
            lamb: Ridge penalty, one for all variables or one each. ``None`` estimates it per cluster, as
                harmonypy 2 does (``alpha`` times the expected number of cells).
            n_clusters: Number of clusters (default ``min(n_cells / 30, 100)``).
            tau: Protection of small batches against overcorrection (0: off).
            block_size: The share of cells reassigned at a time.
            max_iter_harmony: The most rounds of clustering and correction.
            max_iter_kmeans: Assignment rounds per clustering.
            epsilon_cluster: Convergence threshold of the assignment rounds.
            epsilon_harmony: Convergence threshold of the rounds of clustering and correction.
            alpha: Scale of the estimated ridge penalty.
            seed: Random seed.

        Examples:
            ```python
            ps.pp.pca(adata, n_components=50, scale=True)
            ps.pp.harmony_integrate(adata, "sample")
            ps.pp.neighbors(adata, use_rep="X_pca_harmony")
            ```

        See also:
            ``ps.pp.neighbors``, ``ps.pl.umap``.
        """
        from polariseq import _checks
        from polariseq._compare import categorical

        keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
        if not keys:
            raise ValueError("batch_key names no column")
        codes, n_levels = [], []
        for key in keys:
            _checks.column(adata, key, what="harmony_integrate batch_key")
            cat = categorical(adata.obs[key]).remove_unused_categories()
            c = np.asarray(cat.codes)
            if (c < 0).any():
                raise ValueError(f"adata.obs['{key}'] has missing values; give every cell a batch")
            codes.append(c.astype(np.uint32))
            n_levels.append(len(cat.categories))
        if all(n < 2 for n in n_levels):
            raise ValueError("Need >= 2 batches for correction")
        _checks.result(adata, use_rep, what="harmony_integrate", where="obsm",
                       made_by=_checks.made_by(use_rep))
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
    """Deprecated: use ``ps.read_h5ad([...])`` and ``ps.pp.harmony_integrate(adata, 'dataset')``.

    Each dataset is preprocessed (normalize, log1p, HVG, PCA) on its own and the
    PCA embeddings are concatenated. Each file gets its own PCA, so the joined
    embeddings lie in different bases and are not comparable across datasets.
    Kept so that existing scripts run; it warns when called.

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
        """Find the marker genes of each group.

        Compares each group's cells with all other cells (``reference="rest"``) or with one group, gene by
        gene, with Welch's t-test (Welch 1947) as Scanpy's ``rank_genes_groups(method="t-test")`` does,
        and ranks the genes by their t-statistic. P-values are adjusted for the number of genes tested
        (Benjamini and Hochberg 1995).

        Writes ``uns['rank_genes_groups']`` (for each group, the top genes with their scores, log fold
        changes and p-values) and ``uns['rank_genes_groups_params']``. ``ps.get.markers`` turns it into a
        table. To compare conditions with donors as replicates, use ``compare_conditions``.

        Args:
            adata: The analysis, log-normalized.
            groupby: The ``obs`` column of the groups: clusters, or any labels.
            groups: The groups to rank (default: all but the reference).
            reference: ``"rest"``, or the group the others are compared with.
            n_genes: How many top genes to keep per group.

        Examples:
            ```python
            ps.tl.rank_genes_groups(adata, groupby="leiden")
            ps.get.markers(adata, n=5)
            ps.pl.rank_genes_groups(adata, n_genes=8)
            ```

        See also:
            ``ps.get.markers`` gives the result as a table. ``ps.pl.rank_genes_groups`` plots the top genes
            of each group. ``ps.tl.compare_conditions`` compares conditions with donors as replicates.
        """
        from polariseq._compare import categorical

        src = adata.uns.get("disk_backed")
        if src is not None:
            self._rank_genes_groups_file(adata, groupby, n_genes, src)
            return

        pp = _Preprocessing()
        pp._ensure_materialized(adata)

        from polariseq import _checks

        _checks.groups_column(adata, groupby, what="rank_genes_groups")
        cat = categorical(adata.obs[groupby])
        names = [str(c) for c in cat.categories]
        codes = np.asarray(cat.codes, dtype=np.int64)
        if groups is not None:
            _checks.values_in(groups, names, what="rank_genes_groups groups", column=f"obs[{groupby!r}]")
        wanted = set(names if groups is None else map(str, groups))
        var_names = list(adata.var_names) if adata.var_names else [f"gene_{i}" for i in range(adata.n_vars)]
        X = adata.X if (_is_sparse(adata.X) or _is_disk(adata.X)) else SparseMatrix.from_dense(
            np.ascontiguousarray(adata.X, dtype=np.float32))

        if reference != "rest":
            _checks.values_in([reference], names, what="rank_genes_groups reference",
                              column=f"obs[{groupby!r}]")
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
        from polariseq import _checks
        from polariseq._polariseq import rank_genes_groups_file

        _checks.groups_column(adata, groupby, what="rank_genes_groups")
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

    def compare_analyses(self, analyses: dict, key: str = "leiden", *, labels: str | None = None,
                         seeds: int = 0, n_markers: int = 20) -> Any:
        """Measure how far analyses of the same cells agree.

        For analyses made with different settings (``adata.from_counts``), compares their clusters on the
        cells they share, matched by name, with the adjusted Rand index (Hubert and Arabie 1985) and the
        normalized mutual information. With ``seeds``, each analysis's clustering is also rerun with other
        seeds, so a difference between settings can be told from the variation between runs. The
        analyses themselves are left as they were.

        Args:
            analyses: ``{name: analysis}``, two or more.
            key: The ``obs`` column of clusters in each.
            labels: An ``obs`` column of reference labels (for example published cell types) to score
                each analysis against.
            seeds: How many other seeds to rerun each analysis's Leiden clustering with.
            n_markers: How many top marker genes to compare between matched clusters, when both analyses
                ran ``rank_genes_groups`` on ``key``.

        Returns:
            The comparison: ``ari`` and ``nmi`` for every pair, ``n_clusters``, ``matches`` (which
            cluster became which) and ``crossing`` for each pair, and ``labels`` and ``noise`` when asked.
            Printing it shows a summary.

        Examples:
            ```python
            runs = {"hvg2000": a, "hvg3000": b}
            cmp = ps.tl.compare_analyses(runs, seeds=2)
            print(cmp)
            cmp.matches[("hvg2000", "hvg3000")]
            ```

        See also:
            ``ps.pl.compare_analyses`` draws the analyses' clusters side by side.
        """
        from polariseq._analyses import compare_analyses

        return compare_analyses(analyses, key, labels=labels, seeds=seeds, n_markers=n_markers)

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
        """Compare conditions within each cell type: which genes change, and by how much.

        With ``sample_key`` (recommended), the counts are added up per replicate (for example the donor),
        condition and cell type, and the sums are tested as bulk RNA-seq samples, the way edgeR and limma
        test them: the expression filter, TMM normalization and limma-trend. The replicates are the unit
        of evidence, so the p-values do not shrink as the number of cells grows. A donor measured under
        several conditions is compared with itself. The raw counts are read wherever they are: the matrix
        while it is still raw, the copy on disk, or the source files.

        Without ``sample_key``, each condition's cells are compared with the reference's by Welch's
        t-test. Cells from one donor are not independent, so these p-values overstate the evidence: use
        them to explore, and replicates to conclude.

        Writes the table, a summary per cell type and the settings to ``uns[key_added]``.

        The method follows edgeR and limma (Robinson, McCarthy and Smyth 2010; Ritchie et al. 2015;
        Robinson and Oshlack 2010; Law et al. 2014; Phipson et al. 2016).

        Args:
            adata: The analysis, with the conditions, cell types and replicates in ``obs``.
            condition_key: The ``obs`` column of the conditions.
            reference: The condition the others are compared with.
            groupby: The ``obs`` column of the cell types (default: all cells as one group).
            sample_key: The ``obs`` column of the replicates.
            conditions: The conditions to compare (default: all but the reference).
            groups: The cell types to compare (default: all).
            min_cells: With ``sample_key``, the fewest cells a replicate needs in a cell type to take
                part; without, the fewest cells each condition needs.
            key_added: Where in ``uns`` the results go.

        Returns:
            A ``pandas.DataFrame`` with one row per cell type, condition and gene, most significant
            first: ``log2_fold_change``, the mean expression on each side, the share of cells expressing
            the gene on each side, ``t``, ``p_value`` and ``adj_p_value`` (Benjamini-Hochberg, within each
            cell type and condition).

        Examples:
            ```python
            res = ps.tl.compare_conditions(adata, "condition", "control",
                                           groupby="cell_type", sample_key="donor")
            ps.pl.volcano(res, "CD14+ Monocytes")
            ```

        See also:
            ``ps.get.pseudobulk``, ``ps.pl.volcano``, ``ps.pl.fold_changes``.
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
        """Score each cell for a gene set.

        A cell's score is its mean expression of the genes in ``gene_list`` minus its mean expression of
        control genes: ``max(50, 5 × the set's size)`` genes drawn at random from all the other genes.
        Seurat's module score (Tirosh et al. 2016) works the same way but draws its control genes from
        genes of similar mean expression. Genes of the set that are not in the data are left out, with a
        warning. Runs in memory only.

        Writes the scores to ``obs[score_name]``.

        Args:
            adata: The analysis, log-normalized, in memory.
            gene_list: The genes of the set.
            score_name: The ``obs`` column of the scores (default: ``score_`` and the first gene's name).
            seed: Random seed of the control genes.

        Examples:
            ```python
            ps.tl.score_genes(adata, ["IFI6", "ISG15", "MX1"], score_name="interferon")
            ps.pl.umap(adata, color="interferon")
            ```

        See also:
            ``ps.tl.auto_annotate``.
        """
        _in_memory_only(adata, "score_genes")
        pp = _Preprocessing()
        pp._ensure_materialized(adata)

        if score_name is None:
            score_name = f"score_{gene_list[0][:10]}"

        from polariseq import _checks

        var_names = list(adata.var_names) if adata.var_names else []
        known = set(var_names)
        found = [g for g in gene_list if g in known]
        if not found:  # the error names each gene and why it is missing
            _checks.gene_positions(var_names, gene_list, what="score_genes",
                                   source_names=_source_gene_names(adata))
        missing = [g for g in gene_list if g not in known]
        if missing:
            import warnings

            warnings.warn(f"score_genes: {len(missing)} of the {len(gene_list)} genes are not in "
                          f"the data and are left out of the score: {_checks.listing(missing)}",
                          stacklevel=2)
        gene_idx = [var_names.index(g) for g in found]

        # Mean expression of the gene set.
        if _is_sparse(adata.X):
            subset = adata.X[:, gene_idx]
            set_mean = np.asarray(subset.mean(axis=1)).flatten()
        else:
            set_mean = adata.X[:, gene_idx].mean(axis=1)

        # Control set: random genes from all the genes outside the set (not matched by expression).
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
        _derived(adata, score_name)

    def auto_annotate(
        self,
        adata: AnnData,
        groupby: str = "leiden",
        *,
        source: str | list[str] | tuple[str, ...] | None = ("curated", "hpa"),
        tissue: str | list[str] | None = None,
        species: str | None = None,
        markers: Any = None,
        max_markers: int | None = 30,
        margin: float = 0.05,
        key_added: str = "cell_type",
        unassigned: str = "unassigned",
        method: str | None = None,
    ) -> Any:
        """Name each cluster's cell type from marker genes, with a score.

        Every cell is scored for every cell type in the chosen marker databases, and each cluster takes
        the type its cells score highest for on average. The names are a first guess: check them against
        each cluster's marker genes (``ps.get.markers``, a dot plot). Give ``tissue`` when you know it; it
        removes look-alike types of other organs, such as liver macrophages for blood monocytes.

        How the scores are made, following ScType (Ianevski, Giri and Aittokallio 2022) and the Cell
        Marker Accordion (Busarello et al. 2025):

        1. Markers: the chosen sources' tables (``ps.marker_table`` shows them), narrowed to ``tissue``,
           plus your own ``markers``. Ribosomal, mitochondrial and dissociation-stress genes are left out.
           For mouse data, human markers are mapped to mouse genes.
        2. Weights: a marker counts more when many records name it and when few cell types share it.
           Each type keeps its ``max_markers`` best markers.
        3. Scores: each marker is standardized over all cells, using every gene of the source files, so
           markers outside the selected genes still count. A cell's score for a type adds up its
           markers and subtracts the genes the type should not express.
        4. Clusters: a cluster stays ``unassigned`` when its best score is not above zero or fewer than a
           quarter of its cells support it. When the runner-up is within ``margin`` of the best, the
           cluster takes their nearest shared Cell Ontology type ("T cell" for two kinds of T cell).
        5. Cells: every cell also gets its own best type, so a small population hidden inside a larger
           cluster shows in the table's ``also`` column.

        On published labels (PBMC 3k, Kang et al. 2018, the Cao et al. 2020 fetal atlas organ by organ),
        the default named 73% of cells in the right lineage on average, 77% with ``tissue``, and 80%
        with the Accordion added.

        Marker sources (``source=``, one or a list; types are merged by Cell Ontology term):

        - ``"curated"``: a short list of canonical markers for about 40 types, with the blood subtypes.
        - ``"hpa"``: the Human Protein Atlas single cell types, 154 types in 31 tissues.
        - ``("curated", "hpa")``: both, the default, shipped with Polariseq.
        - ``"msigdb_c8"``: MSigDB C8 cell type signatures from 22 single-cell studies.
        - ``"accordion"``, ``"cellmarker"``, ``"panglaodb"``, ``"sctype"``: downloaded once on first use
          (their terms do not allow shipping them). ``"accordion"`` needs ``pyreadr``, ``"cellmarker"``
          and ``"sctype"`` need ``openpyxl`` (both: ``pip install 'polariseq[markers]'``).
        - ``"celltypist"``: CellTypist's trained models in place of markers (needs ``celltypist``).

        Writes the names to ``obs[key_added]``, with ``_ontology_id``, ``_score``, ``_cell`` (each cell's
        own best type) and ``_cell_score`` beside it, and the details to ``uns['annotation']``.

        Args:
            adata: The analysis, clustered.
            groupby: The ``obs`` column of the clusters.
            source: The marker source or sources above. ``None`` with ``markers``: only your markers.
            tissue: Keep this tissue's types, a name or a list: ``"immune"``, ``"brain"``, ``"lung"``,
                ``"liver"``, ``"kidney"``, ``"heart"``, ``"gut"``, ... (``ps.marker_table`` lists them).
                Immune types and types found in many tissues are kept with any tissue.
            species: ``"human"`` or ``"mouse"``; ``None`` guesses from the gene names.
            markers: Your own markers: ``{"type": ["GENE", ...]}``,
                ``{"type": {"positive": [...], "negative": [...]}}``, a table with ``cell_type`` and
                ``gene`` columns, or a ``.gmt``, ``.csv`` or ``.tsv`` file.
            max_markers: How many of its best markers each type keeps (``None``: all).
            margin: How close, as a fraction, the runner-up must score for the shared parent type to be
                used (0: never).
            key_added: The ``obs`` column of the names.
            unassigned: The name given to clusters and cells no type fits.
            method: The earlier name of ``source``.

        Returns:
            One row per cluster: ``cell_type``, ``ontology_id``, ``score``, ``runner_up`` and its score,
            ``support`` (the share of the cluster's cells scoring above zero for the type), ``note`` and
            ``also`` (groups of cells scoring highest for another type).

        Examples:
            ```python
            table = ps.tl.auto_annotate(adata, "leiden", tissue="immune")
            ps.pl.umap(adata, color="cell_type")
            ```

        See also:
            ``ps.marker_table``, ``ps.tl.annotate``, ``ps.get.markers``.
        """
        from polariseq import _annotate

        if method is not None:
            source = {"markers": "curated"}.get(method, method)
        if source == "celltypist":
            self._annotate_celltypist(adata, tissue if isinstance(tissue, str) else None)
            return None
        adata = _Preprocessing._ensure_materialized(adata)
        return _annotate.auto_annotate(adata, groupby, source=source, tissue=tissue,
                                       species=species, markers=markers, max_markers=max_markers,
                                       margin=margin, key_added=key_added, unassigned=unassigned)

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
            _derived(adata, "cell_type")

        adata.uns["annotation"] = {
            "method": "celltypist",
            "model": model_name,
            "tissue": tissue,
        }

    def annotate(self, adata: AnnData, mapping: dict, groupby: str = "leiden", *,
                 key_added: str = "cell_type", unassigned: str | None = None) -> None:
        """Name clusters by hand.

        ``mapping`` maps clusters to names, ``{"0": "CD4 T cell", "4": "B cell"}``, or names to lists of
        clusters, ``{"B cell": ["4", "9"]}``. Clusters given the same name become one group, the simplest
        way to join clusters that are one cell type. A cluster left out of ``mapping`` keeps its own label,
        or takes ``unassigned`` when that is given. Writes the names to ``obs[key_added]``.

        Args:
            adata: The analysis, clustered.
            mapping: ``{cluster: name}`` or ``{name: [clusters]}``.
            groupby: The ``obs`` column the clusters come from.
            key_added: The ``obs`` column the names are written to.
            unassigned: The name of clusters left out of ``mapping`` (default: their own label).

        Examples:
            ```python
            ps.tl.annotate(adata, {"0": "CD4 T cell", "1": "CD14+ monocyte", "2": "B cell"})
            ps.pl.umap(adata, color="cell_type")
            ```

        See also:
            ``ps.tl.auto_annotate``, ``ps.tl.merge_clusters``, ``ps.tl.subcluster``.
        """
        from polariseq._curation import annotate
        annotate(adata, mapping, groupby, key_added=key_added, unassigned=unassigned)

    def merge_clusters(self, adata: AnnData, groups: list, groupby: str = "leiden", *,
                       key_added: str | None = None, names: list[str] | None = None) -> None:
        """Join clusters into one.

        Each inner list of ``groups`` becomes one cluster, named after its members joined by ``+``
        (``"3+7"``) or by the matching entry of ``names``; a single list is one group. Every other cluster
        keeps its label. Writes the result to ``obs[key_added]``, by default ``obs['<groupby>_curated']``,
        so the original clusters stay as they were.

        Args:
            adata: The analysis, clustered.
            groups: The clusters to join: a list of lists, or one list for a single group.
            groupby: The ``obs`` column the clusters come from.
            key_added: The ``obs`` column written (``groupby`` replaces the original).
            names: A name for each joined cluster.

        Examples:
            ```python
            ps.tl.merge_clusters(adata, [["3", "7"], ["10", "12"]])
            ps.pl.umap(adata, color="leiden_curated")
            ```

        See also:
            ``ps.tl.subcluster``, ``ps.tl.annotate``.
        """
        from polariseq._curation import merge_clusters
        merge_clusters(adata, groups, groupby, key_added=key_added, names=names)

    def subcluster(self, adata: AnnData, cluster: Any, groupby: str = "leiden", *,
                   resolution: float = 1.0, seed: int = 0, key_added: str | None = None) -> int:
        """Split one cluster into smaller ones.

        Runs Leiden on the neighbor graph between the cluster's cells only. Cluster ``"5"`` becomes
        ``"5.0"``, ``"5.1"`` and so on, largest first; every other cell keeps its cluster. Writes the result
        to ``obs[key_added]``, by default ``obs['<groupby>_curated']``, so the original clusters stay as
        they were. Only the graph is read, so it works the same out of core.

        Args:
            adata: The analysis, after ``pp.neighbors`` and clustering.
            cluster: The cluster to split.
            groupby: The ``obs`` column it belongs to.
            resolution: Leiden resolution within the cluster (higher: more parts).
            seed: Random seed.
            key_added: The ``obs`` column written (``groupby`` replaces the original).

        Returns:
            The number of parts.

        Examples:
            ```python
            ps.tl.subcluster(adata, "5", resolution=0.5)
            ps.pl.umap(adata, color="leiden_curated")
            ```

        See also:
            ``ps.tl.merge_clusters``, ``ps.tl.leiden``.
        """
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
        """Build the tree of clusters with Paris, and cut it.

        Paris (Bonald et al. 2018) builds the whole hierarchy of the neighbor graph in one pass: a tree in
        which every cut is a clustering, from one cluster down to single cells. Coarse types and their
        subtypes are then cuts of the same tree, without clustering again. The tree matches
        scikit-network's ``Paris`` exactly.

        With ``n_clusters``, the tree is cut straight into that many clusters (or a few more where heights
        tie), numbered from 0 by decreasing size. With ``balanced=True``, it is cut as Scarf does
        (Dhapola et al. 2022), into clusters of at most ``max_size`` cells.

        Writes the tree to ``uns[key_added]['dendrogram']`` (in the layout of SciPy's ``linkage``), reused
        while the graph is unchanged, and the labels of a cut to ``obs[key_added]``.

        Args:
            adata: The analysis, after ``pp.neighbors``.
            n_clusters: Cut straight into this many clusters.
            balanced: Use Scarf's balanced cut.
            min_size: Balanced cut: the size above which two halves' heights are compared.
            max_size: Balanced cut: the largest cluster.
            max_distance_fc: Balanced cut: the largest ratio of heights at which two halves are joined.
            key_added: The ``obs`` and ``uns`` key.
            recompute: Rebuild the tree even when one is stored.

        Examples:
            ```python
            ps.tl.paris(adata, n_clusters=10)
            ps.tl.paris(adata, n_clusters=30, key_added="paris_fine")   # reuses the tree
            ```

        See also:
            ``ps.tl.leiden``.
        """
        from polariseq import _checks

        _checks.result(adata, "connectivities", what="paris", where="obsp",
                       made_by="ps.pp.neighbors")
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
            _derived(adata, key_added)
            stored["cut"] = {"balanced": True, "min_size": int(min_size), "max_size": int(max_size),
                             "max_distance_fc": float(max_distance_fc)}
        elif n_clusters is not None:
            labels = dendrogram_cut_straight(d, int(n_clusters))
            adata.obs[key_added] = labels.astype(np.int32)
            _derived(adata, key_added)
            stored["cut"] = {"n_clusters": int(n_clusters)}
        if key_added in adata.obs:
            stored["n_clusters"] = int(np.asarray(adata.obs[key_added]).max()) + 1

    def dpt(
        self,
        adata: AnnData,
        root: int = 0,
    ) -> None:
        """Pseudotime from a root cell: each cell's shortest-path distance from it.

        The paths run over the neighbor graph of ``pp.neighbors``, with its
        distances as edge lengths and every edge usable in both directions. The
        distances are divided by the largest, so the root is at 0 and the
        farthest cell it reaches at 1; cells it cannot reach get ``inf``. This
        is a graph distance; diffusion pseudotime (Haghverdi et al. 2016) is a
        different measure.

        Writes ``obs['dpt_pseudotime']``, and ``uns['dpt']`` with the root and
        the number of cells reached.

        Args:
            adata: The analysis, after ``pp.neighbors``.
            root: Position of the root cell.

        Examples:
            ```python
            ps.tl.dpt(adata, root=0)
            ps.pl.umap(adata, color="dpt_pseudotime")
            ```

        See also:
            ``ps.tl.paga``.
        """
        from polariseq import _checks

        _checks.result(adata, "distances", what="dpt", where="obsp", made_by="ps.pp.neighbors")
        if not 0 <= int(root) < adata.n_obs:
            raise ValueError(f"dpt: root={root} is not a cell; give a position from 0 to "
                             f"{adata.n_obs - 1:,}")

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
        _derived(adata, "dpt_pseudotime")
        adata.uns["dpt"] = {"root": root, "n_reachable": result["n_reachable"]}

    def paga(
        self,
        adata: AnnData,
        groupby: str = "leiden",
    ) -> None:
        """A graph between clusters: how strongly the neighbor graph links each pair.

        For two clusters, the connectivity is the number of neighbor-graph edges
        between them, counted in both directions, divided by the number of cells
        in the two; the diagonal is 1. PAGA (Wolf et al. 2019) measures it
        differently, comparing the edges between two clusters with the number
        expected by chance.

        Writes ``uns['paga']``: ``connectivities`` (clusters by clusters),
        ``cluster_sizes`` and ``groupby``.

        Args:
            adata: The analysis, after ``pp.neighbors``.
            groupby: The ``obs`` column of numbered clusters, as ``tl.leiden``
                writes it.

        Examples:
            ```python
            ps.tl.paga(adata, groupby="leiden")
            adata.uns["paga"]["connectivities"]
            ```

        See also:
            ``ps.tl.dpt``, ``ps.tl.leiden``.
        """
        from polariseq import _checks

        _checks.result(adata, "distances", what="paga", where="obsp", made_by="ps.pp.neighbors")
        _checks.groups_column(adata, groupby, what="paga")

        from polariseq._polariseq import paga_connectivities as _paga
        dist = adata.obsp["distances"]
        try:
            labels = np.asarray(adata.obs[groupby]).astype(np.uint32)
        except (TypeError, ValueError):
            raise ValueError(f"paga: obs[{groupby!r}] holds names; paga groups cells by numbered "
                             "clusters (as leiden makes): pass that column") from None
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
        """Lay the cells out in two dimensions with UMAP.

        Places the cells so that neighbors in the graph of ``pp.neighbors`` stay close (McInnes, Healy and
        Melville 2018). The layout runs on all threads and differs slightly from run to run, as
        umap-learn's does; ``deterministic=True`` runs it on one thread for the same layout every time.

        By default the layout starts from Polariseq's multilevel spectral start: similar cells are grouped,
        the standard spectral start is computed for the much smaller graph of groups, and each cell starts
        at its group's position, smoothed over the full graph. This is fast for millions of cells and keeps
        the large-scale structure.

        Writes ``obsm['X_umap']`` and ``uns['umap']``.

        Args:
            adata: The analysis, after ``pp.neighbors``.
            min_dist: How tightly points may pack (default 0.5 in the ``'polariseq'`` workflow, 0.1 in
                ``'scanpy'``).
            spread: The scale of the layout.
            n_epochs: Optimization rounds (0: 500 up to 10,000 cells, otherwise 100 in the
                ``'polariseq'`` workflow and 200 in ``'scanpy'``).
            seed: Random seed.
            deterministic: Run on one thread, giving the same layout every time.
            init: Where the cells start: ``'multilevel'`` (the default in the ``'polariseq'``
                workflow), ``'spectral'`` (the exact spectral start, the default in ``'scanpy'``),
                ``'leiden'`` (the multilevel start through the Leiden clusters), ``'kmeans'`` (Scarf's
                start), ``'pca'`` (the first two principal components), ``'random'``, or an array with
                two columns.
            init_smooth: Smoothing steps of the multilevel starts.

        Examples:
            ```python
            ps.tl.umap(adata)
            ps.pl.umap(adata, color="leiden")
            ```

        See also:
            ``ps.pl.umap`` draws the layout. ``ps.pp.neighbors`` builds the graph it is computed from.
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
            from polariseq import _checks

            _checks.graph(adata, what="umap")
        graph = adata.obsp["connectivities" if use_conn else "distances"]
        if use_conn and hasattr(graph, "has_sorted_indices") and not graph.has_sorted_indices:
            graph.sort_indices()  # the core expects rows sorted by column
        # The graph is read into memory when it fits beside the layout's
        # working memory, else in place (`_results.graph_inputs`); the layout itself is built in memory (its
        # SGD writes cells in random order) and written to its file at the end.
        from polariseq.planning import umap_working_bytes

        (g_ip, g_ix, g_d), inputs, cache = _results.graph_inputs(
            [graph.indptr, graph.indices, graph.data], _held_bytes(adata),
            umap_working_bytes(int(graph.nnz), adata.n_obs))
        out = _results.new(adata, "umap", "X_umap", (adata.n_obs, 2), np.float32)
        layout = np.zeros(out.shape, np.float32) if isinstance(out, np.memmap) else out
        with cache:
            self._umap_layout(adata, layout, g_ip, g_ix, g_d, use_conn, init, init_smooth,
                              min_dist, spread, n_epochs, seed, deterministic)
        if layout is not out:
            out[...] = layout
        adata.obsm["X_umap"] = _results.view(out)
        adata.uns["umap"] = {"min_dist": min_dist, "spread": spread, "inputs": inputs}

    def _umap_layout(self, adata, out, g_ip, g_ix, g_d, use_conn, init, init_smooth,
                     min_dist, spread, n_epochs, seed, deterministic) -> None:
        """The start and the SGD of :meth:`umap`, into ``out``."""
        mode = init if isinstance(init, str) else "given"
        if isinstance(init, str) and init in ("pca", "kmeans"):
            if "X_pca" not in adata.obsm:
                raise KeyError(f"init={init!r} needs obsm['X_pca']; run ps.pp.pca() first")
            if init == "pca":
                start = np.asarray(adata.obsm["X_pca"][:, :2], dtype=np.float64)
            else:
                pcs = np.ascontiguousarray(adata.obsm["X_pca"], dtype=np.float32)
                start = np.asarray(umap_kmeans_start(pcs, 1000, int(seed)), dtype=np.float64)
                del pcs
            mode = "given"
        elif isinstance(init, str) and init in ("multilevel", "leiden"):
            if use_conn:
                c_ip, c_ix, c_d = _as_u32(g_ip), _as_u32(g_ix), np.asarray(g_d, np.float32)
            else:
                fz = fuzzy_connectivities(_as_u32(g_ip), _as_u32(g_ix),
                                          np.asarray(g_d, dtype=np.float32), adata.n_obs)
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
                del pcs
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
            _as_u32(g_ip),
            _as_u32(g_ix),
            np.asarray(g_d, dtype=np.float32),
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

    def leiden(
        self,
        adata: AnnData,
        resolution: float = 1.0,
        *,
        seed: int = 0,
        key_added: str = "leiden",
        method: str = "auto",
    ) -> None:
        """Cluster the cells with the Leiden algorithm.

        Groups cells that are densely connected in the neighbor graph of ``pp.neighbors``
        (``obsp['connectivities']``). Raise ``resolution`` for more, smaller clusters; lower it for fewer,
        larger ones. The same graph and seed always give the same clusters, at any number of threads.

        Writes the cluster numbers to ``obs[key_added]`` and the run's details to ``uns['leiden']``: the
        number of clusters (``n_communities``), the modularity of the clustering (``quality``), the
        resolution and the method.

        The algorithm is Leiden (Traag et al. 2019), as the authors' CWTS implementation runs it.
        ``method='fast'`` runs the earlier Louvain method (Blondel et al. 2008) on the same graph, and
        ``method='igraph'`` runs igraph's Leiden, which needs the ``python-igraph`` package.

        Args:
            adata: The analysis, after ``pp.neighbors``.
            resolution: Higher gives more, smaller clusters; lower gives fewer, larger ones.
            seed: Random seed; the same seed gives the same clusters.
            key_added: The ``obs`` column the cluster numbers are written to.
            method: ``'auto'`` (Leiden), ``'fast'`` (Louvain) or ``'igraph'`` (igraph's Leiden).

        Examples:
            ```python
            ps.pp.neighbors(adata, n_neighbors=15)
            ps.tl.leiden(adata, resolution=1.0)
            ps.pl.umap(adata, color="leiden")
            ```

        See also:
            ``ps.tl.subcluster``, ``ps.tl.paris``, ``ps.tl.compare_analyses``.
        """
        # The graph every method runs on: the symmetrised fuzzy
        # connectivities (scanpy semantics; the graph UMAP shares), used as
        # given. Same weights for ours, fast, and igraph, so quality numbers
        # are comparable across implementations. The arrays are handed over
        # as views, not copies: at 4M cells the graph is ~1 GB, and each
        # `astype` here used to duplicate it.
        from polariseq import _checks

        _checks.graph(adata, what="leiden")
        if "connectivities" in adata.obsp:
            conn = adata.obsp["connectivities"]
            if hasattr(conn, "has_sorted_indices") and not conn.has_sorted_indices:
                conn.sort_indices()  # in place; the core expects sorted rows
            # Read into memory when it fits beside Leiden's working memory,
            # else in place (`_results.graph_inputs`).
            from polariseq.planning import leiden_working_bytes

            (c_indptr, c_indices, c_data), inputs, cache = _results.graph_inputs(
                [conn.indptr, conn.indices, conn.data], _held_bytes(adata),
                leiden_working_bytes(int(conn.nnz)))
            c_indptr, c_indices = _as_u32(c_indptr), _as_u32(c_indices)
            c_data = np.asarray(c_data, dtype=np.float32)
        else:
            dist = adata.obsp["distances"]
            conn = fuzzy_connectivities(_as_u32(dist.indptr), _as_u32(dist.indices),
                                        np.asarray(dist.data, dtype=np.float32), adata.n_obs)
            c_indptr = conn["indptr"]
            c_indices = conn["indices"]
            c_data = conn["data"]
            inputs, cache = "memory", contextlib.nullcontext()

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
            _derived(adata, key_added)
            adata.uns["leiden"] = {
                "n_communities": int(len(part)),
                "quality": float(g.modularity(part.membership, weights="weight")),
                "resolution": resolution,
                "method": "igraph_leiden",
                "params": {"seed": seed, "key_added": key_added},
            }
            return

        # 'auto' is always Leiden (its modularity matches or exceeds
        # igraph's on every dataset measured; reproduce branch, benchmarks/relicence); 'fast'
        # (multi-level Louvain, same engine) stays available as an opt-in.
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
        with cache:
            if use_fast:
                result = fast_community(c_indptr, c_indices, c_data, adata.n_obs, resolution, seed)
                algo = "fast_label_propagation"
            else:
                result = leiden_cluster(c_indptr, c_indices, c_data, adata.n_obs, resolution, seed)
                algo = "leiden"

        adata.obs[key_added] = result["labels"].astype(np.int32)
        _derived(adata, key_added)
        adata.uns["leiden"] = {
            "n_communities": int(result["n_communities"]),
            "quality": float(result["quality"]),
            "resolution": resolution,
            "method": algo,
            "n_edges": int(result["n_edges"]),
            "total_weight": float(result["total_weight"]),
            "params": {"seed": seed, "key_added": key_added},
            "inputs": inputs,
        }


class _Get:
    """The ``ps.get`` namespace: data taken out of an analysis."""

    def markers(self, adata: AnnData, n: int = 10, *, groups: list | None = None,
                wide: bool = False) -> Any:
        """The top marker genes of each cluster, as a table.

        Reads what ``tl.rank_genes_groups`` found. The long table has one row per cluster and gene;
        ``wide=True`` gives one column of gene names per cluster, the table to read when naming clusters.

        Args:
            adata: The analysis, after ``tl.rank_genes_groups``.
            n: Genes per cluster.
            groups: The clusters to include (default: all that were ranked).
            wide: One column of gene names per cluster.

        Returns:
            A ``pandas.DataFrame``: ``group``, ``rank``, ``gene``, ``score``, ``logfoldchange``,
            ``pval_adj`` and ``pct`` (the share of the cluster's cells expressing the gene); with
            ``wide=True``, one column of gene names per cluster.

        Examples:
            ```python
            ps.tl.rank_genes_groups(adata, groupby="leiden")
            ps.get.markers(adata, n=5, wide=True)
            ```

        See also:
            ``ps.tl.rank_genes_groups``, ``ps.pl.dotplot``.
        """
        from polariseq._curation import markers
        return markers(adata, n, groups=groups, wide=wide)

    def pseudobulk(self, adata: AnnData | LazyAnnData, sample_key: str, *,
                   groupby: str | None = None, condition_key: str | None = None,
                   min_cells: int = 0) -> AnnData:
        """Raw counts added up per replicate, and per cell type and condition.

        The table edgeR, DESeq2 or limma test: one row per combination of ``groupby``, ``condition_key``
        and ``sample_key`` with at least ``min_cells`` cells, the gene counts summed, and the number of
        cells in ``obs['n_cells']``. The raw counts are read wherever they are held, as in
        ``tl.compare_conditions``. Pass the result's ``to_anndata()`` to pydeseq2, or write its ``X`` for
        R.

        Args:
            adata: The analysis.
            sample_key: The ``obs`` column of the replicates (for example the donor).
            groupby: The ``obs`` column of the cell types (default: all cells as one group).
            condition_key: The ``obs`` column of the conditions, to keep them apart.
            min_cells: Combinations with fewer cells are left out.

        Returns:
            An analysis with one row per combination (``obs`` names its cell type, condition and
            replicate, and holds ``n_cells``), the summed counts in ``X`` and the genes in ``var_names``.

        Examples:
            ```python
            pb = ps.get.pseudobulk(adata, "donor", groupby="cell_type", condition_key="condition",
                                   min_cells=10)
            ```

        See also:
            ``ps.tl.compare_conditions``.
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
    """Deprecated: use ``ps.read_h5ad([...])``, ``ps.pp.pca`` and ``ps.pp.harmony_integrate(adata, 'dataset')``.

    Kept so that existing scripts run; it warns when called. It streams several datasets without
    loading their matrices:
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
        # h5py is not a dependency: the core's own reader gives the same names (gene names alone
        # when no cell names are asked for).
        try:
            meta = scan_h5ad(str(path)) if "obs" in keys else scan_h5ad_genes(str(path))
            for key in keys:
                names = meta.get(f"{key}_names")
                if names:
                    out[key] = np.asarray(names, dtype=str)
        except (OSError, RuntimeError, ValueError):
            pass
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
    include: list[str] | None = None,
) -> AnnData:
    """Run the whole analysis, from quality control to Leiden clusters, in one call, out of core.

    Copies the counts from the file into the project folder without holding the matrix in memory, then
    runs quality control, the cell and gene filters, normalization, gene selection, PCA, the neighbor
    graph and Leiden clustering. It applies the same steps, in the same order and with the same
    defaults, as the step-by-step functions, so both analyze the same cells and genes. The workflow
    (``set_workflow``) decides the gene selection, normalization and scaling.

    The filters keep every cell and gene unless you set them.

    Args:
        path: The ``.h5ad`` file.
        n_hvg: The number of genes to select.
        n_pcs: The number of principal components.
        n_neighbors: Neighbors per cell.
        target_sum: The total each cell is scaled to.
        resolution: Leiden resolution.
        seed: Random seed (the step functions use 0 by default).
        min_genes: The fewest detected genes a cell may have, as in ``ps.pp.filter_cells``.
        max_genes: The most detected genes a cell may have.
        min_counts: The fewest total counts a cell may have.
        max_counts: The most total counts a cell may have.
        max_pct_mt: The largest share of mitochondrial counts, in percent.
        mito_prefix: The start of mitochondrial gene names, as in ``ps.pp.calculate_qc_metrics``.
        min_cells: The fewest kept cells a gene must be detected in, as in ``ps.pp.filter_genes``.
            Removed genes take no part in normalization, gene selection or PCA (``var['kept']``).
        include: Genes to keep whatever the ranking says, added to the top ``n_hvg``. A name not in
            the file, or a gene ``min_cells`` removes, is an error naming it.

    Returns:
        The analysis, with the quality metrics in ``obs`` and ``var``, ``obsm['X_pca']``, the neighbor
        graph and the Leiden clusters.

    Examples:
        ```python
        ps.project("atlas_analysis")
        adata = ps.process_diskbacked("atlas.h5ad", min_genes=200, min_cells=3)
        ps.tl.umap(adata)
        ```

    See also:
        ``ps.read_h5ad``, ``ps.plan``.
    """
    import os
    import time

    from polariseq import _checks

    path = _checks.input_file(path, what="process_diskbacked")
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
    forced: list[tuple[int, str]] = []
    if include is not None:
        from polariseq import _checks

        if var_names is None:
            raise ValueError(f"include: {path} has no gene names to match the genes by")
        pos = _checks.gene_positions(var_names, include, what="include")
        forced = [(int(i), str(var_names[i])) for i in pos]

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
        include=forced,
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
    if forced:
        included = np.zeros(int(r["n_vars"]), dtype=bool)
        included[[i for i, _ in forced]] = True
        adata.var["included"] = included
        adata.uns["hvg"]["n_top_genes"] = int(n_hvg)
        adata.uns["hvg"].update(include=[n for _, n in forced],
                                n_selected=int(adata.var["highly_variable"].sum()))
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
        from polariseq import _checks

        raise _checks.InputError(f"no {what} named {e.args[0]!r}.{_checks.suggest(e.args[0], lookup)}") from None


def _source_gene_names(adata: Any) -> list[str] | None:
    """Every gene of the files the data were read from (``None``: unknown), so
    a missing gene can be told apart from one an earlier step removed."""
    src = getattr(adata, "_source", None)
    if not src:
        return None
    try:
        names: list[str] = []
        for p in src["paths"]:
            names.extend(str(g) for g in scan_h5ad_genes(os.fspath(p))["var_names"])
        return names
    except Exception:  # noqa: BLE001  (moved or unreadable: no distinction made)
        return None


def _add_included(adata: Any, X: Any, hvg_mask: np.ndarray, forced: np.ndarray,
                  floor: int) -> np.ndarray:
    """The ranking's genes with the genes asked for added; records which, and
    warns about any detected in too few cells to carry much signal."""
    import warnings

    names = [str(adata.var_names[i]) for i in forced]
    added = [n for n, i in zip(names, forced, strict=True) if not hvg_mask[i]]
    mask = hvg_mask.copy()
    mask[forced] = True
    included = np.zeros(mask.size, dtype=bool)
    included[forced] = True
    adata.var["highly_variable"] = mask
    adata.var["included"] = included
    adata.uns["hvg"].update(include=names, include_added=added, n_selected=int(mask.sum()))
    # How many cells express each gene asked for: from the gene filter when it ran, else counted.
    if "n_cells" in adata.var:
        cells = np.asarray(adata.var["n_cells"])[forced]
    elif _is_sparse(X):
        cells = np.bincount(np.asarray(X[:, forced].indices, dtype=np.int64), minlength=forced.size)
    else:
        cells = None
    if cells is not None:
        low_at = floor if floor > 0 else int(0.01 * adata.n_obs)
        low = [f"{n} ({int(c):,} cells, {c / max(adata.n_obs, 1):.2%})"
               for n, c in zip(names, cells, strict=True) if c <= low_at]
        if low:
            warnings.warn(f"include: detected in {low_at:,} cells or fewer, so they carry little "
                          f"signal (kept as asked): {', '.join(low)}", stacklevel=3)
    return mask


def _all_gene_names(adata: Any) -> list[str]:
    """Every gene of the source files, in the order the data were read in."""
    src = adata._source
    paths = [os.fspath(p) for p in src["paths"]]
    if src.get("maps") is None:
        return [str(g) for g in scan_h5ad_genes(paths[0])["var_names"]]
    names = [list(scan_h5ad_genes(p)["var_names"]) for p in paths]
    modes = [getattr(adata, "genes", None)] if getattr(adata, "genes", None) else ["intersection", "union"]
    for mode in modes:
        merged, _ = _merged_genes(names, mode, paths)
        if len(merged) == int(src["n_cols"]):
            return merged
    raise ValueError("the source files' genes no longer give the matrix's genes: did a file change?")


def _from_counts(adata: Any, name: str, genes: str) -> Any:
    """See :meth:`AnnData.from_counts`."""
    import copy as _copy
    import re

    from polariseq import _checks

    if not re.fullmatch(r"[\w.+-]+", str(name)) or str(name).startswith("."):
        raise ValueError(f"from_counts: an analysis is named with letters, digits and . _ + - only, "
                         f"not {name!r}")
    if genes not in ("all", "current"):
        raise ValueError(f"from_counts: genes must be 'all' or 'current', not {genes!r}")
    if name == getattr(adata, "analysis", None):
        raise ValueError(f"from_counts: {name!r} is this analysis's own name; give the new one "
                         "another, so their results stay apart")
    X = adata._X if isinstance(adata, LazyAnnData) else adata.X
    src = getattr(adata, "_source", None)
    rows = getattr(adata, "_rows", None)
    cols = getattr(adata, "_cols", None) if genes == "current" else None
    if _is_disk(X):
        n_file, g_file = X.file_shape
        col_keep = (np.asarray(X.col_mask(), dtype=bool) if genes == "current"
                    else np.ones(g_file, dtype=bool))
        new_x = X.subset(None, None)
        new_x.restore_view(np.ascontiguousarray(X.row_mask(), dtype=bool),
                           np.ascontiguousarray(col_keep), None, False)
    elif src is not None and all(str(p).endswith(".h5ad") for p in src["paths"]):
        from polariseq._state import _read_source, _replay

        paths = [os.fspath(p) for p in src["paths"]]
        for p in paths:
            _checks.input_file(p, what="from_counts")
        new_x = _read_source(src, paths)
        if new_x is None:
            raise ValueError(f"from_counts: {paths[0]} holds a dense matrix; read it with "
                             "ps.read_h5ad to analyze it again")
        new_x = _replay(new_x, rows, cols, [])
    elif getattr(adata, "_xlog", None) == [] and (genes == "current" or cols is None):
        new_x = X.copy() if hasattr(X, "copy") else np.array(X)  # still the counts as read
    else:
        raise ValueError("from_counts needs the raw counts: this object's matrix was changed and "
                         "it has no source .h5ad to read them from again (made from an array, or "
                         "read from 10x files). Read the data with ps.read_h5ad, or call "
                         "from_counts before normalizing")
    if genes == "all":
        var_names = _all_gene_names(adata) if src is not None else list(map(str, adata.var_names))
    else:
        var_names = list(map(str, adata.var_names))
    # the files' own gene columns; what the analysis computed per gene stays behind
    file_var = {}
    if src is not None and src.get("maps") is None:
        with contextlib.suppress(Exception):
            file_var = _read_frame(os.fspath(src["paths"][0]), "var")
    if genes == "all":
        var = dict(file_var)
    else:
        var = {k: _take(v, slice(None)) for k, v in adata.var.items() if k in file_var}
    derived = getattr(adata, "_derived", None) or set()

    def own(v: Any) -> Any:
        return v.copy() if hasattr(v, "copy") else _take(v, slice(None))

    new = _copy.copy(adata)
    if isinstance(new, LazyAnnData):
        new._temp_files, new._mmap_handles, new._closed = [], [], False
        new._X, new._materialized = new_x, True
        new._n_vars = len(var_names)
    else:
        new._X = new_x
    new.obs = {k: own(v) for k, v in adata.obs.items() if k not in derived}
    new.var = var
    new.obs_names, new.var_names = list(adata.obs_names), var_names
    new.obsm, new.obsp = {}, {}
    new.uns = {k: v for k, v in adata.uns.items() if k == "scrublet"}  # scored from the counts
    new._rows = None if rows is None else np.array(rows)
    new._cols = None if cols is None else np.array(cols)
    new._xlog = [] if not _is_disk(new_x) else None
    new._derived = set()
    new.analysis = str(name)
    return new


def _derived(adata: Any, key: str) -> None:
    """Record that ``obs[key]`` was derived by the analysis (see ``AnnData._derived``)."""
    if getattr(adata, "_derived", None) is None:
        adata._derived = set()
    adata._derived.add(key)


def _set_x(adata: Any, value: Any) -> None:
    """Give ``adata`` a subset of its own matrix: the values are unchanged, so
    the record of the steps that made them stays (see ``AnnData._xlog``)."""
    log = getattr(adata, "_xlog", None)
    adata.X = value
    adata._xlog = log


def _log_x(adata: Any, op: dict[str, Any]) -> None:
    """Record a step that rewrote the values of an in-memory sparse matrix,
    with the source genes it saw (``None``: all)."""
    log = getattr(adata, "_xlog", None)
    if log is not None:
        cols = getattr(adata, "_cols", None)
        log.append({**op, "cols": None if cols is None else np.asarray(cols, dtype=np.int64)})


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
    log = getattr(adata, "_xlog", None)
    new._xlog = None if log is None or _is_disk(X) else list(log)
    new._derived = set(getattr(adata, "_derived", None) or ())
    for attr in ("name", "batch_key", "labels", "analysis"):
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
    from polariseq import _checks

    sources = _sources(paths)
    scans = [scan_h5ad_genes(_checks.input_file(p, what="plan")) for _, p in sources]
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


def _held_bytes(adata) -> int:
    """What a run holds in memory besides a graph step's inputs, as the plan
    counts it: the matrix when it is in memory, and the process itself."""
    from polariseq.planning import PROCESS_OVERHEAD_BYTES, _csr_bytes

    x = getattr(adata, "_X", None)
    if isinstance(x, SparseMatrix):
        return _csr_bytes(int(x.nnz), int(x.shape[0])) + PROCESS_OVERHEAD_BYTES
    if isinstance(x, np.ndarray) and _results.mapping(x) is None:
        return int(x.nbytes) + PROCESS_OVERHEAD_BYTES
    return PROCESS_OVERHEAD_BYTES


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
        # No SciPy (it is the optional `scipy` extra): the graph's CSR arrays, as the core wrote them.
        return _CSRGraph(np.asarray(indptr), np.asarray(indices), np.asarray(data), (n_obs, n_obs))


class _CSRGraph:
    """A neighbour graph as CSR arrays, stored in ``obsp`` when SciPy is not installed.

    Carries what Polariseq's own steps read (``indptr``, ``indices``, ``data``, ``shape``, ``nnz``). The
    core writes every row sorted by column with no duplicates, which the two flags record.
    ``to_scipy()`` converts it once SciPy is installed.
    """

    has_sorted_indices = True
    has_canonical_format = True

    def __init__(self, indptr, indices, data, shape):
        self.indptr, self.indices, self.data, self.shape = indptr, indices, data, shape

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1]) if self.indptr.size else 0

    def to_scipy(self) -> Any:
        sp = _scipy_sparse("to_scipy()")
        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)

    def __repr__(self) -> str:
        return f"<{self.shape[0]} x {self.shape[1]} neighbour graph, {self.nnz} entries (CSR arrays)>"
