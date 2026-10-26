"""Per-cell results of steps the Rust core computes, as memory-mapped files.

With a project folder, a step's result arrays are files in
``runs/<name>/results/<step>/`` that the Rust core fills in place: nothing is
copied into Python, and Python holds views that the operating system pages in
when they are read, so a result costs memory only while it is used. The views
are copy-on-write: code that changes an array in place (SciPy sorting a
graph's indices, say) changes its own copy of the touched pages, never the
file. Without a project folder (or for an object with no dataset name) the
arrays live in RAM.

Every run writes new files: a view of an earlier run's file stays valid (on
Windows a mapped file cannot be replaced), and files no view holds any more
are removed at the next run.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import shutil
import tempfile
import uuid
import warnings
from typing import Any

import numpy as np


def new(adata: Any, step: str, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """A writable array for ``step``'s result ``key``: memory-mapped in the
    dataset's results folder when a project is set, else in RAM."""
    from polariseq._project import current

    proj, name = current(), getattr(adata, "name", None)
    if proj is None or not name:
        return np.zeros(shape, dtype=dtype)
    folder = proj.results_dir(name, getattr(adata, "analysis", None)) / step
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{key}.*.npy"):
        with contextlib.suppress(OSError):  # still mapped (Windows); removed at a later run
            old.unlink()
    path = folder / f"{key}.{uuid.uuid4().hex[:8]}.npy"
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def view(arr: np.ndarray, length: int | None = None) -> np.ndarray:
    """What an annotation holds: a filled result as a copy-on-write view of its
    file (pages load when read), or the array itself when in RAM. ``length``
    keeps only the first entries (a graph that filled less than it reserved)."""
    if isinstance(arr, np.memmap):
        arr.flush()
        arr = np.load(arr.filename, mmap_mode="c")
    return arr if length is None else arr[:length]


def keep(adata: Any, step: str, key: str, arr: np.ndarray) -> np.ndarray:
    """``arr`` (already computed) kept as a result: saved to the dataset's
    results folder and returned as a read-only view, or returned as it is
    without a project folder."""
    out = new(adata, step, key, arr.shape, arr.dtype)
    if not isinstance(out, np.memmap):
        return arr
    out[...] = arr
    return view(out)


def mapping(arr: Any) -> mmap.mmap | None:
    """The file mapping behind ``arr`` (a result view, a slice of one, an array
    built on one), or None for an array in memory."""
    a = arr
    while a is not None:
        if isinstance(a, mmap.mmap):
            return a
        if isinstance(a, np.memmap) and getattr(a, "_mmap", None) is not None:
            return a._mmap
        a = getattr(a, "base", None)
    return None


def in_ram(arr: Any) -> Any:
    """``arr`` read into memory in one sequential pass when it is a view of a
    result file, else ``arr`` itself."""
    return np.array(arr) if mapping(arr) is not None else arr


def _source_file(arr: Any) -> np.memmap | None:
    """The memory-mapped array of a whole result file behind ``arr``."""
    a, found = arr, None
    while a is not None:
        if isinstance(a, np.memmap) and getattr(a, "filename", None):
            found = a
        a = getattr(a, "base", None)
    return found


def _local_scratch(size: int) -> str | None:
    """A folder on a local disk with room for ``size`` bytes: the system's
    temporary folder, unless it is on a network or held in memory (a cluster
    node's /tmp often is: a copy there would cost the memory it is meant to
    spare)."""
    from polariseq.resources import MEMORY_FILESYSTEMS, NETWORK_FILESYSTEMS, filesystem_type

    tmp = tempfile.gettempdir()
    kind = filesystem_type(tmp)
    if kind is None or kind in NETWORK_FILESYSTEMS or kind in MEMORY_FILESYSTEMS:
        return None
    try:
        free = shutil.disk_usage(tmp).free
    except OSError:
        return None
    return tmp if free > 2 * size else None


def _local_copies(arrays: list[Any], folder: str) -> tuple[list[Any], Any]:
    """The arrays as views of copies of their files in ``folder`` (each file
    copied in one sequential read), and a context that removes the copies."""
    work = tempfile.mkdtemp(prefix="polariseq_inputs_", dir=folder)
    copies: dict[str, np.memmap] = {}
    out = []
    try:
        for a in arrays:
            src = _source_file(a)
            if src is None:
                out.append(a)
                continue
            if src.filename not in copies:
                dst = os.path.join(work, f"{len(copies)}_{os.path.basename(src.filename)}")
                shutil.copyfile(src.filename, dst)
                copies[src.filename] = np.load(dst, mmap_mode="c")
            new = copies[src.filename]
            offset = a.__array_interface__["data"][0] - src.__array_interface__["data"][0]
            out.append(np.ndarray(a.shape, a.dtype, buffer=new, offset=offset, strides=a.strides))
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    @contextlib.contextmanager
    def removed_after():
        try:
            yield
        finally:
            # a mapped file cannot be removed on Windows: it goes with the temporary folder later
            shutil.rmtree(work, ignore_errors=True)

    return out, removed_after()


def graph_inputs(arrays: list[Any], held: int, working: int) -> tuple[list[Any], str, Any]:
    """Where a graph step reads ``arrays`` from: the arrays to use, how, and a
    context to run the step in (it removes local copies afterwards).

    How: ``"memory"`` (already in memory), ``"loaded"`` (read into memory
    now, in one sequential pass), ``"mapped"`` (in place from the project's
    files on a local disk), ``"mapped-local-copy"`` (the project is on a
    network file system: copied to a local disk first, read in place from
    there) or ``"mapped-network"`` (in place from a network file system: no
    local disk had room; warned).

    Loaded when ``held`` (what the run holds), the inputs and ``working``
    (the step's working memory) fit the budget and the inputs and the working
    memory fit the memory free now, within a cluster job's allocation
    (``planning.loads_inputs``, the plan's rule). Else in place, which on a
    local disk costs little. On a network file system it can stall for hours:
    the steps read in random order, every page not in the cache is a round
    trip, and the node drops cached pages when other jobs need the memory;
    on a cluster's Lustre file system, Leiden on Tahoe-100M's graph ran at 0.03 cores for
    hours, and a read failing there killed the neighbours step (a failed read
    through a memory map is a bus error, which no code can catch). Asking the
    system to keep the pages cached (``madvise``) did not help there."""
    from polariseq import budget
    from polariseq.planning import loads_inputs
    from polariseq.resources import free_memory_now, is_network_filesystem

    mapped = [a for a in arrays if mapping(a) is not None]
    size = sum(int(a.nbytes) for a in mapped)
    if size == 0:
        return list(arrays), "memory", contextlib.nullcontext()
    if loads_inputs(held, size, working, budget().ram_bytes, free_memory_now()):
        loaded = [in_ram(a) for a in arrays]
        return loaded, "loaded", contextlib.nullcontext()
    files = {f.filename for f in (_source_file(a) for a in mapped) if f is not None}
    if not any(is_network_filesystem(f) for f in files):
        return list(arrays), "mapped", contextlib.nullcontext()
    folder = _local_scratch(size)
    if folder is not None:
        copies, cleanup = _local_copies(list(arrays), folder)
        return copies, "mapped-local-copy", cleanup
    need = (held + size + working) / 1e9
    warnings.warn(
        f"a graph step reads {size / 1e9:.1f} GB in place from a network file system (the project "
        f"folder), which can stall for hours, and no local disk has room for a copy; a budget of "
        f"{need:.1f} GB (ps.set_budget(ram_gb=...)) lets it read them into memory",
        RuntimeWarning, stacklevel=3)
    return list(arrays), "mapped-network", contextlib.nullcontext()
