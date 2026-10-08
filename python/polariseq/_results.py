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

import uuid
from typing import Any

import numpy as np


def new(adata: Any, step: str, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """A writable array for ``step``'s result ``key``: memory-mapped in the
    dataset's results folder when a project is set, else in RAM."""
    from polariseq._project import current

    proj, name = current(), getattr(adata, "name", None)
    if proj is None or not name:
        return np.zeros(shape, dtype=dtype)
    folder = proj.run_dir(name) / "results" / step
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{key}.*.npy"):
        try:
            old.unlink()
        except OSError:
            pass  # still mapped (Windows); removed at a later run
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
