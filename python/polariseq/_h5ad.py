"""``adata.write_h5ad(path)``: the anndata on-disk format, written natively.

The layout follows anndata's specification (encoding types and versions on
every element), so anndata, Scanpy, cellxgene and the R readers open the file.
The Rust core writes the HDF5 objects (:class:`polariseq._polariseq.H5Writer`);
an out-of-core matrix is streamed into the file from its on-disk copy, a few
blocks at a time.
"""

from __future__ import annotations

import os
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

#: Rows per block when a matrix is streamed into the file (an out-of-core
#: matrix uses its own block size).
BLOCK_ROWS = 65_536


def write_h5ad(adata: Any, path: str | os.PathLike, *, counts: bool = False) -> Path:
    from polariseq import H5Writer, _export_parts

    path = Path(path).expanduser()
    if path.is_dir():
        raise ValueError(f"write_h5ad: {path} is a folder; give the path of the .h5ad file to write")
    if not path.parent.exists():
        raise FileNotFoundError(f"write_h5ad: the folder {path.parent} does not exist; create it "
                                "first")
    x, obs, var, obsm, obsp, uns = _export_parts(adata)
    raw = _raw_counts(adata, x) if counts else None
    uns["polariseq"] = _pointer(adata, x, counts)
    tmp = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    w = H5Writer(str(tmp))
    try:
        e = _Encoder(w)
        e.kind("/", "anndata", "0.1.0")
        if x is not None:
            e.matrix("X", x)
        e.frame("obs", obs)
        e.frame("var", var)
        e.mapping("obsm", obsm)
        e.mapping("obsp", obsp)
        e.mapping("varm", {})
        e.mapping("varp", {})
        e.mapping("layers", {} if raw is None else {"counts": raw})
        e.mapping("uns", uns)
        w.close()
        os.replace(tmp, path)
    except BaseException:
        w.close()
        tmp.unlink(missing_ok=True)
        raise
    if e.skipped:
        warnings.warn("write_h5ad: left out of the file (the format cannot hold them): "
                      + "; ".join(e.skipped), stacklevel=3)
    return path


def _raw_counts(adata: Any, x: Any) -> Any:
    """The raw counts of the current cells and genes, for ``layers['counts']``."""
    from polariseq import _is_disk

    if _is_disk(x):
        view = x.subset(None, None)
        view.restore_view(np.ascontiguousarray(x.row_mask(), dtype=bool),
                          np.ascontiguousarray(x.col_mask(), dtype=bool), None, False)
        return view
    if getattr(adata, "_xlog", None) == []:
        return x  # still the counts as read
    src = getattr(adata, "_source", None)
    if src is None or not all(str(p).endswith(".h5ad") for p in src["paths"]):
        raise ValueError("write_h5ad(counts=True) needs the raw counts: this matrix was changed and "
                         "has no source .h5ad to read them from (made from an array, or read from "
                         "10x files). Write without counts, or read the data with ps.read_h5ad")
    from polariseq._state import _read_source, _replay

    paths = [os.fspath(p) for p in src["paths"]]
    return _replay(_read_source(src, paths), getattr(adata, "_rows", None),
                   getattr(adata, "_cols", None), [])


def _pointer(adata: Any, x: Any, counts: bool) -> dict[str, Any]:
    """``uns['polariseq']``: where the raw counts are, and what made ``X`` from them."""
    from polariseq import __version__, _is_disk

    src = getattr(adata, "_source", None)
    paths = [str(Path(os.fspath(p)).resolve()) for p in src["paths"]] if src else []
    stats = [os.stat(p) if os.path.exists(p) else None for p in paths]
    rows = getattr(adata, "_rows", None)
    cols = getattr(adata, "_cols", None)
    if _is_disk(x):
        scaled, logged = x.transforms()
        steps = (["normalize_total"] if scaled else []) + (["log1p"] if logged else [])
    elif getattr(adata, "_xlog", None) is not None:
        steps = [op["op"] + (f"(target_sum={op['target_sum']:g})" if "target_sum" in op else "")
                 for op in adata._xlog]
    else:
        steps = ["unknown: the matrix was set by hand"]
    return {
        "version": __version__,
        "dataset": str(getattr(adata, "name", "") or ""),
        "analysis": str(getattr(adata, "analysis", "") or ""),
        "source_paths": paths,
        "source_bytes": np.array([s.st_size if s else -1 for s in stats], dtype=np.int64),
        "source_mtimes": np.array([int(s.st_mtime) if s else -1 for s in stats], dtype=np.int64),
        "rows": np.arange(adata.n_obs, dtype=np.int64) if rows is None else np.asarray(rows, np.int64),
        "cols": np.arange(adata.n_vars, dtype=np.int64) if cols is None else np.asarray(cols, np.int64),
        "x_steps": steps or ["none: X holds the raw counts"],
        "counts_in_file": bool(counts),
    }


class _Encoder:
    """anndata's encoding of each kind of value."""

    def __init__(self, w: Any) -> None:
        self.w = w
        self.skipped: list[str] = []

    def kind(self, path: str, etype: str, version: str) -> None:
        self.w.attr_str(path, "encoding-type", etype)
        self.w.attr_str(path, "encoding-version", version)

    def matrix(self, path: str, x: Any) -> None:
        from polariseq import _is_disk, _is_sparse

        if _is_sparse(x) or _is_disk(x):
            self.w.group(path)
            self.kind(path, "csr_matrix", "0.1.0")
            self.w.attr_ints(path, "shape", list(map(int, x.shape)))
            # an out-of-core matrix's own blocks, at most BLOCK_ROWS rows: a few are in flight at once
            block = int(x.block_rows) if _is_disk(x) else BLOCK_ROWS
            self.w.csr(path, x, max(1, min(block, BLOCK_ROWS)))
        elif hasattr(x, "indptr"):
            self.csr(path, x)
        else:
            self.array(path, np.asarray(x))

    def csr(self, path: str, m: Any) -> None:
        if hasattr(m, "tocsr") and getattr(m, "format", "csr") != "csr":
            m = m.tocsr()
        self.w.group(path)
        self.kind(path, "csr_matrix", "0.1.0")
        self.w.attr_ints(path, "shape", list(map(int, m.shape)))
        self.w.array(f"{path}/data", np.ascontiguousarray(m.data))
        self.w.array(f"{path}/indices", np.ascontiguousarray(m.indices))
        self.w.array(f"{path}/indptr", np.ascontiguousarray(m.indptr))

    def array(self, path: str, a: np.ndarray) -> None:
        if a.dtype.kind in "biuf":
            self.w.array(path, np.ascontiguousarray(a))
            self.kind(path, "array", "0.2.0")
        elif a.dtype.kind in "OUS":
            if a.ndim != 1:
                raise TypeError(f"{path}: text arrays are written one-dimensional")
            self.w.strings(path, [x.decode() if isinstance(x, bytes) else str(x) for x in a])
            self.kind(path, "string-array", "0.2.0")
        elif a.dtype.kind == "M":
            self.array(path, a.astype(str).astype(object))
        else:
            raise TypeError(f"{path}: an array of {a.dtype} cannot be written")

    def categorical(self, path: str, cat: Any) -> None:
        self.w.group(path)
        self.kind(path, "categorical", "0.2.0")
        self.w.attr_bool(path, "ordered", bool(cat.ordered))
        n = len(cat.categories)
        dtype = np.int8 if n < 2**7 else np.int16 if n < 2**15 else np.int32
        self.w.array(f"{path}/codes", np.asarray(cat.codes).astype(dtype))
        self.array(f"{path}/categories", np.asarray(cat.categories, dtype=object
                                                    if cat.categories.dtype == object else None))

    def column(self, path: str, values: Any) -> None:
        import pandas as pd

        if isinstance(values, pd.Categorical):
            self.categorical(path, values)
            return
        a = np.asarray(values)
        if a.dtype == object and any(v is None or v is pd.NA for v in a):
            # text with gaps: as anndata stores it, a category with missing values
            self.categorical(path, pd.Categorical(a))
            return
        self.array(path, a)

    def frame(self, path: str, df: Any) -> None:
        import pandas as pd

        self.w.group(path)
        self.kind(path, "dataframe", "0.2.0")
        self.w.attr_str(path, "_index", "_index")
        names = [str(c) for c in df.columns]
        if "_index" in names:
            raise ValueError(f"{path}: a column named '_index' clashes with the index anndata stores")
        self.w.attr_strs(path, "column-order", names)
        self.array(f"{path}/_index", np.asarray([str(i) for i in df.index], dtype=object))
        for c, name in zip(df.columns, names, strict=True):
            col = df[c]
            self.column(f"{path}/{name}", pd.Categorical(col)
                        if isinstance(col.dtype, pd.CategoricalDtype) else col.to_numpy())

    def mapping(self, path: str, d: dict) -> None:
        self.w.group(path)
        self.kind(path, "dict", "0.1.0")
        for k, v in d.items():
            self.value(f"{path}/{str(k).replace('/', '_')}", v)

    def value(self, path: str, v: Any) -> None:
        import pandas as pd

        from polariseq import _CSRGraph, _is_disk, _is_sparse

        if v is None:
            pass  # as anndata does: an empty entry is not written
        elif _is_sparse(v) or _is_disk(v):
            self.matrix(path, v)
        elif isinstance(v, dict):
            self.mapping(path, v)
        elif isinstance(v, pd.DataFrame):
            self.frame(path, v)
        elif isinstance(v, pd.Categorical):
            self.categorical(path, v)
        elif isinstance(v, (pd.Series, pd.Index)):
            self.column(path, pd.Categorical(v) if isinstance(v.dtype, pd.CategoricalDtype)
                        else v.to_numpy())
        elif isinstance(v, _CSRGraph) or hasattr(v, "tocsr"):
            self.csr(path, v)
        elif isinstance(v, str):
            self.w.scalar_str(path, v)
            self.kind(path, "string", "0.2.0")
        elif isinstance(v, (bool, int, float, np.generic)) and np.ndim(v) == 0:
            self.w.scalar(path, v.item() if isinstance(v, np.generic) else v)
            self.kind(path, "numeric-scalar", "0.2.0")
        elif isinstance(v, np.ndarray):
            self.array(path, v)
        elif isinstance(v, (list, tuple)):
            a = np.asarray(v, dtype=object) if v else np.zeros(0)
            if all(isinstance(x, str) for x in v) or (
                    v and all(isinstance(x, (bool, int, float, np.generic)) for x in v)):
                self.array(path, np.asarray(v) if not all(isinstance(x, str) for x in v) else a)
            else:  # mixed or nested: as a dict keyed by position
                self.mapping(path, {str(i): x for i, x in enumerate(v)})
        else:
            self.skipped.append(f"{path} (a {type(v).__name__})")
