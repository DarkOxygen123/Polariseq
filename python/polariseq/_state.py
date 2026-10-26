"""Saved analyses: ``adata.save(name)`` and ``ps.load(project, name)``.

A saved analysis is a folder ``<project>/states/<name>/``: ``manifest.json``
and the arrays it names, one ``.npy`` file each. The counts are not in it.
They are pointed to:

- **Out of core** the on-disk copy of the counts is already in the project
  (``runs/<dataset>/matrix/``); the manifest records the view the steps left
  on it (which cells and genes, each cell's normalization factor, the log), and
  loading reopens the copy with that view. If the copy was deleted it is made
  again from the source files.
- **In memory** the manifest records which of the source files' cells and
  genes the matrix holds and the steps that rewrote its values, each with the
  genes it saw (``AnnData._xlog``). Loading reads those cells from the files
  and runs the same kernels in the same order, so the matrix is the one saved,
  to the last bit; a checksum of its first rows, taken at save, confirms it. A
  matrix with no such record (assigned by hand, dense, read from 10x files) is
  stored in the folder instead.

Results the steps keep in the project (``runs/<dataset>/results/``) are
memory-mapped ``.npy`` files; the saved analysis links them (a hard link: no
copy, and a later run of the step, which replaces its own file, leaves the
saved one as it was). An array changed in memory since it was read, or one with
no file, is written.

Each source file is recorded by path (relative to the project when inside it,
so a project moved with its data keeps loading), size, modification time and
shape. A file that is gone stops the load with :class:`SourceMovedError`,
saying where it was expected; ``relink`` points to its new place.

The manifest's encoding is typed JSON, never pickle, so loading a saved
analysis runs no code from it.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import re
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = 1
#: Arrays at least this large are memory-mapped on load (pages load when read);
#: smaller ones are read, so a saved analysis holds few open files.
_MAP_BYTES = 1 << 20
#: Rows of an in-memory matrix whose checksum confirms a rebuild.
_CHECK_ROWS = 256
#: What a dataset object carries besides its matrix and annotations.
_ATTRS = ("name", "mode", "source_format", "batch_key", "labels", "genes", "_n_obs", "_n_vars",
          "_nnz", "_encoding", "_est_memory_gb", "_counts", "_maps", "_rows", "_cols", "_xlog",
          "analysis", "_derived")


class SourceMovedError(FileNotFoundError):
    """A saved analysis points to a source file that is no longer where it was."""


# ── save ─────────────────────────────────────────────────────────────────────


def save(adata: Any, name: str, *, overwrite: bool = False) -> Path:
    from polariseq import LazyAnnData, MergedLazyAnnData, __version__
    from polariseq._project import require

    if not re.fullmatch(r"[\w.+-]+", name) or name.startswith("."):
        raise ValueError(f"a saved analysis is named with letters, digits and . _ + - only, not {name!r}")
    proj = require("Saving an analysis")
    states = proj.path / "states"
    dest = states / name
    if dest.exists() and not overwrite:
        raise FileExistsError(f"a saved analysis {name!r} exists in {states}; pass overwrite=True "
                              "to replace it")
    states.mkdir(exist_ok=True)
    tmp = states / f".{name}.saving-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        w = _Writer(tmp)
        kind = ("merged" if isinstance(adata, MergedLazyAnnData)
                else "lazy" if isinstance(adata, LazyAnnData) and adata.source_format == "h5ad"
                else "anndata")
        sources = _source_files(adata)
        manifest = {
            "format": FORMAT,
            "polariseq": __version__,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "shape": list(adata.shape),
            "sources": [_file_entry(p, proj.path) for p in sources],
            "attrs": {k: w.enc(sorted(v) if isinstance(v, set) else v, k)
                      for k in _ATTRS if hasattr(adata, k) for v in [getattr(adata, k)]},
            "source": w.enc(_without_paths(getattr(adata, "_source", None)), "_source"),
            "matrix": _save_matrix(adata, w, proj.path, sources),
            "obs_names": w.enc(_names(adata.obs_names), "obs_names"),
            "var_names": w.enc(_names(adata.var_names), "var_names"),
            "obs": w.enc(dict(adata.obs), "obs"),
            "var": w.enc(dict(adata.var), "var"),
            "obsm": w.enc(dict(adata.obsm), "obsm"),
            "obsp": w.enc(dict(adata.obsp), "obsp"),
            "uns": w.enc(dict(adata.uns), "uns"),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if dest.exists():
            old = states / f".{name}.replaced-{uuid.uuid4().hex[:8]}"
            try:
                dest.rename(old)
            except PermissionError:
                raise PermissionError(
                    f"cannot replace the saved analysis {name!r}: its files are open in this session "
                    "(an analysis loaded from it; Windows keeps open files in place). Save it under "
                    "another name, or close the loaded one first") from None
            tmp.rename(dest)
            # Files of the old one may still be mapped (Windows); they go at a later save.
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    for left in states.glob(".*.replaced-*"):
        shutil.rmtree(left, ignore_errors=True)
    return dest


def _source_files(adata: Any) -> list[str]:
    src = getattr(adata, "_source", None)
    if src is not None:
        return [os.fspath(p) for p in src["paths"]]
    path = getattr(adata, "source_path", None)
    return [os.fspath(path)] if path else []


def _without_paths(src: dict | None) -> dict | None:
    """``_source`` with its paths left out: they are the manifest's ``sources``."""
    return None if src is None else {k: v for k, v in src.items() if k != "paths"}


def _file_entry(path: str, root: Path) -> dict[str, Any]:
    from polariseq import scan_h5ad

    p = Path(path).resolve()
    st = p.stat()
    entry: dict[str, Any] = {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}
    with contextlib.suppress(ValueError):
        entry["in_project"] = p.relative_to(root).as_posix()
    if p.is_file():
        with contextlib.suppress(Exception):  # not an .h5ad (10x): size and time identify it
            s = scan_h5ad(str(p))
            entry["shape"] = [int(s["n_obs"]), int(s["n_vars"]), int(s["n_elements"])]
    return entry


def _names(names: Any) -> np.ndarray:
    return np.array([str(n).encode() for n in names], dtype=bytes)


def _save_matrix(adata: Any, w: _Writer, root: Path, sources: list[str]) -> dict[str, Any]:
    from polariseq import DiskMatrix, LazyAnnData, SparseMatrix

    X = adata._X if isinstance(adata, LazyAnnData) else adata.X
    if isinstance(X, DiskMatrix):
        prefix = Path(X.prefix).resolve()
        record = prefix.parent / "source.json"
        scale = X.scale_factors()
        return {
            "kind": "store",
            "prefix": _rel(prefix, root),
            "file_shape": list(X.file_shape),
            "block_rows": int(X.block_rows),
            "nnz": int(X.nnz),
            "store_source": json.loads(record.read_text()) if record.exists() else None,
            "row_keep": w.enc(np.asarray(X.row_mask()), "matrix"),
            "col_keep": w.enc(np.asarray(X.col_mask()), "matrix"),
            "scale": w.enc(None if scale is None else np.asarray(scale), "matrix"),
            "log1p": bool(X.transforms()[1]),
        }
    log = getattr(adata, "_xlog", None)
    src = getattr(adata, "_source", None)
    if (isinstance(X, SparseMatrix) and log is not None and src is not None and sources
            and all(str(p).endswith(".h5ad") for p in sources)):
        k = min(_CHECK_ROWS, X.shape[0])
        digest = _digest(X[:k]) if k else None
        sample = _rebuild_rows(src, sources, adata._rows, adata._cols, log, k)
        if sample is None or (k and _digest(sample) != digest):
            warnings.warn("the matrix does not follow from its source files and the recorded steps "
                          "(changed by hand, or the files changed since they were read); it is "
                          "stored in the saved analysis", stacklevel=4)
        else:
            return {"kind": "rebuild", "nnz": int(X.nnz), "check_rows": k, "digest": digest}
    if isinstance(X, SparseMatrix):
        data, indices, indptr = X.to_arrays()
        return {"kind": "stored-csr", "shape": list(X.shape), "data": w.enc(np.asarray(data), "X"),
                "indices": w.enc(np.asarray(indices), "X"), "indptr": w.enc(np.asarray(indptr), "X")}
    if X is None:
        return {"kind": "none"}
    if hasattr(X, "tocsr"):
        X = X.tocsr()
        return {"kind": "stored-scipy", "value": w.enc(X, "X")}
    return {"kind": "stored-dense", "value": w.enc(np.asarray(X), "X")}


def _rel(p: Path, root: Path) -> dict[str, str]:
    out = {"path": str(p)}
    with contextlib.suppress(ValueError):
        out["in_project"] = p.relative_to(root).as_posix()
    return out


def _digest(X: Any) -> str:
    data, indices, indptr = X.to_arrays()
    h = hashlib.sha256()
    for a in (data, indices, indptr):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class _Writer:
    """Encodes values as typed JSON, writing arrays to files of their own."""

    def __init__(self, folder: Path) -> None:
        self.folder, self.n = folder, 0

    def put(self, arr: np.ndarray) -> dict[str, str]:
        """``arr`` in a file of its own: a link to the file it is mapped from
        when that file holds it unchanged (read back as ``arr``'s type when the
        file's differs, as int32 graph indices over uint32 files), else written."""
        self.n += 1
        fname = f"a{self.n:05d}.npy"
        dst = self.folder / fname
        found = _unchanged_file(arr)
        if found is not None:
            src, dtype = found
            try:
                os.link(src, dst)
                return {"array": fname} if dtype == arr.dtype else {"array": fname, "view": arr.dtype.str}
            except OSError:
                pass  # another file system: write it
        np.save(dst, np.ascontiguousarray(arr), allow_pickle=False)
        return {"array": fname}

    def enc(self, v: Any, where: str) -> Any:
        import pandas as pd

        from polariseq import _CSRGraph

        if v is None or isinstance(v, (bool, str)):
            return {"v": v}
        if isinstance(v, (int, float)) and not isinstance(v, np.generic):
            return {"v": v}
        if isinstance(v, np.generic):
            return {"scalar": v.dtype.str, "v": v.item()}
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                flat = v.ravel()
                if all(isinstance(x, str) for x in flat):
                    return {"strings": self.put(_names(flat).reshape(v.shape))}
                return {"objects": [self.enc(x, where) for x in flat], "shape": list(v.shape)}
            return self.put(v)
        if isinstance(v, pd.Categorical):
            return {"categorical": self.put(np.asarray(v.codes)),
                    "categories": self.enc(np.asarray(v.categories, dtype=object
                                                      if v.categories.dtype == object else None), where),
                    "ordered": bool(v.ordered)}
        if isinstance(v, pd.DataFrame):
            return {"frame": [[self.enc(c, where), self.enc(_column(v[c]), f"{where}[{c!r}]")]
                              for c in v.columns],
                    "index": self.enc(_column(v.index), where),
                    "index_name": self.enc(v.index.name, where)}
        if isinstance(v, (pd.Series, pd.Index)):
            return {"series": self.enc(_column(v), where), "name": self.enc(v.name, where),
                    "is_index": isinstance(v, pd.Index)}
        if hasattr(v, "tocsr") and not isinstance(v, _CSRGraph):
            v = v.tocsr()  # any SciPy sparse matrix, saved by rows
        if isinstance(v, _CSRGraph) or (hasattr(v, "tocsr") and hasattr(v, "indptr")):
            return {"csr": {"data": self.put(v.data), "indices": self.put(v.indices),
                            "indptr": self.put(v.indptr), "shape": list(v.shape)}}
        if isinstance(v, dict):
            return {"dict": [[self.enc(k, where), self.enc(x, f"{where}[{k!r}]")] for k, x in v.items()]}
        if isinstance(v, (list, tuple)):
            return {"list" if isinstance(v, list) else "tuple":
                    [self.enc(x, f"{where}[{i}]") for i, x in enumerate(v)]}
        raise TypeError(f"cannot save {where}: a {type(v).__name__}; saved analyses hold arrays, "
                        "tables, numbers, text, and lists and dicts of them. Remove it or convert it "
                        "before saving.")


def _column(v: Any) -> Any:
    """A pandas column as an array, or as a Categorical when it is one."""
    import pandas as pd

    if isinstance(v.dtype, pd.CategoricalDtype):
        return pd.Categorical(v)
    a = v.to_numpy()
    if a.dtype == object and any(x is pd.NA for x in a):
        raise TypeError(f"cannot save column {v.name!r}: it holds pandas' NA; fill it or "
                        "convert it to float (NaN) first")
    return a


def _unchanged_file(arr: Any) -> tuple[str, np.dtype] | None:
    """The ``.npy`` file ``arr`` is mapped from, and the file's type, when
    ``arr`` covers that file's values whole and its bytes were not changed in
    memory since; else ``None``. ``arr`` may be a view of the mapping (a
    reinterpretation, a full slice)."""
    root = arr
    while root is not None and not (isinstance(root, np.memmap) and root.filename):
        root = getattr(root, "base", None)
    if root is None or not str(root.filename).endswith(".npy") or not arr.flags.c_contiguous:
        return None
    try:
        disk = np.load(root.filename, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if (disk.nbytes != arr.nbytes or disk.itemsize != arr.itemsize
            or arr.__array_interface__["data"][0] != root.__array_interface__["data"][0]
            or root.nbytes != disk.nbytes):
        return None
    # Views are copy-on-write: a change in memory never reaches the file.
    if not np.array_equal(disk.reshape(-1).view(np.uint8), arr.reshape(-1).view(np.uint8)):
        return None
    return str(root.filename), disk.dtype


# ── rebuilding an in-memory matrix ──────────────────────────────────────────


def _read_source(src: dict, paths: list[str], r0: int | None = None, r1: int | None = None) -> Any:
    """The counts of the source files as one ``SparseMatrix`` (rows ``[r0, r1)``
    of the joined files, or all of them)."""
    from polariseq import (
        SparseMatrix,
        _read_h5ad_matrix,
        _remap_rows,
        read_h5ad_many,
        read_h5ad_rows,
    )

    maps = src.get("maps")
    n_cols = int(src["n_cols"])
    if r0 is None:
        if maps is None:
            X = _read_h5ad_matrix(paths[0])
        else:
            X = read_h5ad_many(paths, [np.asarray(m, dtype=np.uint32) for m in maps], n_cols)
        return X if isinstance(X, SparseMatrix) else None
    counts = src.get("counts") or [int(src["n_rows"])]
    starts = np.cumsum([0, *counts])
    data, indices, indptr = [], [], [np.zeros(1, dtype=np.int64)]
    for k, path in enumerate(paths):
        a, b = max(r0, int(starts[k])), min(r1, int(starts[k + 1]))
        if a >= b:
            continue
        rows = read_h5ad_rows(path, a - int(starts[k]), b - int(starts[k]))
        if maps is None:
            d, ix, ip = (np.asarray(rows["data"]), np.asarray(rows["indices"]),
                         np.asarray(rows["indptr"], dtype=np.int64))
        else:
            d, ix, ip = _remap_rows(rows, np.asarray(maps[k], dtype=np.uint32))
        data.append(d)
        indices.append(ix)
        indptr.append(np.asarray(ip[1:], dtype=np.int64) + indptr[-1][-1])
    return SparseMatrix.from_arrays(
        np.ascontiguousarray(np.concatenate(data) if data else np.zeros(0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(indices) if indices else np.zeros(0), dtype=np.uint32),
        np.ascontiguousarray(np.concatenate(indptr), dtype=np.uint32), r1 - r0, n_cols)


def _rebuild_rows(src: dict, paths: list[str], rows: Any, cols: Any, log: list, k: int) -> Any:
    """The first ``k`` rows of the matrix, rebuilt from the files (``None``:
    they cannot be read)."""
    if k == 0:
        return None
    want = np.arange(k) if rows is None else np.asarray(rows[:k], dtype=np.int64)
    r0, r1 = int(want.min()), int(want.max()) + 1
    src = {**src, "counts": _file_counts(src, paths)}
    try:
        X = _read_source(src, paths, r0, r1)
    except Exception:  # noqa: BLE001  (an unreadable source: stored instead)
        return None
    return _replay(X, want - r0, cols, log)


def _file_counts(src: dict, paths: list[str]) -> list[int]:
    from polariseq import scan_h5ad

    if len(paths) == 1:
        return [int(src["n_rows"])]
    return [int(scan_h5ad(p)["n_obs"]) for p in paths]


def _replay(X: Any, rows: Any, cols: Any, log: list) -> Any:
    """Keep ``rows`` of the counts ``X``, run the recorded steps on the genes
    each saw, then keep ``cols``: the kernels of the session, in its order."""
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and np.all(np.diff(rows) > 0):
            mask = np.zeros(X.shape[0], dtype=bool)
            mask[rows] = True
            X.retain_rows_mask(mask)
        else:
            X = X[rows]
    n_src = X.shape[1]
    cur: np.ndarray | None = None  # source genes of X's columns; None: all
    totals = target = None
    for op in log:
        X, cur = _to_cols(X, cur, op.get("cols"), n_src)
        if op["op"] == "normalize_total":
            totals = np.asarray(X.normalize_total(float(op["target_sum"])), dtype=np.float32)
            target = op["target_sum"]
        elif op["op"] == "log1p":
            X.log1p()
        elif op["op"] == "renormalize":
            back = totals / np.float32(target)
            X.renormalize_inplace(back, bool(op["from_log"]), float(op["target_sum"]),
                                  bool(op["log1p"]))
            totals = target = None
        else:
            raise ValueError(f"unknown step {op['op']!r} in the record")
    X, _ = _to_cols(X, cur, cols, n_src)
    return X


def _to_cols(X: Any, cur: Any, want: Any, n_src: int) -> tuple[Any, Any]:
    """``X`` (over the source genes ``cur``) narrowed to the source genes ``want``."""
    if want is None:
        return X, cur
    want = np.asarray(want, dtype=np.int64)
    if cur is None:
        pos = want
    else:
        inv = np.full(n_src, -1, dtype=np.int64)
        inv[cur] = np.arange(cur.size)
        pos = inv[want]
        if np.any(pos < 0):
            raise ValueError("the record asks for genes the matrix no longer holds")
    if pos.size == X.shape[1] and np.array_equal(pos, np.arange(pos.size)):
        return X, want
    return X[:, pos], want


# ── load ─────────────────────────────────────────────────────────────────────


def load(project: str | os.PathLike, name: str, *, relink: dict | None = None) -> Any:
    from polariseq import LazyAnnData, MergedLazyAnnData, __version__
    from polariseq._project import project as declare

    proj = declare(project)
    folder = proj.path / "states" / name
    manifest = folder / "manifest.json"
    if not manifest.exists():
        have = sorted(p.name for p in (proj.path / "states").glob("*")
                      if (p / "manifest.json").exists()) if (proj.path / "states").exists() else []
        raise FileNotFoundError(f"no saved analysis {name!r} in {proj.path}; saved: "
                                f"{', '.join(have) if have else 'none'}")
    m = json.loads(manifest.read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{folder} is saved in format {m.get('format')}; Polariseq {__version__} "
                         f"reads format {FORMAT}")
    if m.get("polariseq") != __version__:
        warnings.warn(f"{name!r} was saved by Polariseq {m.get('polariseq')}; this is {__version__}",
                      stacklevel=2)
    r = _Reader(folder)
    paths = [_resolve(e, proj.path, relink or {}) for e in m["sources"]]
    attrs = {k: r.dec(v) for k, v in m["attrs"].items()}
    if attrs.get("_derived") is not None:
        attrs["_derived"] = set(attrs["_derived"])
    src = r.dec(m["source"])
    if src is not None:
        src["paths"] = paths
    X = _load_matrix(m["matrix"], r, proj.path, paths, src, attrs, m["kind"], name)

    obs_names = [b.decode() for b in r.dec(m["obs_names"])]
    var_names = [b.decode() for b in r.dec(m["var_names"])]
    kind = m["kind"]
    if kind in ("lazy", "merged"):
        cls = MergedLazyAnnData if kind == "merged" else LazyAnnData
        obj = cls.__new__(cls)
        obj._init_state()
        if kind == "merged":
            obj.sources, obj.source_path = list(paths), None
        else:
            obj.source_path = paths[0]
        obj._scan_time = 0.0
        obj._X, obj._materialized = X, True
        obj.obs_names, obj.var_names = obs_names, var_names
    else:
        from polariseq import AnnData

        obj = AnnData.__new__(AnnData)
        obj._X = X
        obj.obs_names, obj.var_names = obs_names, var_names
    for k, v in attrs.items():
        setattr(obj, k, v)
    obj._source = src
    obj.obs, obj.var = r.dec(m["obs"]), r.dec(m["var"])
    obj.obsm, obj.obsp, obj.uns = r.dec(m["obsm"]), r.dec(m["obsp"]), r.dec(m["uns"])
    return obj


def _resolve(entry: dict, root: Path, relink: dict) -> str:
    """Where a recorded source file is now, checked against what was recorded."""
    old = entry["path"]
    moved = relink.get(old)
    if moved is None and "in_project" in entry:
        moved = str(root / entry["in_project"])
    p = Path(moved or old).expanduser()
    if not p.exists():
        when = datetime.datetime.fromtimestamp(entry["mtime"]).strftime("%Y-%m-%d %H:%M")
        raise SourceMovedError(
            f"this analysis reads its counts from {old} ({entry['bytes'] / 1e9:.2f} GB, modified "
            f"{when}), which is no longer there. Put the file back at that path, or give its new "
            f"place: ps.load(..., relink={{{old!r}: '<new path>'}})")
    st = p.stat()
    if p.is_file() and st.st_size != entry["bytes"]:
        raise ValueError(f"{p} is {st.st_size:,} bytes; the analysis was saved from a file of "
                         f"{entry['bytes']:,}: not the same data")
    if "shape" in entry:
        from polariseq import scan_h5ad

        s = scan_h5ad(str(p))
        now = [int(s["n_obs"]), int(s["n_vars"]), int(s["n_elements"])]
        if now != entry["shape"]:
            raise ValueError(f"{p} holds {now[0]:,} x {now[1]:,} ({now[2]:,} values); the analysis "
                             f"was saved from {entry['shape'][0]:,} x {entry['shape'][1]:,}")
    return str(p)


def _load_matrix(mx: dict, r: _Reader, root: Path, paths: list[str], src: Any, attrs: dict,
                 kind: str, name: str) -> Any:
    from polariseq import DiskMatrix, SparseMatrix

    k = mx["kind"]
    if k == "store":
        prefix = Path(mx["prefix"]["path"])
        if "in_project" in mx["prefix"]:
            prefix = root / mx["prefix"]["in_project"]
        n, g = mx["file_shape"]
        record = prefix.parent / "source.json"
        current = (prefix.with_suffix(".indptr").exists() and record.exists()
                   and json.loads(record.read_text()) == mx["store_source"])
        if current:
            dm = DiskMatrix.open(str(prefix), n, g, mx["block_rows"])
        else:
            dm = _copy_again(prefix, paths, src, attrs, kind, mx, n, g)
        scale = r.dec(mx["scale"])
        dm.restore_view(np.ascontiguousarray(r.dec(mx["row_keep"]), dtype=bool),
                        np.ascontiguousarray(r.dec(mx["col_keep"]), dtype=bool),
                        None if scale is None else np.ascontiguousarray(scale, dtype=np.float32),
                        bool(mx["log1p"]))
        return dm
    if k == "rebuild":
        X = _read_source(src, paths)
        if X is None:
            raise ValueError(f"{paths[0]}: X is no longer a sparse matrix")
        X = _replay(X, attrs.get("_rows"), attrs.get("_cols"), attrs.get("_xlog") or [])
        if int(X.nnz) != mx["nnz"] or (mx["check_rows"] and _digest(X[:mx["check_rows"]]) != mx["digest"]):
            raise ValueError(f"the matrix rebuilt from {', '.join(paths)} differs from the one saved "
                             f"in {name!r}: the source files changed since")
        return X
    if k == "stored-csr":
        data, indices, indptr = r.dec(mx["data"]), r.dec(mx["indices"]), r.dec(mx["indptr"])
        return SparseMatrix.from_arrays(np.ascontiguousarray(data, dtype=np.float32),
                                        np.ascontiguousarray(indices, dtype=np.uint32),
                                        np.ascontiguousarray(indptr, dtype=np.uint32), *mx["shape"])
    if k in ("stored-scipy", "stored-dense"):
        return r.dec(mx["value"])
    if k == "none":
        return None
    raise ValueError(f"unknown matrix kind {k!r}")


def _copy_again(prefix: Path, paths: list[str], src: Any, attrs: dict, kind: str, mx: dict,
                n: int, g: int) -> Any:
    """The on-disk copy of the counts, made again from the sources (it was
    deleted, or the files are not the ones it was made from)."""
    from polariseq import DiskMatrix, _file_record

    print(f"[polariseq] the on-disk copy at {prefix.parent} is missing or out of date; copying "
          "the counts again from the source files")
    matrix = prefix.parent
    if matrix.exists():
        shutil.rmtree(matrix)
    matrix.mkdir(parents=True)
    if kind == "merged":
        maps = [np.asarray(x, dtype=np.uint32) for x in attrs["_maps"]]
        dm = DiskMatrix.from_h5ad_many(paths, maps, g, str(prefix), mx["block_rows"])
        record = {"sources": [_file_record(p) for p in paths], "genes": attrs.get("genes"),
                  "n_obs": n, "n_vars": g}
    else:
        dm = DiskMatrix.from_h5ad(paths[0], str(prefix), mx["block_rows"])
        record = {**_file_record(paths[0]), "n_obs": n, "n_vars": g, "nnz": int(dm.nnz)}
    (matrix / "source.json").write_text(json.dumps(record, indent=1))
    return dm


class _Reader:
    """Decodes what :class:`_Writer` encoded."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def get(self, fname: str) -> np.ndarray:
        p = self.folder / fname
        mmap = "c" if p.stat().st_size >= _MAP_BYTES else None
        return np.load(p, mmap_mode=mmap, allow_pickle=False)

    def dec(self, node: Any) -> Any:
        import pandas as pd

        if "v" in node:
            if "scalar" in node:
                return np.dtype(node["scalar"]).type(node["v"])
            return node["v"]
        if "array" in node:
            a = self.get(node["array"])
            return a.view(np.dtype(node["view"])) if "view" in node else a
        if "strings" in node:
            a = self.dec(node["strings"])
            out = np.empty(a.shape, dtype=object)
            out.ravel()[:] = [b.decode() for b in a.ravel()]
            return out
        if "objects" in node:
            out = np.empty(len(node["objects"]), dtype=object)
            out[:] = [self.dec(x) for x in node["objects"]]
            return out.reshape(node["shape"])
        if "categorical" in node:
            return pd.Categorical.from_codes(np.asarray(self.dec(node["categorical"])),
                                             self.dec(node["categories"]), ordered=node["ordered"])
        if "frame" in node:
            cols = {self.dec(c): self.dec(v) for c, v in node["frame"]}
            index = pd.Index(self.dec(node["index"]), name=self.dec(node["index_name"]))
            return pd.DataFrame(cols, index=index)
        if "series" in node:
            values, label = self.dec(node["series"]), self.dec(node["name"])
            return pd.Index(values, name=label) if node["is_index"] else pd.Series(values, name=label)
        if "csr" in node:
            c = node["csr"]
            data, indices, indptr = self.dec(c["data"]), self.dec(c["indices"]), self.dec(c["indptr"])
            try:
                import scipy.sparse as sp

                return sp.csr_matrix((data, indices, indptr), shape=tuple(c["shape"]), copy=False)
            except ImportError:
                from polariseq import _CSRGraph

                return _CSRGraph(indptr, indices, data, tuple(c["shape"]))
        if "dict" in node:
            return {self.dec(k): self.dec(v) for k, v in node["dict"]}
        if "list" in node:
            return [self.dec(x) for x in node["list"]]
        if "tuple" in node:
            return tuple(self.dec(x) for x in node["tuple"])
        raise ValueError(f"unknown entry in the manifest: {list(node)}")
