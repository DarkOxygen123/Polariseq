"""Cell and gene annotations: read compact, joined across files, kept compact.

A column of labels (condition, donor, cell type) is held as a pandas
``Categorical``: one small integer per cell and each label once, whatever the
number of cells. Columns are read by the Rust core, which also turns text
into codes, so no Python string is made per row. Several files read as one
have their columns joined as ``anndata.concat`` joins them: every column
kept, and a column a file lacks left missing for its cells.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from polariseq._polariseq import read_h5ad_frame


def _column(col: dict) -> Any:
    kind = col["kind"]
    if kind in ("categorical", "text"):
        return pd.Categorical.from_codes(col["codes"], categories=pd.Index(col["categories"]),
                                         ordered=bool(col.get("ordered", False)))
    if kind == "numbers":
        return col["values"]
    if kind == "nullable":
        values, mask = col["values"], col["mask"]
        if values.dtype == np.bool_:
            return pd.arrays.BooleanArray(values, mask)
        return pd.arrays.IntegerArray(values, mask)
    raise ValueError(f"unknown column kind {kind!r}")


def read_frame(path: str, axis: str = "obs", columns: list[str] | None = None,
               missing_ok: bool = False) -> dict[str, Any]:
    """The ``obs`` (or ``var``) columns of an ``.h5ad``, or only those named."""
    raw = read_h5ad_frame(path, axis, None if columns is None else [str(c) for c in columns],
                          missing_ok)
    if raw["skipped"]:
        warnings.warn(f"{path}: {axis} columns in an encoding not read: "
                      + ", ".join(f"{n} ({e})" for n, e in raw["skipped"]), stacklevel=3)
    return {name: _column(col) for name, col in raw["columns"]}


def _union_categories(parts: list[pd.Categorical | None]) -> pd.Index:
    cats = None
    for p in parts:
        if p is None:
            continue
        cats = p.categories if cats is None else cats.append(p.categories[~p.categories.isin(cats)])
    return cats if cats is not None else pd.Index([], dtype=object)


def concat_frames(frames: list[dict[str, Any]], counts: list[int]) -> dict[str, Any]:
    """The columns of several files' cells, one file after another."""
    names: list[str] = []
    for f in frames:
        names += [n for n in f if n not in names]
    out: dict[str, Any] = {}
    for name in names:
        parts = [f.get(name) for f in frames]
        present = [p for p in parts if p is not None]
        if all(isinstance(p, pd.Categorical) for p in present):
            cats = _union_categories(parts)
            codes = [np.full(n, -1, dtype=np.int32) if p is None
                     else np.where(p.codes >= 0, cats.get_indexer(p.categories)[p.codes], -1)
                     for p, n in zip(parts, counts, strict=True)]
            ordered = (len(present) == len(parts) and all(p.ordered for p in present)
                       and all(p.categories.equals(cats) for p in present))
            out[name] = pd.Categorical.from_codes(np.concatenate(codes), categories=cats,
                                                  ordered=ordered)
            continue
        series = [pd.Series(p) if p is not None else pd.Series(np.full(n, np.nan))
                  for p, n in zip(parts, counts, strict=True)]
        col = pd.concat(series, ignore_index=True)
        out[name] = col.array if isinstance(col.dtype, pd.api.extensions.ExtensionDtype) \
            else col.to_numpy()
    return out


def labels_column(labels: list[str], counts: list[int]) -> pd.Categorical:
    """Each file's label for each of its cells, as ``anndata.concat`` labels
    them: categorical, in file order."""
    return pd.Categorical.from_codes(np.repeat(np.arange(len(labels)), counts),
                                     categories=pd.Index(labels))


def take(values: Any, index: Any) -> Any:
    """``values`` at ``index`` (a mask or positions), keeping its type."""
    if isinstance(values, (np.ndarray, pd.api.extensions.ExtensionArray)):
        return values[index]
    return np.asarray(values)[index]


def as_column(values: Any) -> Any:
    """``values`` as a DataFrame column: categoricals and nullable arrays as
    they are, anything else as an array."""
    if isinstance(values, pd.api.extensions.ExtensionArray):
        return values
    return np.asarray(values)
