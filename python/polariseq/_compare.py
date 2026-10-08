"""Comparing conditions (control against treated) within cell types.

Both steps run in the Rust core: the raw counts are summed per replicate,
condition and cell type in one streamed pass over wherever they are held (the
in-memory matrix while it is raw, the on-disk store, or the source files of an
in-memory analysis already normalized), and each cell type is tested in
parallel. Python receives only the result table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_MAX = 2**32 - 1


def categorical(values: Any) -> pd.Categorical:
    """An annotation column as a categorical (labels are kept compact)."""
    if isinstance(values, pd.Categorical):
        return values
    return pd.Categorical(np.asarray(values))


def _codes(cat: pd.Categorical) -> np.ndarray:
    return np.asarray(cat.codes, dtype=np.int64)


def raw_counts(adata: Any) -> dict[str, Any]:
    """Where the raw counts of ``adata``'s current cells and genes are read."""
    from polariseq import SparseMatrix, _is_disk, _is_sparse

    X = adata.X
    if _is_disk(X):
        return {"source": X}  # the store's raw counts, under any normalization
    if "normalize" not in adata.uns and "log1p" not in adata.uns:
        if not _is_sparse(X):
            X = SparseMatrix.from_dense(np.ascontiguousarray(X, dtype=np.float32))
        return {"source": X}
    src = getattr(adata, "_source", None)
    if src is None:
        raise ValueError(
            "comparing conditions needs the raw counts, and this matrix is normalized with no "
            "file to read them from: read the data with ps.read_h5ad, or compare before "
            "normalize_total")
    rows = adata._rows if adata._rows is not None else np.arange(src["n_rows"])
    cols = adata._cols if adata._cols is not None else np.arange(src["n_cols"])
    maps = src["maps"] or [np.arange(src["n_cols"], dtype=np.uint32)]
    return {"source": list(src["paths"]), "gene_maps": [np.asarray(m, dtype=np.uint32) for m in maps],
            "n_cols": int(src["n_cols"]), "rows": np.asarray(rows, dtype=np.int64),
            "n_rows": int(src["n_rows"]), "cols": np.asarray(cols, dtype=np.uint32)}


def _per_source_row(src: dict[str, Any], unit: np.ndarray) -> np.ndarray:
    """Cells' units placed on the rows of the files they are read from."""
    if "rows" not in src:
        return unit
    out = np.full(src["n_rows"], _MAX, dtype=np.uint32)
    out[src["rows"]] = unit
    return out


def _file_args(src: dict[str, Any]) -> dict[str, Any]:
    if "rows" not in src:
        return {}
    return {"cols": src["cols"], "gene_maps": src["gene_maps"], "n_cols": src["n_cols"]}


def _units(cats: list[pd.Categorical], keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's unit (the combination of its labels) and each unit's codes."""
    codes = [_codes(c) for c in cats]
    for c in codes:
        keep = keep & (c >= 0)
    key = np.zeros(len(keep), dtype=np.int64)
    for c, cat in zip(codes, cats, strict=True):
        key = key * max(1, len(cat.categories)) + np.maximum(c, 0)
    uniq, inv = np.unique(key[keep], return_inverse=True)
    unit = np.full(len(keep), _MAX, dtype=np.uint32)
    unit[keep] = inv.astype(np.uint32)
    parts = []
    rest = uniq
    for cat in reversed(cats):
        size = max(1, len(cat.categories))
        parts.append(rest % size)
        rest = rest // size
    return unit, np.stack(list(reversed(parts)), axis=1) if parts else np.zeros((0, 0))


def compare_conditions(adata: Any, condition_key: str, reference: str, *,
                       groupby: str | None = None, sample_key: str | None = None,
                       conditions: list[str] | None = None, groups: list[str] | None = None,
                       min_cells: int = 10, key_added: str = "compare_conditions",
                       prior_count: float = 3.0) -> pd.DataFrame:
    from polariseq._polariseq import compare_cells, compare_pseudobulk

    n = adata.n_obs
    cond = categorical(adata.obs[condition_key])
    if reference not in list(cond.categories):
        raise KeyError(f"{reference!r} is not a value of obs[{condition_key!r}]: "
                       f"{list(cond.categories)}")
    grp = (categorical(adata.obs[groupby]) if groupby is not None
           else pd.Categorical.from_codes(np.zeros(n, dtype=np.int8), categories=["all cells"]))
    keep = np.ones(n, dtype=bool)
    if conditions is not None:
        keep &= np.isin(np.asarray(cond, dtype=object), [reference, *conditions])
    if groups is not None:
        keep &= np.isin(np.asarray(grp, dtype=object), list(groups))
    ref = list(cond.categories).index(reference)
    name = groupby or "group"
    genes = np.asarray(list(map(str, adata.var_names)), dtype=object)
    if sample_key is None:
        cc, gc = _codes(cond), _codes(grp)
        ok = keep & (cc >= 0) & (gc >= 0)
        label = np.where(ok, gc * len(cond.categories) + cc, _MAX).astype(np.uint32)
        rows, infos = compare_cells(adata.X, label, len(grp.categories), len(cond.categories), ref,
                                    min_cells=int(min_cells))
        method = "cells: Welch's t-test on the normalized values"
    else:
        rep = categorical(adata.obs[sample_key])
        unit, parts = _units([grp, cond, rep], keep)
        src = raw_counts(adata)
        rows, infos = compare_pseudobulk(
            src["source"], _per_source_row(src, unit), len(parts),
            parts[:, 0].astype(np.uint32), parts[:, 1].astype(np.uint32),
            parts[:, 2].astype(np.uint32), ref, min_cells=int(min_cells),
            prior_count=float(prior_count), **_file_args(src))
        method = ("pseudobulk: counts summed per replicate, edgeR filterByExpr and TMM, "
                  "limma-trend")
    table = pd.DataFrame({
        name: pd.Categorical.from_codes(rows["group"].astype(np.int64), categories=grp.categories),
        "condition": pd.Categorical.from_codes(rows["condition"].astype(np.int64),
                                               categories=cond.categories),
        "gene": genes[rows["gene"]],
        "log2_fold_change": rows["log_fc"],
        "mean_reference": rows["mean_reference"],
        "mean_condition": rows["mean_condition"],
        "pct_reference": rows["pct_reference"],
        "pct_condition": rows["pct_condition"],
        "t": rows["t"],
        "p_value": rows["p_value"],
        "adj_p_value": rows["adj_p_value"],
    })
    table = table.sort_values([name, "condition", "adj_p_value", "p_value"], kind="stable",
                              ignore_index=True)
    info = pd.DataFrame(infos)
    if len(info):
        info["group"] = [grp.categories[g] for g in info["group"]]
        if "condition" in info:
            info["condition"] = [cond.categories[c] for c in info["condition"]]
        skipped = info[info["note"].notna()]
        if len(skipped):
            import warnings

            warnings.warn("not compared: " + "; ".join(
                f"{r['group']}: {r['note']}" for _, r in skipped.iterrows()), stacklevel=3)
    adata.uns[key_added] = {
        "table": table, "groups": info, "method": method,
        "params": {"condition_key": condition_key, "reference": reference, "groupby": groupby,
                   "sample_key": sample_key, "min_cells": min_cells},
    }
    return table


def pseudobulk(adata: Any, sample_key: str, *, groupby: str | None = None,
               condition_key: str | None = None, min_cells: int = 0) -> Any:
    from polariseq import AnnData, _results
    from polariseq._polariseq import pseudobulk_sums

    keys = [k for k in (groupby, condition_key, sample_key) if k is not None]
    cats = [categorical(adata.obs[k]) for k in keys]
    unit, parts = _units(cats, np.ones(adata.n_obs, dtype=bool))
    src = raw_counts(adata)
    out = pseudobulk_sums(src["source"], _per_source_row(src, unit), len(parts),
                          **_file_args(src))
    kept = np.asarray(out["cells"]) >= min_cells
    obs: dict[str, Any] = {}
    for k, cat, col in zip(keys, cats, parts.T, strict=True):
        obs[k] = pd.Categorical.from_codes(col[kept].astype(np.int64), categories=cat.categories)
    obs["n_cells"] = np.asarray(out["cells"])[kept]
    names = ["_".join(str(obs[k][i]) for k in keys) for i in range(int(kept.sum()))]
    counts = _results.keep(adata, "pseudobulk", "counts", np.asarray(out["sums"])[kept])
    return AnnData(X=counts, obs=obs, obs_names=names, var_names=list(map(str, adata.var_names)))
