"""Cell types named from marker genes (:meth:`tl.auto_annotate`).

The scoring follows two published methods: ScType written from its paper (its
code is under the GPL and was not used), the Cell Marker Accordion's weights
following its MIT-licensed code (see THIRD_PARTY_NOTICES.md):

- ScType (Ianevski et al., Nat. Commun. 13, 1246, 2022): each marker gene is
  standardized over the cells; a cell's score for a type is the sum of its
  standardized positive markers over the square root of their number, minus
  the same for the negative markers; a cluster takes the type with the highest
  score summed over its cells, and none when that score is too low.
- Cell Marker Accordion (Busarello et al., Nat. Commun. 16, 11, 2025): each
  marker is weighted by its evidence consistency (how many sources name it,
  log10 + 1) times its specificity (how few types share it, rescaled to the
  evidence range, log10 + 1); a type keeps its best-weighted markers (30 by
  default); the sums are taken over these weights and their square root; a
  cluster stays unassigned when its 75th percentile score for its best type
  is not above zero (under a quarter of its cells score above zero).

Scores are computed in the Rust core in two streamed passes over the cells
(``marker_scores``), for every gene of the source files when they can be read,
so markers left out of the variable-gene selection still count. On top of the
two methods: a cluster whose two best types score within ``margin`` of each
other is named by their nearest shared Cell Ontology type (two kinds of T cell:
"T cell"), each cell gets its own best type as well, and a cluster holding a
group of cells that score highest for another type says so (``also``), the way
a rare population inside a larger cluster shows.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from polariseq import _markers


def _scoring_matrix(adata: Any) -> tuple[str, Any, list[str], str]:
    """The values to score ``adata``'s cells on: ``(kind, matrix, gene names, how)``.

    ``kind`` is ``"view"`` (an out-of-core view of every gene, log-normalized),
    ``"counts"`` (the raw counts of every gene in memory, normalized later on the
    marker columns only) or ``"matrix"`` (the current matrix as it is, when the
    raw counts cannot be read again)."""
    from polariseq import (
        LazyAnnData,
        SparseMatrix,
        _all_gene_names,
        _is_disk,
        _is_sparse,
    )

    X = adata._X if isinstance(adata, LazyAnnData) else adata.X
    src = getattr(adata, "_source", None)
    current = list(map(str, adata.var_names))
    if _is_disk(X):
        n_file_genes = X.file_shape[1]
        names = current
        col_keep = np.asarray(X.col_mask(), dtype=bool)
        if src is not None:
            try:
                every = _all_gene_names(adata)
                if len(every) == n_file_genes:
                    names, col_keep = every, np.ones(n_file_genes, dtype=bool)
            except Exception:  # noqa: BLE001  (files moved: the current genes still serve)
                pass
        view = X.subset(None, None)
        view.restore_view(np.ascontiguousarray(X.row_mask(), dtype=bool),
                          np.ascontiguousarray(col_keep), None, False)
        view.normalize_total(1e4)
        view.log1p()
        how = (f"the raw counts of all {len(names):,} genes, normalized per cell"
               if len(names) > len(current) else "the raw counts of the current genes")
        return "view", view, names, how

    raw, names = None, current
    if src is not None and all(str(p).endswith(".h5ad") for p in src["paths"]):
        from polariseq._state import _read_source, _replay

        try:
            paths = [os.fspath(p) for p in src["paths"]]
            full = _read_source(src, paths)
            if full is not None:
                raw = _replay(full, getattr(adata, "_rows", None), None, [])
                names = _all_gene_names(adata)
        except Exception:  # noqa: BLE001  (moved or unreadable: fall back to the matrix held)
            raw, names = None, current
    if raw is None and "normalize" not in adata.uns and "log1p" not in adata.uns:
        raw = X  # still counts: no step has normalized them
    if raw is None:
        logged = "log1p" in adata.uns
        how = ("the current matrix as it is" + ("" if logged else " (not log-transformed)")
               + f", {len(current):,} genes: markers outside them are not used")
        if not (_is_sparse(X) or _is_disk(X)):
            X = SparseMatrix.from_dense(np.ascontiguousarray(X, dtype=np.float32))
        return "matrix", X, current, how
    if not _is_sparse(raw):
        raw = SparseMatrix.from_dense(np.ascontiguousarray(raw, dtype=np.float32))
    return "counts", raw, names, (f"the raw counts of all {len(names):,} genes, normalized per cell"
                                if len(names) > len(current)
                                else "the raw counts of the current genes, normalized per cell")


def _log_normalized_columns(m: Any, cols: np.ndarray) -> Any:
    """Columns ``cols`` of the counts ``m`` (a :class:`SparseMatrix`), each cell scaled
    to 10,000 counts over all its genes, then ``log1p``."""
    from polariseq import SparseMatrix

    totals = np.asarray(m.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    data, indices, indptr = m[:, np.asarray(cols, dtype=np.int64)].to_arrays()
    rows = np.repeat(np.arange(totals.size), np.diff(np.asarray(indptr, dtype=np.int64)))
    data = np.log1p(np.asarray(data, dtype=np.float64) * scale[rows]).astype(np.float32)
    return SparseMatrix.from_arrays(data, indices, indptr, totals.size, len(cols))


def _rescale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``x`` mapped linearly onto ``[lo, hi]`` (the middle when ``x`` is constant)."""
    a, b = float(np.min(x)), float(np.max(x))
    if b - a < 1e-12:
        return np.full(x.shape, (lo + hi) / 2.0)
    return lo + (x - a) * (hi - lo) / (b - a)


def weigh(t: pd.DataFrame, max_markers: int | None) -> pd.DataFrame:
    """The Cell Marker Accordion weights of a matched marker table.

    ``t`` has one row per (type, gene, sign) with ``evidence``; adds ``ecs``,
    ``sps`` (types sharing the gene), ``weight`` (specificity x evidence) and
    keeps each type's ``max_markers`` best-weighted markers."""
    t = t.copy()
    t["ecs"] = np.log10(t["evidence"].astype(float)) + 1.0
    t["sps"] = t.groupby(["gene", "sign"])["type"].transform("nunique")
    t["sps_reg"] = 0.0
    for sign in (1, -1):
        rows = t["sign"] == sign
        if not rows.any():
            continue
        n = t.loc[rows, "type"].nunique()
        flipped = n + 1.0 - t.loc[rows, "sps"].to_numpy(dtype=float)  # 1 type -> n, n types -> 1
        ev = t.loc[rows, "evidence"].to_numpy(dtype=float)
        t.loc[rows, "sps_reg"] = np.log10(_rescale(flipped, ev.min(), ev.max())) + 1.0
    t["weight"] = t["sps_reg"] * t["ecs"]
    if max_markers:
        t = (t.sort_values(["type", "weight", "strength", "gene"],
                           ascending=[True, False, False, True])
             .groupby("type", sort=False).head(int(max_markers)))
    return t.reset_index(drop=True)


def auto_annotate(adata: Any, groupby: str, *, source: Any, tissue: Any, species: str | None,
                  markers: Any, max_markers: int, margin: float, key_added: str,
                  unassigned: str) -> pd.DataFrame:
    """See :meth:`polariseq._Tools.auto_annotate`."""
    from polariseq import _checks, _derived
    from polariseq._polariseq import marker_scores

    _checks.groups_column(adata, groupby, what="auto_annotate")
    if max_markers is not None and int(max_markers) < 1:
        raise _checks.InputError(f"auto_annotate: max_markers must be 1 or more (or None: all), "
                                 f"not {max_markers!r}")
    if not 0 <= float(margin) < 1:
        raise _checks.InputError(f"auto_annotate: margin must be in [0, 1), not {margin!r}")

    kind, matrix, names, how = _scoring_matrix(adata)
    if species is None:
        species = _markers.guess_species(names)
    table = _markers.marker_table(source, tissue=tissue, species=species, markers=markers)

    # the table's genes in the matrix: exact names, else the same letters in another case
    where = {g: i for i, g in enumerate(names)}
    upper: dict[str, int] = {}
    for g, i in where.items():
        upper.setdefault(g.upper(), i)
    col = table["gene"].map(lambda g: where.get(g, upper.get(str(g).upper(), -1)))
    found = table[col.to_numpy() >= 0].assign(col=col[col >= 0].astype(np.int64))
    if found.empty:
        raise _checks.InputError(
            f"auto_annotate: none of the {table['gene'].nunique():,} marker genes of "
            f"{_checks.listing(_sources(source, markers))} is a gene of this data (for example "
            f"{_checks.listing(table['gene'].drop_duplicates().head(5))}; the data's genes look "
            f"like {_checks.listing(names[:5])}). Is species={species!r} right? Gene symbols are "
            "needed, not Ensembl ids.")

    # types merge when their names are the same Cell Ontology term (B cells, B cell); a term
    # reached by dropping words from a longer name ("Salivary basal cells": basal cell) names
    # the type's lineage, but keeps it apart from others of that lineage
    names_ = found["cell_type"].drop_duplicates()
    exact = found["cell_type"].map(dict(zip(names_, names_.map(_markers.exact_cl), strict=True)))
    merge = (found["cl_id"] != "") & (exact == found["cl_id"])
    found = found.assign(type=np.where(merge, found["cl_id"],
                                       "name:" + found["cell_type"].str.lower()))
    display = found.drop_duplicates("type").set_index("type")[["cell_type", "cl_id"]]
    signed = found.assign(net=found["sign"] * found["evidence"])
    agg = (signed.groupby(["type", "col"], as_index=False)
           .agg(net=("net", "sum"), strength=("strength", "max"), gene=("gene", "first")))
    agg = agg[agg["net"] != 0]
    agg = agg.assign(sign=np.sign(agg["net"]).astype(int), evidence=agg["net"].abs())
    used = weigh(agg, max_markers)

    types = list(dict.fromkeys(used["type"]))
    t_index = {t: i for i, t in enumerate(types)}
    genes = np.array(sorted(used["col"].unique()), dtype=np.int64)
    g_index = {int(g): i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), len(types)), dtype=np.float64)
    for (tp, sign), part in used.groupby(["type", "sign"]):
        w = part["weight"].to_numpy(dtype=float)
        rows = [g_index[int(c)] for c in part["col"]]
        W[rows, t_index[tp]] = sign * w / np.sqrt(w.sum())
    used = used.assign(score_weight=[W[g_index[int(c)], t_index[t]]
                                     for c, t in zip(used["col"], used["type"], strict=True)])

    # the clusters
    cat = _clusters(adata.obs[groupby])
    codes = np.asarray(cat.codes, dtype=np.int64)
    n_groups = len(cat.categories)
    labels = np.where(codes < 0, n_groups, codes).astype(np.uint32)

    if kind == "counts":  # only the marker columns are kept and normalized
        src, cols = _log_normalized_columns(matrix, genes), np.arange(len(genes), dtype=np.uint32)
        del matrix
    else:
        src, cols = matrix, genes.astype(np.uint32)
    out = marker_scores(src, cols, np.ascontiguousarray(W.ravel(), dtype=np.float32),
                        len(types), labels, n_groups)

    counts = np.asarray(out["group_counts"], dtype=np.int64)
    positive = np.asarray(out["group_positive"], dtype=np.int64).reshape(n_groups, len(types))
    sums = np.asarray(out["group_sums"], dtype=np.float64).reshape(n_groups, len(types))
    means = sums / np.maximum(counts, 1)[:, None]
    best_cell = np.asarray(out["best"], dtype=np.int64)
    best_cell_score = np.asarray(out["best_score"], dtype=np.float32)

    names_of = [str(display.loc[t, "cell_type"]) for t in types]
    cl_of = [str(display.loc[t, "cl_id"]) for t in types]
    rows, cluster_label, cluster_cl, cluster_score = [], {}, {}, {}
    min_also = 10
    min_support = 0.25  # the Accordion's test: the cluster's 75th percentile score above zero
    for g, cluster in enumerate(cat.categories):
        n = int(counts[g])
        order = np.argsort(-means[g], kind="stable")
        b, s2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        in_cluster = labels == g
        # cells scoring clearly for a type: at least half what the cluster scores for its own
        strong = best_cell_score[in_cluster] >= max(0.5 * float(means[g, b]), 1.0)
        cell_best = best_cell[in_cluster][strong]
        support = float(positive[g, b]) / n if n else 0.0
        label, cl, note = names_of[b], cl_of[b], ""
        score, runner = float(means[g, b]), float(means[g, s2])
        close = len(order) > 1 and margin > 0 and score > 0 and runner >= (1 - margin) * score
        if close:
            shared = _markers.common_ancestor(cl_of[b], cl_of[s2])
            if shared and shared not in (cl_of[b], cl_of[s2]):
                label, cl = _markers.cl_name(shared), shared
                note = f"between {names_of[b]} and {names_of[s2]}"
            elif shared:  # one is the other's ancestor: the more specific holds
                note = f"close to {names_of[s2]}"
            else:
                note = f"close to {names_of[s2]}"
        if n == 0:
            label, cl, note = unassigned, "", "no cells"
        elif score <= 0:
            label, cl, note = unassigned, "", f"best: {names_of[b]} ({score:.2f})"
        elif support < min_support:
            label, cl, note = unassigned, "", (f"best: {names_of[b]}, but only {support:.0%} of "
                                               "the cells score above zero for it")
        # groups of cells inside the cluster that score highest for another type
        also = []
        if n:
            ids, k = np.unique(cell_best, return_counts=True)
            for i, c in sorted(zip(ids, k, strict=True), key=lambda x: -x[1]):
                if int(i) != b and c >= max(min_also, 0.02 * n) and len(also) < 3:
                    also.append(f"{names_of[int(i)]} ({int(c)})")
        rows.append({"cluster": cluster, "n_cells": n, "cell_type": label, "ontology_id": cl,
                     "score": round(score, 4), "runner_up": names_of[s2],
                     "runner_up_score": round(runner, 4), "support": round(support, 4),
                     "note": note, "also": ", ".join(also)})
        cluster_label[g], cluster_cl[g], cluster_score[g] = label, cl, score

    table_out = pd.DataFrame(rows)
    valid = labels < n_groups
    lab = np.full(labels.size, unassigned, dtype=object)
    lab[valid] = [cluster_label[int(g)] for g in labels[valid]]
    levels = list(dict.fromkeys([*table_out["cell_type"]]))
    adata.obs[key_added] = pd.Categorical(lab, categories=levels if unassigned in levels
                                          else [*levels, unassigned])
    ont = np.full(labels.size, "", dtype=object)
    ont[valid] = [cluster_cl[int(g)] for g in labels[valid]]
    adata.obs[f"{key_added}_ontology_id"] = ont
    sc = np.full(labels.size, np.nan, dtype=np.float32)
    sc[valid] = [cluster_score[int(g)] for g in labels[valid]]
    adata.obs[f"{key_added}_score"] = sc
    per_cell = np.array(names_of, dtype=object)[best_cell]
    per_cell[best_cell_score <= 0] = unassigned
    adata.obs[f"{key_added}_cell"] = pd.Categorical(per_cell)
    adata.obs[f"{key_added}_cell_score"] = best_cell_score
    for k in (key_added, f"{key_added}_ontology_id", f"{key_added}_score", f"{key_added}_cell",
              f"{key_added}_cell_score"):
        _derived(adata, k)

    marker_frame = used.assign(cell_type=[names_of[t_index[t]] for t in used["type"]])[
        ["cell_type", "gene", "sign", "evidence", "sps", "weight", "score_weight"]]
    adata.uns["annotation"] = {
        "table": table_out,
        "scores": pd.DataFrame(means, index=[str(c) for c in cat.categories], columns=names_of),
        "markers": marker_frame.reset_index(drop=True),
        "params": {"groupby": groupby, "source": _sources(source, markers),
                   "tissue": tissue if tissue is None or isinstance(tissue, str) else list(tissue),
                   "species": species, "max_markers": max_markers, "margin": margin,
                   "key_added": key_added, "matrix": how, "n_types": len(types),
                   "n_genes": len(genes)},
    }
    return table_out


def _clusters(values: Any) -> pd.Categorical:
    """The cluster labels as a categorical: a categorical's own order, else numbers
    (cluster "2" before "10") numerically, then text."""
    if isinstance(values, pd.Categorical) or isinstance(
            getattr(values, "dtype", None), pd.CategoricalDtype):
        return pd.Categorical(values)
    vals = np.asarray(values)

    def key(g: Any) -> tuple:
        try:
            return (0, float(g), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(g))
    return pd.Categorical(vals, categories=sorted(set(vals.tolist()), key=key))


def _sources(source: Any, markers: Any) -> list[str]:
    names = [] if source is None else [source] if isinstance(source, str) else list(source)
    return [_markers.source_name(s) for s in names] + (["markers"] if markers is not None else [])
