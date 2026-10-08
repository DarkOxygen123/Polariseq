"""Curating clusters into cell types: name them, join them, split them, and read their markers.

A clustering is where an annotation starts, not where it ends. A researcher reads each cluster's top
genes, joins clusters that are one cell type, splits a cluster that holds two, and names the result.
These functions do each step on a label column of ``obs`` and nothing else: they never touch the
matrix, so they cost the same in memory and out of core, and every result is a new categorical column
(the original clustering is kept). Each step is recorded in ``uns['curation']``, so an annotation can
be read back and repeated.

    ps.tl.rank_genes_groups(ad, "leiden")
    ps.get.markers(ad, n=10)                                   # what each cluster expresses
    ps.tl.merge_clusters(ad, [["3", "7"]], key_added="curated")
    ps.tl.subcluster(ad, "5", groupby="curated", resolution=0.5)   # 5 becomes 5.0, 5.1, ...
    ps.tl.annotate(ad, {"0": "CD4 T cell", "3+7": "B cell", "5.0": "NK cell"}, groupby="curated")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _categorical(adata: Any, groupby: str) -> pd.Categorical:
    """``obs[groupby]`` as a categorical with string categories, without a Python object per cell."""
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} is not a column of obs; available: {sorted(adata.obs)}")
    col = adata.obs[groupby]
    if isinstance(col, pd.Series):
        col = col.array
    cat = col if isinstance(col, pd.Categorical) else pd.Categorical(np.asarray(col))
    return cat.rename_categories([str(c) for c in cat.categories])


def _record(adata: Any, step: dict) -> None:
    adata.uns.setdefault("curation", []).append(step)


def _store(adata: Any, key: str, codes: np.ndarray, categories: list[str]) -> None:
    """Write a label column, dropping categories no cell carries."""
    cat = pd.Categorical.from_codes(np.asarray(codes, dtype=np.int64), categories=categories)
    adata.obs[key] = cat.remove_unused_categories()


def _check_groups(cat: pd.Categorical, wanted: list[str], groupby: str) -> None:
    unknown = [g for g in wanted if g not in set(cat.categories)]
    if unknown:
        raise KeyError(f"{unknown} not in obs[{groupby!r}]; available: {list(cat.categories)}")


def annotate(adata: Any, mapping: dict, groupby: str = "leiden", *, key_added: str = "cell_type",
             unassigned: str | None = None) -> None:
    """Name clusters: ``obs[key_added]`` holds a name for every cell.

    ``mapping`` maps clusters to names, ``{"0": "CD4 T cell", "4": "B cell"}``, or names to lists of
    clusters, ``{"B cell": ["4", "9"]}``. Clusters given the same name become one group, which is the
    simplest way to join clusters that are one cell type. A cluster left out of ``mapping`` keeps its
    own label, or takes ``unassigned`` when that is given (``unassigned="unknown"``).

    Args:
        adata: The analysis (in memory or out of core).
        mapping: ``{cluster: name}`` or ``{name: [clusters]}``.
        groupby: The label column the clusters come from.
        key_added: The column the names are written to.
        unassigned: The name of clusters left out of ``mapping`` (default: their own label).
    """
    cat = _categorical(adata, groupby)
    pairs: dict[str, str] = {}
    for k, v in mapping.items():
        if isinstance(v, (list, tuple, set, np.ndarray)):
            for c in v:
                pairs[str(c)] = str(k)
        else:
            pairs[str(k)] = str(v)
    _check_groups(cat, list(pairs), groupby)
    names = [pairs.get(c, unassigned if unassigned is not None else c) for c in cat.categories]
    categories = list(dict.fromkeys([pairs[c] for c in cat.categories if c in pairs] + names))
    remap = np.array([categories.index(n) for n in names] + [-1], dtype=np.int64)
    codes = np.asarray(cat.codes, dtype=np.int64)
    _store(adata, key_added, remap[codes], categories)
    _record(adata, {"step": "annotate", "groupby": groupby, "key_added": key_added,
                    "mapping": pairs, "unassigned": unassigned})


def merge_clusters(adata: Any, groups: list, groupby: str = "leiden", *, key_added: str | None = None,
                   names: list[str] | None = None) -> None:
    """Join clusters into one: ``merge_clusters(ad, [["3", "7"], ["10", "12", "15"]])``.

    Each inner list becomes one cluster, named ``"3+7"`` (its members joined by ``+``) or by the
    matching entry of ``names``. Every other cluster keeps its label. The result goes to
    ``obs[key_added]`` (default ``f"{groupby}_curated"``; pass ``key_added=groupby`` to replace it).

    Args:
        adata: The analysis (in memory or out of core).
        groups: The clusters to join, as a list of lists (a single list is one group).
        groupby: The label column the clusters come from.
        key_added: The column written.
        names: A name for each joined cluster.
    """
    cat = _categorical(adata, groupby)
    if groups and not isinstance(groups[0], (list, tuple, set, np.ndarray)):
        groups = [groups]
    groups = [[str(c) for c in g] for g in groups]
    flat = [c for g in groups for c in g]
    _check_groups(cat, flat, groupby)
    dup = sorted({c for c in flat if flat.count(c) > 1})
    if dup:
        raise ValueError(f"clusters {dup} appear in more than one group")
    if any(len(g) < 2 for g in groups):
        raise ValueError("each group to join needs at least two clusters")
    if names is not None and len(names) != len(groups):
        raise ValueError(f"{len(names)} names for {len(groups)} groups")
    new_name = {}
    for i, g in enumerate(groups):
        joined = str(names[i]) if names is not None else "+".join(g)
        for c in g:
            new_name[c] = joined
    labels = [new_name.get(c, c) for c in cat.categories]
    clash = sorted(set(new_name.values()) & {c for c in cat.categories if c not in new_name})
    if clash:
        raise ValueError(f"the new names {clash} are already clusters of obs[{groupby!r}]")
    categories = list(dict.fromkeys(labels))
    remap = np.array([categories.index(n) for n in labels] + [-1], dtype=np.int64)
    key = key_added or f"{groupby}_curated"
    _store(adata, key, remap[np.asarray(cat.codes, dtype=np.int64)], categories)
    _record(adata, {"step": "merge_clusters", "groupby": groupby, "key_added": key,
                    "groups": groups, "names": list(names) if names is not None else None})


def subcluster(adata: Any, cluster: Any, groupby: str = "leiden", *, resolution: float = 1.0,
               seed: int = 0, key_added: str | None = None) -> int:
    """Split one cluster: Leiden on the neighbour graph between that cluster's cells only.

    The cluster ``"5"`` becomes ``"5.0"``, ``"5.1"``, ... (largest first); every other cluster keeps
    its label and every other cell its assignment. Only the graph is read (``obsp['connectivities']``,
    from ``pp.neighbors``), so this works the same out of core. Returns the number of parts.

    Args:
        adata: The analysis, after ``pp.neighbors``.
        cluster: The cluster to split.
        groupby: The label column it belongs to.
        resolution: Leiden resolution within the cluster (higher: more parts).
        seed: Random seed.
        key_added: The column written (default ``f"{groupby}_curated"``; ``groupby`` to replace it).
    """
    from scipy import sparse

    from polariseq._polariseq import leiden_cluster

    if "connectivities" not in adata.obsp:
        raise KeyError("subcluster needs obsp['connectivities']; run ps.pp.neighbors first")
    cat = _categorical(adata, groupby)
    target = str(cluster)
    _check_groups(cat, [target], groupby)
    codes = np.asarray(cat.codes, dtype=np.int64)
    t = list(cat.categories).index(target)
    idx = np.flatnonzero(codes == t)
    if idx.size < 3:
        raise ValueError(f"cluster {target!r} has {idx.size} cells; too few to split")
    conn = adata.obsp["connectivities"]
    if not sparse.isspmatrix_csr(conn) and not isinstance(conn, sparse.csr_array):
        conn = sparse.csr_matrix(conn)
    sub = conn[idx][:, idx].tocsr()
    sub.sort_indices()
    res = leiden_cluster(np.asarray(sub.indptr, dtype=np.uint32), np.asarray(sub.indices, dtype=np.uint32),
                         np.asarray(sub.data, dtype=np.float32), int(idx.size), float(resolution), int(seed))
    part = np.asarray(res["labels"], dtype=np.int64)
    sizes = np.bincount(part)
    order = np.argsort(-sizes, kind="stable")  # parts named by decreasing size
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    parts = [f"{target}.{k}" for k in range(order.size)]
    clash = sorted(set(parts) & set(cat.categories))
    if clash:
        raise ValueError(f"the new labels {clash} are already clusters of obs[{groupby!r}]")
    before = list(cat.categories[:t])
    categories = before + parts + list(cat.categories[t + 1:])
    shift = np.array(list(range(t)) + [-2] + [t + len(parts) + i for i in range(len(cat.categories) - t - 1)]
                     + [-1], dtype=np.int64)
    new = shift[codes]
    new[idx] = t + rank[part]
    key = key_added or f"{groupby}_curated"
    _store(adata, key, new, categories)
    _record(adata, {"step": "subcluster", "groupby": groupby, "key_added": key, "cluster": target,
                    "resolution": float(resolution), "seed": int(seed), "parts": len(parts)})
    return len(parts)


def markers(adata: Any, n: int = 10, *, groups: list | None = None, wide: bool = False) -> pd.DataFrame:
    """The top marker genes of each cluster, from ``tl.rank_genes_groups``, as a table.

    Long form (default): one row per cluster and gene, with the rank, the score (Welch's t), the log
    fold change, the adjusted p-value and the fraction of the cluster's cells expressing the gene.
    ``wide=True``: one column per cluster listing its top ``n`` genes, the table to read when naming
    clusters.

    Args:
        adata: The analysis, after ``tl.rank_genes_groups``.
        n: Genes per cluster.
        groups: Clusters to include (default: all ranked).
        wide: One column of gene names per cluster instead of the long table.
    """
    res = adata.uns.get("rank_genes_groups")
    if not res:
        raise KeyError("no markers stored; run ps.tl.rank_genes_groups first")
    keys = list(res) if groups is None else [str(g) for g in groups]
    unknown = [k for k in keys if k not in res]
    if unknown:
        raise KeyError(f"{unknown} were not ranked; ranked: {list(res)}")
    if wide:
        return pd.DataFrame({k: pd.Series(list(res[k]["names"][:n])) for k in keys})
    rows = []
    for k in keys:
        r = res[k]
        m = min(n, len(r["names"]))
        for i in range(m):
            rows.append({"group": k, "rank": i + 1, "gene": r["names"][i], "score": float(r["scores"][i]),
                         "logfoldchange": float(r["logfoldchanges"][i]),
                         "pval_adj": float(r["pvals_adj"][i]), "pct": float(r["pts"][i])})
    return pd.DataFrame(rows, columns=["group", "rank", "gene", "score", "logfoldchange", "pval_adj", "pct"])
