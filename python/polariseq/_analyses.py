"""``ps.tl.compare_analyses``: how far analyses of the same cells agree.

Analyses made with different settings (``adata.from_counts``) are compared on
the cells they share: cluster agreement between every pair (adjusted Rand
index, normalized mutual information), which cluster of one matches which of
the other and how far their marker genes overlap, agreement with given labels,
and, optionally, the agreement one analysis has with itself under another
seed, the floor below which a difference between settings means little.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: How a clustering method is recorded in ``uns['leiden']``, and how ``tl.leiden`` is asked for it.
_METHODS = {"leiden": "auto", "fast_label_propagation": "fast", "igraph_leiden": "igraph"}


@dataclass
class AnalysisComparison:
    """What :func:`compare_analyses` measured.

    Attributes:
        key: The ``obs`` column of clusters compared.
        n_cells: Cells every analysis holds, on which everything is measured.
        n_clusters: Clusters of each analysis.
        ari: Adjusted Rand index between every pair (1: the same partition,
            about 0: no more agreement than chance).
        nmi: Normalized mutual information between every pair (0 to 1).
        matches: For each pair ``(a, b)``, each cluster of ``a`` with the
            cluster of ``b`` sharing the most of its cells (Jaccard index of
            the two sets of cells) and, when both have marker genes for
            ``key``, the Jaccard index of their top markers.
        crossing: For each pair, cells per cluster of ``a`` (rows) and of
            ``b`` (columns).
        labels: Agreement of each analysis with the given labels (ARI, NMI),
            when ``labels`` was given.
        noise: Mean ARI of each analysis with itself under other seeds, when
            ``seeds`` was given: differences between settings smaller than
            ``1 - noise`` are within the run-to-run variation.
    """

    key: str
    n_cells: int
    n_clusters: pd.Series
    ari: pd.DataFrame
    nmi: pd.DataFrame
    matches: dict[tuple[str, str], pd.DataFrame] = field(repr=False)
    crossing: dict[tuple[str, str], pd.DataFrame] = field(repr=False)
    labels: pd.DataFrame | None = None
    noise: pd.Series | None = None

    def __repr__(self) -> str:
        lines = [f"Comparison of {len(self.n_clusters)} analyses by obs[{self.key!r}] on "
                 f"{self.n_cells:,} shared cells", "", "clusters:",
                 self.n_clusters.to_string(), "", "adjusted Rand index:", self.ari.round(3).to_string()]
        if self.noise is not None:
            lines += ["", "agreement with itself under other seeds (ARI):", self.noise.round(3).to_string()]
        if self.labels is not None:
            lines += ["", "agreement with the labels:", self.labels.round(3).to_string()]
        return "\n".join(lines)


def _codes(values: Any) -> np.ndarray:
    return pd.factorize(np.asarray(values).astype(str))[0]


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = a.max() + 1, b.max() + 1
    return np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)


def ari(a: Any, b: Any) -> float:
    """Adjusted Rand index of two labelings of the same cells."""
    ct = _contingency(_codes(a), _codes(b))

    def pairs(x: np.ndarray) -> float:
        return float((x * (x - 1) / 2).sum())

    n = ct.sum()
    s, ra, rb = pairs(ct), pairs(ct.sum(1)), pairs(ct.sum(0))
    expected = ra * rb / (n * (n - 1) / 2) if n > 1 else 0.0
    top = 0.5 * (ra + rb) - expected
    return 1.0 if top == 0 else (s - expected) / top


def nmi(a: Any, b: Any) -> float:
    """Normalized mutual information (arithmetic mean of the entropies)."""
    ct = _contingency(_codes(a), _codes(b))
    p = ct / ct.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum())

    def h(x: np.ndarray) -> float:
        x = x[x > 0]
        return float(-(x * np.log(x)).sum())

    denom = (h(pa) + h(pb)) / 2
    return 1.0 if denom == 0 else mi / denom


def _top_markers(adata: Any, key: str, n: int) -> dict[str, set[str]] | None:
    params = adata.uns.get("rank_genes_groups_params") or {}
    rgg = adata.uns.get("rank_genes_groups")
    if params.get("groupby") != key or not isinstance(rgg, dict):
        return None
    return {str(g): {str(x) for x in list(v.get("names", []))[:n]}
            for g, v in rgg.items() if isinstance(v, dict)}


def compare_analyses(analyses: dict[str, Any], key: str = "leiden", *, labels: str | None = None,
                     seeds: int = 0, n_markers: int = 20) -> AnalysisComparison:
    from polariseq import _checks

    if not isinstance(analyses, dict) or len(analyses) < 2:
        raise ValueError("compare_analyses takes two analyses or more, as a dict {name: analysis}")
    names = [str(k) for k in analyses]
    objs = list(analyses.values())
    for name, a in zip(names, objs, strict=True):
        _checks.groups_column(a, key, what=f"compare_analyses ({name!r})", arg="key")
    # the cells every analysis holds, in the first one's order
    common = set(map(str, objs[0].obs_names))
    for a in objs[1:]:
        common &= set(map(str, a.obs_names))
    order = [c for c in map(str, objs[0].obs_names) if c in common]
    if not order:
        raise ValueError("compare_analyses: the analyses share no cells (matched by obs_names)")
    for name, a in zip(names, objs, strict=True):
        if a.n_obs != len(order):
            warnings.warn(f"compare_analyses: {name!r} holds {a.n_obs - len(order):,} cells the "
                          f"others do not; compared on the {len(order):,} shared cells", stacklevel=3)

    def aligned(a: Any, column: str) -> np.ndarray:
        pos = {c: i for i, c in enumerate(map(str, a.obs_names))}
        idx = np.fromiter((pos[c] for c in order), dtype=np.int64, count=len(order))
        return np.asarray(a.obs[column]).astype(str)[idx]

    clusters = {n: aligned(a, key) for n, a in zip(names, objs, strict=True)}
    n_clusters = pd.Series({n: len(np.unique(v)) for n, v in clusters.items()}, name="clusters")
    ari_m = pd.DataFrame(1.0, index=names, columns=names)
    nmi_m = pd.DataFrame(1.0, index=names, columns=names)
    matches, crossing = {}, {}
    markers = {n: _top_markers(a, key, n_markers) for n, a in zip(names, objs, strict=True)}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ari_m.loc[x, y] = ari_m.loc[y, x] = ari(clusters[x], clusters[y])
            nmi_m.loc[x, y] = nmi_m.loc[y, x] = nmi(clusters[x], clusters[y])
            ct = pd.crosstab(pd.Series(clusters[x], name=x), pd.Series(clusters[y], name=y))
            crossing[(x, y)] = ct
            sx, sy = ct.sum(1), ct.sum(0)
            jac = ct / (sx.to_numpy()[:, None] + sy.to_numpy()[None, :] - ct)
            best = jac.idxmax(axis=1)
            rows = pd.DataFrame({
                "cells": sx, "match": best, "match_cells": sy[best].to_numpy(),
                "shared_cells": [int(ct.loc[c, best[c]]) for c in ct.index],
                "jaccard": [float(jac.loc[c, best[c]]) for c in ct.index]})
            if markers[x] is not None and markers[y] is not None:
                rows["markers_jaccard"] = [
                    len(markers[x].get(c, set()) & markers[y].get(best[c], set()))
                    / max(1, len(markers[x].get(c, set()) | markers[y].get(best[c], set())))
                    for c in ct.index]
            rows.index.name = f"{x} cluster"
            matches[(x, y)] = rows.sort_values("cells", ascending=False)
    label_table = None
    if labels is not None:
        holder = next((a for a in objs if labels in a.obs), None)
        if holder is None:
            _checks.groups_column(objs[0], labels, what="compare_analyses", arg="labels")
        truth = aligned(holder, labels)
        label_table = pd.DataFrame({"ari": [ari(clusters[n], truth) for n in names],
                                    "nmi": [nmi(clusters[n], truth) for n in names],
                                    "clusters": n_clusters.to_numpy()}, index=names)
    noise = _noise(names, objs, key, seeds) if seeds else None
    return AnalysisComparison(key=key, n_cells=len(order), n_clusters=n_clusters, ari=ari_m,
                              nmi=nmi_m, matches=matches, crossing=crossing, labels=label_table,
                              noise=noise)


def _noise(names: list[str], objs: list[Any], key: str, seeds: int) -> pd.Series:
    """Mean ARI of each analysis's clusters with the same clustering under other seeds."""
    from polariseq import tl

    out = {}
    for name, a in zip(names, objs, strict=True):
        rec = a.uns.get("leiden") or {}
        if (rec.get("params") or {}).get("key_added", "leiden") != key or "connectivities" not in a.obsp:
            warnings.warn(f"compare_analyses: obs[{key!r}] of {name!r} is not its latest Leiden "
                          "clustering (or its graph is gone); no seed agreement for it", stacklevel=4)
            out[name] = np.nan
            continue
        base = np.asarray(a.obs[key])
        seed0 = int((rec.get("params") or {}).get("seed", 0))
        tmp = "_compare_analyses_seed"
        saved = dict(rec)
        derived = set(getattr(a, "_derived", None) or ())
        scores = []
        try:
            for s in range(1, seeds + 1):
                tl.leiden(a, resolution=rec.get("resolution", 1.0), seed=seed0 + s, key_added=tmp,
                          method=_METHODS.get(rec.get("method"), "auto"))
                scores.append(ari(base, np.asarray(a.obs[tmp])))
        finally:
            a.obs.pop(tmp, None)
            a.uns["leiden"] = saved
            a._derived = derived
        out[name] = float(np.mean(scores))
    return pd.Series(out, name="ari_between_seeds")
