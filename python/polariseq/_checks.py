"""Checks of what a step is given, with errors that say what is missing, why, and what to do.

Every public step checks its inputs here before it reads any data, so a wrong
column or gene name fails at once, naming the closest match, the names that
exist, and the step that makes a missing result.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Any

import numpy as np


class InputError(KeyError):
    """A name a step was given is not in the data (a column, gene, embedding or
    result). A ``KeyError``, so code that catches one keeps working; printed as
    the plain message."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def suggest(name: Any, choices: Iterable[Any]) -> str:
    """`` Did you mean 'X'?`` for the closest of ``choices`` (a different case
    first), or an empty string."""
    choices = [str(c) for c in choices]
    name = str(name)
    same = [c for c in choices if c.lower() == name.lower()]
    close = same or difflib.get_close_matches(name, choices, n=3, cutoff=0.6)
    if not close:
        return ""
    return " Did you mean " + " or ".join(repr(c) for c in close[:3]) + "?"


def listing(names: Iterable[Any], limit: int = 12) -> str:
    names = [str(n) for n in names]
    if not names:
        return "none"
    shown = ", ".join(repr(n) for n in names[:limit])
    return shown + (f", ... ({len(names)} in all)" if len(names) > limit else "")


def column(adata: Any, key: str, *, what: str, axis: str = "obs") -> None:
    """``key`` is a column of ``adata.obs`` (or ``var``), else an error naming
    the columns there are."""
    frame = getattr(adata, axis)
    if key in frame:
        return
    raise InputError(f"{what}: no column {key!r} in adata.{axis}.{suggest(key, frame)} "
                     f"Columns: {listing(frame)}.")


def result(adata: Any, key: str, *, what: str, where: str, made_by: str) -> None:
    """A step's input ``adata.<where>[key]`` exists, else an error naming the
    step that makes it."""
    held = getattr(adata, where)
    if key in held:
        return
    others = f" Present in adata.{where}: {listing(held)}." if len(held) else ""
    made = (f"which {made_by} makes; run it first." if made_by
            else "which is not there.")
    raise InputError(f"{what} needs adata.{where}[{key!r}], {made}{suggest(key, held)}{others}")


def gene_positions(var_names: Any, genes: Any, *, what: str,
                   source_names: Iterable[Any] | None = None) -> np.ndarray:
    """Positions of ``genes`` among ``var_names``, in the order given, each
    once. A name that is not there is an error that says, for each, whether it
    is absent from the data or was removed by an earlier step, and the closest
    names."""
    if isinstance(genes, (str, bytes)):
        genes = [genes]
    asked = list(dict.fromkeys(str(g) for g in genes))
    if not asked:
        raise ValueError(f"{what}: no genes given")
    index: dict[str, list[int]] = {}
    for i, n in enumerate(var_names):
        index.setdefault(str(n), []).append(i)
    twice = [g for g in asked if len(index.get(g, ())) > 1]
    if twice:
        raise ValueError(f"{what}: the gene name(s) {listing(twice)} appear more than once in "
                         "var_names, so they cannot be told apart; make the names unique first")
    missing = [g for g in asked if g not in index]
    if missing:
        source = None if source_names is None else {str(n) for n in source_names}
        lines = []
        for g in missing:
            if source is not None and g in source:
                lines.append(f"  {g}: in the data, but removed by an earlier step (the gene filter "
                             "or a gene selection with subset=True)")
            else:
                lines.append(f"  {g}: not in the data.{suggest(g, index)}")
        raise InputError(f"{what}: {len(missing)} of the {len(asked)} genes asked for are not "
                         f"among this analysis's {len(index):,} genes:\n" + "\n".join(lines)
                         + "\nGenes are matched by their exact name in var_names (for example "
                         f"{listing(list(index)[:3])}).")
    return np.array([index[g][0] for g in asked], dtype=np.int64)


#: The step that makes each result a later step reads.
MADE_BY = {
    "X_pca": "ps.pp.pca", "X_pca_harmony": "ps.pp.harmony_integrate", "X_umap": "ps.tl.umap",
    "connectivities": "ps.pp.neighbors", "distances": "ps.pp.neighbors",
    "leiden": "ps.tl.leiden", "louvain": "ps.tl.leiden(method='fast')",
    "rank_genes_groups": "ps.tl.rank_genes_groups", "neighbors": "ps.pp.neighbors",
}


def made_by(key: str) -> str:
    """The step that makes ``key``, or an empty string for a name no step makes."""
    return MADE_BY.get(key, "")


def graph(adata: Any, *, what: str) -> None:
    """The neighbour graph exists, else an error saying to build it."""
    if "connectivities" in adata.obsp or "distances" in adata.obsp:
        return
    raise InputError(f"{what} needs the neighbour graph (adata.obsp['connectivities']), which "
                     "ps.pp.neighbors makes; run it first.")


def groups_column(adata: Any, key: str, *, what: str, arg: str = "groupby") -> None:
    """``key`` is an ``obs`` column to group cells by; a missing clustering
    names the step that makes it."""
    if key in adata.obs:
        return
    hint = f" Run {MADE_BY[key]} first, or pass another {arg}." if key in MADE_BY else ""
    raise InputError(f"{what}: {arg}={key!r} is not a column of adata.obs.{hint}"
                     f"{suggest(key, adata.obs)} Columns: {listing(adata.obs)}.")


def values_in(asked: Iterable[Any], present: Iterable[Any], *, what: str, column: str) -> None:
    """Every value asked for occurs in the column."""
    present = [str(p) for p in present]
    missing = [str(a) for a in asked if str(a) not in set(present)]
    if missing:
        raise InputError(f"{what}: {listing(missing)} not found in {column}.{suggest(missing[0], present)} "
                         f"Values there: {listing(present)}.")


def positive(name: str, value: Any, *, what: str, integer: bool = False) -> None:
    ok = value is None or (value > 0 and (not integer or float(value).is_integer()))
    if not ok:
        kind = "a positive whole number" if integer else "a positive number"
        raise ValueError(f"{what}: {name} must be {kind}, not {value!r}")


_HDF5 = b"\x89HDF\r\n\x1a\n"


def input_file(path: Any, *, what: str, kind: str = "h5ad") -> str:
    """``path`` as a string, checked to be a readable file of the kind a reader
    takes; else an error saying what is there instead and which reader fits."""
    import os

    p = os.fspath(path)
    if kind == "mtx":
        if os.path.isdir(p):
            return p
        if os.path.exists(p):
            raise ValueError(f"{what}: {p} is a file; read_10x_mtx takes the folder holding "
                             "matrix.mtx(.gz), barcodes.tsv(.gz) and features.tsv(.gz)")
        raise FileNotFoundError(f"{what}: no folder {p}{_near(p)}")
    if os.path.isdir(p):
        if any(os.path.exists(os.path.join(p, f)) for f in ("matrix.mtx", "matrix.mtx.gz")):
            raise ValueError(f"{what}: {p} is a 10x MatrixMarket folder; read it with "
                             "ps.read_10x_mtx")
        raise ValueError(f"{what}: {p} is a folder; give the path of a .{kind} file in it")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{what}: no file {p}{_near(p)}")
    with open(p, "rb") as fh:
        head = fh.read(8)
    if head != _HDF5:
        known = {".loom": "a loom file: convert it with anndata.read_loom(...).write_h5ad(...)",
                 ".csv": "a table: read it with anndata.read_csv(...) and write an .h5ad",
                 ".rds": "an R object: export it from R (for example with zellkonverter or "
                         "SeuratDisk) as .h5ad"}
        ext = os.path.splitext(p)[1].lower()
        tail = known.get(ext, "Polariseq reads .h5ad, 10x .h5 (ps.read_10x_h5) and MatrixMarket "
                              "folders (ps.read_10x_mtx)")
        raise ValueError(f"{what}: {p} is not an HDF5 file (.h5ad or 10x .h5); {tail}")
    return p


def _near(p: str) -> str:
    """Where the path was looked for, and similar names next to it."""
    import os

    folder = os.path.dirname(os.path.abspath(p))
    out = f" (looked in {folder})" if not os.path.isabs(p) else ""
    if os.path.isdir(folder):
        out += suggest(os.path.basename(p), os.listdir(folder))
    else:
        out += f"; the folder {folder} does not exist either"
    return out
