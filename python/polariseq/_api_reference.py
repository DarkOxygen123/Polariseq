"""The API reference, ``docs/API.md``, written from the code itself.

Every public function and method, with its arguments, their types and
defaults, what it returns and what it does, read from the signatures and
docstrings, so the reference cannot drift from the code (a test compares the
file with this module's output).

    python -m polariseq._api_reference > docs/API.md
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any

#: Attributes of the data object, which no signature describes.
_DATA_ATTRIBUTES = [
    ("X", "matrix", "The expression matrix: a `SparseMatrix` in memory, a `DiskMatrix` out of core "
     "(a view of the on-disk copy with its filters and normalization), or a NumPy array."),
    ("obs", "dict[str, array]", "Per-cell annotations: the files' columns, quality metrics, clusters, "
     "scores. Assign a NumPy array, a list or a `pandas.Categorical` of one value per cell."),
    ("var", "dict[str, array]", "Per-gene annotations (gene selection statistics, `included`, ...)."),
    ("obs_names / var_names", "list[str]", "Cell barcodes and gene names."),
    ("obsm", "dict[str, array]", "Per-cell embeddings: `X_pca`, `X_pca_harmony`, `X_umap`."),
    ("obsp", "dict[str, sparse]", "Cell-by-cell graphs: `connectivities`, `distances`."),
    ("uns", "dict", "Everything else: each step's parameters and results (`hvg`, `pca`, `leiden`, "
     "`scrublet`, `compare_conditions`, `rank_genes_groups`, ...)."),
    ("n_obs / n_vars / shape", "int / tuple", "Cells, genes, and both."),
    ("mode", "str", "`'memory'` or `'out-of-core'`: where the analysis runs (chosen by the planner "
     "unless given to `read_h5ad`)."),
    ("name", "str", "The dataset's folder in the project (`runs/<name>/`)."),
    ("analysis", "str | None", "The analysis's name within its dataset (`from_counts`); `None` for the "
     "first one."),
]

_INTRO = """# Polariseq API reference

Every public function, with its arguments, their types and defaults, and what
it returns. Written from the code (`python -m polariseq._api_reference`), so it
matches the installed version: {version}.

```python
import polariseq as ps
```

| namespace | what is in it |
|---|---|
| `ps` | the project folder, the memory budget, planning, reading, loading saved analyses |
| the data object | what `ps.read_h5ad` returns: the matrix, annotations, and `save`, `from_counts`, `write_h5ad`, `to_anndata`, subsetting |
| `ps.pp` | preprocessing: quality control, filters, doublets, normalization, gene selection, PCA, integration, the neighbour graph |
| `ps.tl` | tools: clustering, embedding, markers, annotation, curation, trajectories, comparisons |
| `ps.get` | tables taken out of an analysis |
| `ps.pl` | plots (the `plot` extra: `pip install "polariseq[plot]"`) |

## Using it well

- **Declare a project folder first** (`ps.project("my_analysis")`): everything
  Polariseq writes goes there, results are memory-mapped files instead of
  memory, and saved analyses live there.
- **Set the budget** (`ps.set_budget(ram_gb=..., threads=...)`) before the
  first step: it decides in-memory or out-of-core, the block size, and fixes
  the thread count (the worker pool cannot change size later).
- **Ask before running** (`ps.plan(path)`): predicted peak memory and disk, per
  step, from the file's metadata alone.
- **Out of core costs nothing to branch**: `from_counts` gives a new analysis
  of the same cells over the same on-disk copy; compare settings with
  `ps.tl.compare_analyses(..., seeds=2)` and read the result against the
  seed-to-seed agreement it reports.
- **Save, don't export, to continue later** (`adata.save` / `ps.load`): the
  counts are pointed to, not copied. Export (`write_h5ad`, `to_anndata`) when
  another tool needs the data.
- **Name cell types, then check them** (`ps.tl.auto_annotate(adata, "leiden",
  tissue="immune")`): a table per cluster with the name, its Cell Ontology
  term, a score, the share of cells supporting it, the runner-up and rare
  groups inside the cluster. `source=` picks the marker database
  (`ps.marker_table` shows one); give `tissue` when you know it.
- **Errors name the fix**: a misspelled column, gene or file names the closest
  match; a step run too early names the step to run first.
"""


def _summary(doc: str) -> str:
    """The docstring's first paragraph."""
    paras = inspect.cleandoc(doc or "").split("\n\n")
    return " ".join(paras[0].split()) if paras and paras[0] else ""


_SECTION = re.compile(r"^(Args|Arguments|Returns|Raises|Yields|Example|Examples|Note|Notes):\s*$")
_ITEM = re.compile(r"^(?:[-*]|\d+[.)])\s")


def _body(doc: str) -> str:
    """The docstring's explanation between its first paragraph and its sections, as
    Markdown: paragraphs joined, bullet and numbered lists kept, ``::`` blocks as code."""
    paras = inspect.cleandoc(doc or "").split("\n\n")[1:]
    out: list[str] = []
    code_next = False
    for para in paras:
        lines = para.splitlines()
        if not lines or _SECTION.match(lines[0].strip()):
            break
        if code_next:
            out.append("```python\n" + textwrap.dedent(para).strip("\n") + "\n```")
            code_next = False
            continue
        lead = ""
        if not _ITEM.match(lines[0].lstrip()) and any(_ITEM.match(ln.lstrip()) for ln in lines[1:]):
            first = next(i for i, ln in enumerate(lines) if i and _ITEM.match(ln.lstrip()))
            lead, lines = _md(" ".join(" ".join(lines[:first]).split())) + "\n\n", lines[first:]
        if _ITEM.match(lines[0].lstrip()):
            items: list[str] = []
            indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines:
                if _ITEM.match(line.lstrip()) and len(line) - len(line.lstrip()) == indent:
                    items.append(line.strip())
                else:
                    items[-1] += " " + line.strip()
            out.append(lead + "\n".join(_md(" ".join(i.split())) for i in items))
            continue
        text = " ".join(para.split())
        if text.endswith("::"):
            text, code_next = text[:-1], True
        out.append(_md(text))
    return "\n\n".join(out)


def _section(doc: str, name: str) -> str:
    """A Google-style section (``Args:``, ``Returns:``, ``Raises:``) of a docstring."""
    lines = inspect.cleandoc(doc or "").splitlines()
    out, inside, indent = [], False, None
    for line in lines:
        if line.strip() == f"{name}:":
            inside = True
            continue
        if inside:
            if line.strip() and not line.startswith(" ") and line.rstrip().endswith(":"):
                break  # the next section
            if line.strip() and not line.startswith(" ") and indent is not None:
                break
            if line.strip() and indent is None:
                indent = len(line) - len(line.lstrip())
            out.append(line)
    return "\n".join(out).rstrip()


def _args(doc: str) -> dict[str, str]:
    """``Args:`` entries: argument name (or names, ``a, b``) to its description."""
    text = _section(doc, "Args")
    if not text:
        return {}
    base = min(len(line) - len(line.lstrip()) for line in text.splitlines() if line.strip())
    entries: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        if not line.strip():
            continue
        level = len(line) - len(line.lstrip())
        m = re.match(r"\s*([\w*, ]+?):\s*(.*)$", line)
        if level == base and m:
            current = m.group(1).strip()
            entries[current] = m.group(2).strip()
        elif current is not None:
            entries[current] += " " + line.strip()
    out = {}
    for names, desc in entries.items():
        for n in names.split(","):
            out[n.strip().lstrip("*")] = desc
    return out


#: Arguments many functions share, described once (a function's own description wins).
_COMMON = {
    "adata": "The analysis (what `ps.read_h5ad` returns); steps add their results to it in place.",
    "show": "Show the figure (`plt.show()`); `False` to keep drawing on it or only save it.",
    "save": "A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`).",
    "ax": "A Matplotlib axis to draw into (default: a new figure).",
    "figsize": "Figure size in inches, `(width, height)`.",
    "dpi": "Figure resolution.",
    "ncols": "Panels per row.",
    "title": "Title above the plot (default: what is shown).",
    "size": "Point size (default: chosen from the number of cells).",
    "cmap": "Matplotlib colour map for values (default: the theme's).",
    "legend_loc": "`'right'`, `'on data'` (labels at the groups) or `None`.",
}


def _delegate_args(fn: Any) -> dict[str, str]:
    """``Args:`` of the function a method hands its work to (the curation steps)."""
    from polariseq import _curation

    impl = getattr(_curation, fn.__name__, None)
    return _args(inspect.getdoc(impl) or "") if impl is not None else {}


def _type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return text.replace("'", "")


def _default(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return "required"
    return f"`{value!r}`"


def _entry(title: str, fn: Any, skip_first: bool = False) -> str:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if skip_first and params and params[0].name == "self":
        params = params[1:]
    shown = sig.replace(parameters=params)
    doc = inspect.getdoc(fn) or ""
    described = {**_COMMON, **_delegate_args(fn), **_args(doc)}
    lines = [f"### `{title}`", "", "```python", f"{title}{_signature(shown)}", "```", ""]
    summary = _summary(doc)
    if summary:
        lines += [_md(summary), ""]
    body = _body(doc)
    if body:
        lines += [body, ""]
    rows = [p for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if rows:
        lines += ["| argument | type | default | meaning |", "|---|---|---|---|"]
        for p in rows:
            desc = described.get(p.name, "")
            lines.append(f"| `{p.name}` | {_cell(_type(p.annotation))} | {_cell(_default(p.default))} "
                         f"| {_cell(desc)} |")
        lines.append("")
    ret = _section(doc, "Returns")
    if ret:
        lines += ["**Returns:** " + _md(" ".join(ret.split())), ""]
    raises = _section(doc, "Raises")
    if raises:
        lines += ["**Raises:** " + _md(" ".join(raises.split())), ""]
    return "\n".join(lines)


def _signature(sig: inspect.Signature) -> str:
    """``(a: int = 1, *, b: str = 'x') -> T``, the types without the quotes postponed
    annotations carry, the defaults as Python writes them."""
    parts, star = [], False
    for p in sig.parameters.values():
        if p.kind == p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        text = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "") + p.name
        if p.kind == p.VAR_POSITIONAL:
            star = True
        if p.annotation is not p.empty:
            text += f": {_type(p.annotation)}"
        if p.default is not p.empty:
            text += f" = {p.default!r}"
        parts.append(text)
    ret = f" -> {_type(sig.return_annotation)}" if sig.return_annotation is not sig.empty else ""
    return f"({', '.join(parts)}){ret}"


def _md(text: str) -> str:
    """Docstring markup (reStructuredText) as Markdown."""
    text = re.sub(r":(?:func|meth|class|attr|mod):`~?([^`]+)`", r"`\1`", text)
    return text.replace("``", "`")


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return _md(text).replace("|", "\\|").replace("\n", " ") if text else ""


def _public(cls: Any) -> list[tuple[str, Any]]:
    return [(n, f) for n, f in inspect.getmembers(cls, inspect.isfunction)
            if not n.startswith("_") and f.__qualname__.split(".")[0] == cls.__name__]


def reference() -> str:
    import polariseq as ps
    from polariseq import _analyses, markers, planning, plotting

    parts = [_INTRO.format(version=ps.__version__)]

    def heading(text: str, about: str = "") -> None:
        parts.append(f"## {text}\n")
        if about:
            parts.append(textwrap.fill(about, 100) + "\n")

    heading("`ps`: project, budget, planning, reading",
            "Declare the project folder and the budget first; read data with `read_h5ad` (or the 10x "
            "readers); `plan` predicts the cost before anything is read; `load` reopens a saved "
            "analysis; `marker_table` shows the markers `tl.auto_annotate` scores with.")
    for name in ("project", "set_budget", "budget", "resources", "plan", "max_cells", "read_h5ad",
                 "read_10x_h5", "read_10x_mtx", "load", "marker_table", "process_diskbacked",
                 "set_workflow", "workflow", "get_build_info"):
        parts.append(_entry(f"ps.{name}", getattr(ps, name)))

    heading("The data object",
            "What `ps.read_h5ad` returns (an `AnnData`, or a `LazyAnnData` whose matrix is read on "
            "first use; both have the methods below). Subset with `adata[cells]` or `adata[cells, "
            "genes]` (masks, positions or names).")
    parts.append("| attribute | type | what it holds |\n|---|---|---|")
    parts += [f"| `{a}` | {t} | {_cell(d)} |" for a, t, d in _DATA_ATTRIBUTES]
    parts.append("")
    for name, fn in _public(ps.AnnData):
        parts.append(_entry(f"adata.{name}", fn, skip_first=True))
    for name in ("head", "sample", "validate", "close"):
        parts.append(_entry(f"adata.{name}", getattr(ps.LazyAnnData, name), skip_first=True))

    for ns, about in (
            ("pp", "Preprocessing, in the usual order: quality control, filters, doublets "
                   "(`scrublet`, on raw counts), normalization, gene selection, PCA, integration "
                   "(`harmony_integrate`), the neighbour graph."),
            ("tl", "Tools on the neighbour graph and the clusters: `leiden`, `umap`, markers "
                   "(`rank_genes_groups`), annotation and curation, `paris`, `paga`, `dpt`, and "
                   "comparisons of conditions (`compare_conditions`) and of analyses "
                   "(`compare_analyses`)."),
            ("get", "Tables taken out of an analysis.")):
        heading(f"`ps.{ns}`", about)
        for name, fn in _public(type(getattr(ps, ns))):
            parts.append(_entry(f"ps.{ns}.{name}", fn, skip_first=True))

    heading("`ps.pl`", "Plots. Every function takes `show=` and `save=` (a file path); most take "
                       "`ax=` to draw into an existing Matplotlib axis.")
    fns = [(n, f) for n, f in inspect.getmembers(plotting, inspect.isfunction)
           if not n.startswith("_") and f.__module__ == plotting.__name__]
    for name, fn in fns:
        parts.append(_entry(f"ps.pl.{name}", fn))

    heading("Results and helpers")
    for title, obj in (("ps.Project", ps.Project), ("ps.Budget", ps.Budget)):
        for name, fn in _public(obj):
            parts.append(_entry(f"{title}.{name}", fn, skip_first=True))
    for name in ("report", "summary"):
        parts.append(_entry(f"Plan.{name}", getattr(planning.Plan, name), skip_first=True))
    parts.append("### `AnalysisComparison`\n\nWhat `ps.tl.compare_analyses` returns.\n\n"
                 + _attributes(_analyses.AnalysisComparison) + "\n")
    for name, fn in inspect.getmembers(markers, inspect.isfunction):
        if not name.startswith("_") and fn.__module__ == markers.__name__:
            parts.append(_entry(f"polariseq.markers.{name}", fn))

    heading("Errors")
    parts.append("| error | raised when |\n|---|---|\n"
                 "| `KeyError` | a column, gene, group, embedding or result named is not there "
                 "(the message names the closest match and what exists) |\n"
                 "| `FileNotFoundError` | an input file or folder is missing (the message says where "
                 "it looked) |\n"
                 "| `ps.SourceMovedError` | a saved analysis's source file has moved (`relink=` gives "
                 "its new place) |\n"
                 "| `ValueError` | a value cannot be used (the message gives the limit or what is "
                 "expected) |\n"
                 "| `ps.ProjectNotSet` | a step must write to disk and no project folder is declared |\n"
                 "| `ps.BudgetWarning` | a step is predicted to need more memory than the budget "
                 "(a warning) |\n")
    return "\n".join(parts).rstrip() + "\n"


def _attributes(cls: Any) -> str:
    doc = inspect.getdoc(cls) or ""
    attrs = _section(doc, "Attributes")
    entries = _args("Args:\n" + attrs) if attrs else {}
    rows = ["| attribute | meaning |", "|---|---|"]
    rows += [f"| `{k}` | {_cell(v)} |" for k, v in entries.items()]
    return "\n".join(rows)


if __name__ == "__main__":
    print(reference(), end="")
