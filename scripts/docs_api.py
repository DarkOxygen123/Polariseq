"""The API reference of the website: a page per namespace, listing its functions by task with
their one-line summaries, and a page per function with its signature and docstring.

The groups are kept here. The build checks them against the code: a public function missing from
every group, or a group naming a function that is gone, fails the build. The code is read
statically (griffe), so the site builds without compiling the Rust core.
"""

from __future__ import annotations

import ast
from pathlib import Path

import griffe
from docs_griffe import _plain

ROOT = Path(__file__).resolve().parent.parent

#: Each namespace: (page, title, what it holds, where its functions are, groups).
#: A function's place is "polariseq.<owner>.<name>"; ``None`` for the top level.
NAMESPACES = [
    ("ps", "ps", "Declare the project folder and the budget first, then read the data. "
     "`plan` predicts the cost of an analysis before anything is read, and `load` reopens a saved "
     "analysis.", None, "ps.", [
         ("Project and budget",
          ["project", "set_budget", "budget", "resources", "set_workflow", "workflow"]),
         ("Planning", ["plan", "max_cells"]),
         ("Reading and loading", ["read_h5ad", "read_10x_h5", "read_10x_mtx", "load"]),
         ("The whole analysis in one call", ["process_diskbacked"]),
         ("Marker databases", ["marker_table"]),
         ("Installation", ["get_build_info"]),
     ]),
    ("data", "The data object", "What `ps.read_h5ad` returns. Subset it with `adata[cells]` or "
     "`adata[cells, genes]` (masks, positions or names).", "data", "adata.", [
         ("Analyses of the same cells", ["from_counts", "copy", "save"]),
         ("Export", ["write_h5ad", "to_anndata"]),
         ("Reading part of a file", ["head", "sample", "validate", "close"]),
     ]),
    ("pp", "ps.pp", "Preprocessing, in the usual order.", "_Preprocessing", "ps.pp.", [
        ("Quality control", ["calculate_qc_metrics", "filter_cells", "filter_genes", "scrublet"]),
        ("Normalization and gene selection",
         ["normalize_total", "log1p", "highly_variable_genes", "renormalize"]),
        ("Embedding and graph", ["pca", "harmony_integrate", "neighbors"]),
    ]),
    ("tl", "ps.tl", "Tools on the neighbor graph and the clusters.", "_Tools", "ps.tl.", [
        ("Clustering", ["leiden", "subcluster", "merge_clusters", "paris"]),
        ("Embedding", ["umap"]),
        ("Markers and cell types", ["rank_genes_groups", "score_genes", "auto_annotate",
                                    "annotate"]),
        ("Trajectories", ["paga", "dpt"]),
        ("Comparisons", ["compare_conditions", "compare_analyses"]),
    ]),
    ("get", "ps.get", "Tables taken out of an analysis.", "_Get", "ps.get.", [
        ("Tables", ["markers", "pseudobulk"]),
    ]),
    ("pl", "ps.pl", "Plots, with the `plot` extra. Every function takes `show=` and `save=` (a "
     "file path); most take `ax=` to draw into an existing Matplotlib axis.", "plotting",
     "ps.pl.", [
         ("Embeddings", ["umap", "pca", "embedding", "compare_analyses"]),
         ("Genes and groups", ["dotplot", "heatmap", "violin", "rank_genes_groups", "stacked_bar",
                               "cluster_sizes"]),
         ("Conditions", ["volcano", "fold_changes"]),
         ("Quality control", ["qc_distributions", "scatter_qc", "highly_variable_genes",
                              "pca_variance_ratio", "scrublet_score_distribution"]),
         ("A whole run", ["overview", "overview_data"]),
         ("Themes", ["set_theme", "theme", "get_theme", "categorical_palette"]),
     ]),
    ("helpers", "Results and helpers", "The objects some steps return, and the marker database "
     "functions.", "helpers", "", [
         ("Objects", ["Project", "Budget", "Plan", "AnalysisComparison"]),
         ("Marker database", ["get_tissue_markers", "get_all_markers", "score_cell_type"]),
     ]),
]

#: Where the objects of the "helpers" page live.
_HELPERS = {"Project": "polariseq._project.Project", "Budget": "polariseq.resources.Budget",
            "Plan": "polariseq.planning.Plan",
            "AnalysisComparison": "polariseq._analyses.AnalysisComparison"}


def _public_functions(obj) -> list[str]:
    return [n for n, m in obj.members.items()
            if not n.startswith("_") and not m.is_alias and m.is_function]


def _locate(pkg, owner: str | None, name: str) -> tuple[str, object]:
    """The import path of a documented object, and the object."""
    if owner is None:
        obj = pkg.members[name]
        if obj.is_module:  # ps.resources: the function shares its name with a module
            return f"polariseq.{name}_function", pkg.members[f"{name}_function"]
        return (obj.target_path if obj.is_alias else obj.path), obj
    if owner == "data":
        cls = "AnnData" if name in pkg.members["AnnData"].members else "LazyAnnData"
        return f"polariseq.{cls}.{name}", pkg.members[cls].members[name]
    if owner == "plotting":
        return f"polariseq.plotting.{name}", pkg.members["plotting"].members[name]
    if owner == "helpers":
        path = _HELPERS.get(name, f"polariseq.markers.{name}")
        obj = pkg
        for part in path.split(".")[1:]:
            obj = obj.members[part]
        return path, obj
    return f"polariseq.{owner}.{name}", pkg.members[owner].members[name]


def _check(pkg) -> None:
    """Every public function is in one group, and every group names something that exists."""
    problems = []
    for page, _, _, owner, _, groups in NAMESPACES:
        listed = [n for _, names in groups for n in names]
        for n in listed:
            try:
                _locate(pkg, owner, n)
            except KeyError:
                problems.append(f"{page}: {n} does not exist")
        expected: list[str] = []
        if owner in ("_Preprocessing", "_Tools", "_Get"):
            expected = _public_functions(pkg.members[owner])
        elif owner == "plotting":
            expected = _public_functions(pkg.members["plotting"])
        elif owner == "data":
            expected = _public_functions(pkg.members["AnnData"])
        elif owner == "helpers":
            expected = _public_functions(pkg.members["markers"])
        missing = sorted(set(expected) - set(listed))
        if missing:
            problems.append(f"{page}: not in any group: {missing}")
    if problems:
        raise ValueError("scripts/docs_api.py: the API groups do not match the code: "
                         + "; ".join(problems) + ". Update NAMESPACES.")


def _summary(obj) -> str:
    doc = obj.docstring.value if obj.docstring else ""
    first = _plain(doc).split("\n\n")[0]
    return " ".join(first.split()).replace("|", "&#124;")


def _function_page(page: str, title: str, prefix: str, name: str, path: str, kind: str) -> str:
    heading = f"{prefix}{name}"
    opts = [f'heading: "{heading}"', "heading_level: 1", "show_root_toc_entry: false"]
    if kind != "class":
        opts.append("members: false")
    return (f"---\ntitle: {heading}\n---\n\n"
            f'<p class="api-parent"><a href="../">{title}</a></p>\n\n'
            f"::: {path}\n    options:\n" + "".join(f"      {o}\n" for o in opts))


def _attributes_table() -> list[str]:
    attrs = griffe.load("polariseq._api_reference", search_paths=[str(ROOT / "python")],
                        allow_inspection=False).members["_DATA_ATTRIBUTES"].value
    rows = ["## Attributes\n", "| attribute | type | what it holds |", "|---|---|---|"]
    for row in attrs.elements:
        a, t, d = (ast.literal_eval(str(e)) for e in row.elements)
        rows.append(f"| `{a}` | {t} | {d.replace('|', '&#124;')} |")
    return rows + [""]


_INDEX = """# API reference

Every public function, with its arguments, their types and defaults, and what it returns.

```python
import polariseq as ps
```

| namespace | what is in it |
|---|---|
| [`ps`](ps/index.md) | the project folder, the memory budget, planning, reading, loading \
saved analyses |
| [The data object](data/index.md) | what `ps.read_h5ad` returns: the matrix, its annotations, \
saving and export |
| [`ps.pp`](pp/index.md) | preprocessing: quality control, doublets, normalization, gene \
selection, PCA, integration, the neighbor graph |
| [`ps.tl`](tl/index.md) | clustering, UMAP, markers, cell types, trajectories, comparisons |
| [`ps.get`](get/index.md) | tables taken out of an analysis |
| [`ps.pl`](pl/index.md) | plots |
| [Results and helpers](helpers/index.md) | `Project`, `Budget`, `Plan`, `AnalysisComparison`, \
the marker database |
"""


def pages(pkg, anndata_figure: str) -> dict[str, str]:
    """Every page of the API reference, by its path in the site."""
    _check(pkg)
    out = {"api/index.md": _INDEX}
    for page, title, about, owner, prefix, groups in NAMESPACES:
        shown = f"`{title}`" if title.startswith("ps") else title
        body = [f"# {shown}\n", about + "\n"]
        if page == "data":
            body += [anndata_figure.replace("../assets/", "../../assets/")]
            body += _attributes_table()
        for group, names in groups:
            body += [f"## {group}\n", "| | |", "|---|---|"]
            for n in names:
                path, obj = _locate(pkg, owner, n)
                body.append(f"| [`{prefix}{n}`]({n}.md) | {_summary(obj)} |")
                kind = "class" if obj.is_class else "function"
                out[f"api/{page}/{n}.md"] = _function_page(page, shown, prefix, n, path, kind)
            body.append("")
        out[f"api/{page}/index.md"] = "\n".join(body)
    return out
