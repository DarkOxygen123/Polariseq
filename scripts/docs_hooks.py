"""MkDocs hooks of the website: the API pages written from the code (``docs_api.py``), the example
scripts and the notebook as pages, the root files (INSTALL, CHANGELOG) as pages, and links that
work on GitHub and on the site alike.

Each API page lists the public functions of one namespace as mkdocstrings directives, so the site
renders every signature and docstring from the source and cannot drift from it. The lists follow
``polariseq/_api_reference.py`` (which writes ``docs/API.md``): the same functions, in the same
groups. The code is read statically (griffe), so the site builds without compiling the Rust core.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import griffe
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docs_api  # noqa: E402  (a sibling module; MkDocs loads this hook by its path)

_ROOT = Path(__file__).resolve().parent.parent

_ANNDATA_FIGURE = """<figure markdown="span">
  ![The matrix X of cells by genes, with obs beside it for each cell, var above it for each gene, \
obsm embeddings and obsp graphs beside it, and uns](../assets/diagrams/anndata-light.svg#only-light)
  ![The matrix X of cells by genes, with obs beside it for each cell, var above it for each gene, \
obsm embeddings and obsp graphs beside it, and uns](../assets/diagrams/anndata-dark.svg#only-dark)
  <figcaption>The parts of an analysis.</figcaption>
</figure>
"""


_REPO = "https://github.com/DarkOxygen123/Polariseq/blob/main/"

#: Repository files shown whole as pages of the site, by the page each becomes.
_MIRRORED = {
    "INSTALL.md": "install.md",
    "CHANGELOG.md": "changelog.md",
    "docs/API.md": "api/index.md",  # the site's API pages replace the one-file reference
    "examples/README.md": "how-to/examples/index.md",
    "examples/NODE_SETUP.md": "how-to/run-on-a-cluster.md",
    "notebooks/02_every_command.ipynb": "getting-started/every-command.ipynb",
}

_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)(#[^)\s]*)?\)")
_HREF = re.compile(r'href="(?!https?://|#|mailto:)([^"#]+)(#[^"]*)?"')


def _example_page(path: Path) -> str:
    """An example script as a page: its docstring as the text, then the code below it."""
    src = path.read_text()
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    head, _, body = doc.partition("\n")
    title = re.sub(r"^\d+\s*[-\u2014]\s*", "", head).rstrip(".")
    # The page gives the command to run; the docstring's own "Run:" line would repeat it.
    body = re.sub(r"^Run:.*\n+", "", body.strip(), flags=re.M)
    code = src.splitlines()[tree.body[0].end_lineno:] if doc else src.splitlines()
    rel = path.relative_to(_ROOT).as_posix()
    return (f"# {title}\n\n{body.strip()}\n\n"
            f"[`{rel}`]({_REPO}{rel}) &nbsp;·&nbsp; run it with `python {rel}`\n\n"
            "```python\n" + "\n".join(code).strip("\n") + "\n```\n")


def _examples() -> dict[str, Path]:
    scripts = sorted((_ROOT / "examples").glob("[0-9]*.py"))
    return {f"how-to/examples/{p.stem}.md": p for p in scripts}


class _Links:
    """Where each page's links point: a page of the site (relative to the page), else the file on
    GitHub, so the Markdown stays valid both on GitHub and on the site."""

    def __init__(self, files) -> None:
        # Every file the site publishes (pages, images, styles), by the repository file it is
        # made from; a file of docs/ the site leaves out is linked on GitHub.
        self.site = {f"docs/{f.src_uri}": f.src_uri for f in files if f.inclusion.is_included()}
        self.site.update(_MIRRORED)
        self.site.update({p.relative_to(_ROOT).as_posix(): uri for uri, p in _examples().items()})
        self.source = {uri: repo for repo, uri in self.site.items()}

    def fix(self, page_uri: str, target: str, anchor: str) -> str | None:
        here = _ROOT / self.source.get(page_uri, f"docs/{page_uri}")
        path = (here.parent / target).resolve()
        if not path.is_relative_to(_ROOT):
            return None
        repo = path.relative_to(_ROOT).as_posix()
        if repo in self.site:
            docs = _ROOT / "docs"
            rel = os.path.relpath(docs / self.site[repo], (docs / page_uri).parent)
            return Path(rel).as_posix() + anchor
        if path.exists():
            return _REPO + repo + anchor
        return None


_links: _Links | None = None


def on_page_markdown(markdown, page, config, files):
    def fix(m: re.Match) -> str:
        new = _links.fix(page.file.src_uri, m.group(1), m.group(2) or "")
        return f"]({new})" if new else m.group(0)

    return _LINK.sub(fix, markdown)


def on_page_content(html, page, config, files):
    # A notebook's Markdown is rendered by its plugin, past on_page_markdown: fix its links here.
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    def fix(m: re.Match) -> str:
        new = _links.fix(page.file.src_uri, m.group(1), m.group(2) or "")
        return f'href="{new}"' if new else m.group(0)

    return _HREF.sub(fix, html)


@event_priority(50)  # before the notebook plugin, which converts the .ipynb files it is given
def on_files(files, config):
    global _links
    # Loaded with the site's docstring extension, as mkdocstrings loads it, so both see the same
    # objects (``ps.resources`` included, which shares its name with a module).
    pkg = griffe.load("polariseq", search_paths=[str(_ROOT / "python")], allow_inspection=False,
                      extensions=griffe.load_extensions(
                          str(_ROOT / "scripts" / "docs_griffe.py") + ":PolariseqDocstrings"))
    for uri, content in docs_api.pages(pkg, _ANNDATA_FIGURE).items():
        files.append(File.generated(config, uri, content=content))
    for repo, uri in _MIRRORED.items():
        if uri != "api/index.md":
            files.append(File.generated(config, uri, abs_src_path=str(_ROOT / repo)))
    for uri, path in _examples().items():
        files.append(File.generated(config, uri, content=_example_page(path)))
    _links = _Links(files)
    return files


def on_post_build(config):
    """Copy the Rust API reference (``cargo doc --no-deps -p polariseq-core -p polariseq-io``)
    into the site at ``rust/api/``, where the Rust page links to it."""
    rustdoc = _ROOT / "target" / "doc"
    if not (rustdoc / "polariseq_core" / "index.html").exists():
        logging.getLogger("mkdocs").info(
            "The Rust API reference is not built, so rust/api/ is left out. Build it with: "
            "cargo doc --no-deps -p polariseq-core -p polariseq-io")
        return
    site = Path(config["site_dir"])
    shutil.copytree(rustdoc, site / "rust" / "api", dirs_exist_ok=True)
    # The site's links into the Rust reference are raw HTML, which MkDocs does not check: check
    # them here, so a renamed module fails the strict build.
    log = logging.getLogger("mkdocs")
    for page in site.rglob("index.html"):
        if (site / "rust" / "api") in page.parents:
            continue
        for href in re.findall(r'href="([^"#]*rust/api/[^"#]*)', page.read_text()):
            target = (page.parent / href).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                log.warning(f"{page.relative_to(site)} links to {href}, which the Rust reference "
                            f"does not have")
