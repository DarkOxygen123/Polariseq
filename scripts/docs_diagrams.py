"""The structure diagrams of the website, drawn with Matplotlib.

    python scripts/docs_diagrams.py

Each diagram draws a data structure as it is laid out (a matrix as a grid, its annotations as the
strips beside it, a file as its parts), small enough to sit beside the text that explains it. Each
is written twice into ``docs/assets/diagrams/``, ``<name>-light.svg`` and ``<name>-dark.svg``, which
the pages show with Material's ``#only-light`` / ``#only-dark``. The SVGs are deterministic: run the
script after changing a diagram, and only that diagram's files change.

Colours follow the site: the accent blue for matrix values, a teal for annotations, and greys for
structure (outlines, empty entries, what a filter leaves out). The blue and teal pair passes a
colour-blind separation check in both modes; every part is labelled as well.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "diagrams"

THEMES = {
    "light": {"ink": "#2e3340", "muted": "#5b6270", "line": "#b4bcc8", "grid": "#d9dde3",
              "surface": "#ffffff", "data": "#3e7fc1", "note": "#1f9e7e", "off": "#8c96a6"},
    "dark": {"ink": "#e8eaee", "muted": "#aab1bd", "line": "#5a6272", "grid": "#343a46",
             "surface": "#1c1f26", "data": "#4f93d6", "note": "#25a882", "off": "#6b7383"},
}
T: dict[str, str] = THEMES["light"]

#: Inches per data unit: the labels show at about the size of the text beside them.
SCALE = 1 / 13.5


def tint(color: str, amount: float) -> tuple[float, float, float]:
    """A theme colour (by role) mixed into the page surface."""
    c, s = to_rgb(T[color]), to_rgb(T["surface"])
    return tuple(amount * a + (1 - amount) * b for a, b in zip(c, s, strict=True))


def canvas(width: float, height: float):
    fig = plt.figure(figsize=(width * SCALE, height * SCALE))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, s, *, size=8, color="muted", ha="center", va="center", bold=False,
         mono=False, rotation=0):
    ax.text(x, y, s, ha=ha, va=va, color=T[color], fontsize=size, rotation=rotation,
            fontweight="bold" if bold else "normal",
            family="DejaVu Sans Mono" if mono else "DejaVu Sans")


def rect(ax, x, y, w, h, fill, edge="line", lw=0.8, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=T[edge], linewidth=lw,
                           hatch=hatch))


def outline(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=T["line"], linewidth=0.9))


def cells(ax, x, y, ncols, nrows, size, filled, amount=0.35):
    """A grid of ``nrows`` × ``ncols`` squares; ``filled`` holds the (row, col) with a value,
    row 0 at the top."""
    for r in range(nrows):
        for c in range(ncols):
            fill = tint("data", amount) if (r, c) in filled else T["surface"]
            rect(ax, x + c * size, y + (nrows - 1 - r) * size, size, size, fill, edge="grid",
                 lw=0.6)
    outline(ax, x, y, ncols * size, nrows * size)


def strip(ax, x, y, w, h, label, *, vertical=False, columns=0):
    """An annotation strip: a teal band with column dividers and its name inside."""
    rect(ax, x, y, w, h, tint("note", 0.18), lw=0.9)
    for k in range(1, columns + 1):
        if vertical:
            xx = x + w * k / (columns + 1)
            ax.plot([xx, xx], [y, y + h], color=tint("note", 0.45), linewidth=0.6)
        else:
            yy = y + h * k / (columns + 1)
            ax.plot([x, x + w], [yy, yy], color=tint("note", 0.45), linewidth=0.6)
    text(ax, x + w / 2, y + h / 2, label, size=8.5, color="ink", bold=True, mono=True,
         rotation=90 if vertical else 0)


def rounded(ax, x, y, w, h, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=fill, edgecolor=T["line"], linewidth=0.9))


def bracket(ax, x0, x1, y, label):
    """A thin bracket under ``x0..x1`` with a label below it."""
    ax.plot([x0, x0, x1, x1], [y, y - 0.8, y - 0.8, y], color=T["muted"], linewidth=0.8)
    text(ax, (x0 + x1) / 2, y - 2.2, label, size=7.5)


# --- the diagrams ------------------------------------------------------------------------------

def anndata():
    """The analysis object: X with obs, var, obsm, obsp and uns around it."""
    fig, ax = canvas(80, 40)
    sz, x0, y0 = 2.6, 14, 4
    filled = {(0, 1), (0, 6), (1, 3), (2, 0), (2, 4), (2, 9), (3, 2), (4, 5), (4, 8), (5, 1),
              (5, 7), (6, 3), (7, 0), (7, 6), (7, 9)}
    cells(ax, x0, y0, 10, 8, sz, filled)
    text(ax, x0 + 5 * sz, y0 - 1.8, "X: cells × genes, sparse", size=7.5)
    strip(ax, 4, y0, 8, 8 * sz, "obs", vertical=True, columns=2)
    text(ax, 8, y0 - 1.8, "per cell", size=7.5)
    strip(ax, x0, y0 + 8 * sz + 2, 10 * sz, 6, "var", columns=2)
    text(ax, x0 - 1, y0 + 8 * sz + 5, "per gene", size=7.5, ha="right")
    ox = x0 + 10 * sz + 3
    for xx, w, name in ((ox, 6, "X_pca"), (ox + 7, 3, "X_umap")):
        rect(ax, xx, y0, w, 8 * sz, tint("data", 0.12), lw=0.9)
        text(ax, xx + w / 2, y0 + 4 * sz, name, size=7.5, color="ink", mono=True, rotation=90)
    text(ax, ox + 5, y0 + 8 * sz + 1.6, "obsm", size=8.5, color="ink", bold=True, mono=True)
    text(ax, ox + 5, y0 - 1.8, "embeddings", size=7.5)
    px, psz = ox + 14, sz
    graph = {(r, c) for r in range(8) for c in range(8)
             if r != c and (abs(r - c) == 1 or {r, c} in ({0, 3}, {5, 7}))}
    cells(ax, px, y0, 8, 8, psz, graph, amount=0.25)
    text(ax, px + 4 * psz, y0 + 8 * sz + 1.6, "obsp", size=8.5, color="ink", bold=True,
         mono=True)
    text(ax, px + 4 * psz, y0 - 1.8, "cells × cells graphs", size=7.5)
    rounded(ax, px, y0 + 8 * sz + 4, 8 * psz, 5.4, tint("off", 0.12))
    text(ax, px + 4 * psz, y0 + 8 * sz + 6.7, "uns", size=8.5, color="ink", bold=True,
         mono=True)
    text(ax, px - 1, y0 + 8 * sz + 6.7, "everything else", size=7.5, ha="right")
    return fig


def csr():
    """A sparse matrix and its three arrays, one row picked out."""
    fig, ax = canvas(74, 27)
    rows = [{1: 3, 4: 1}, {2: 5}, {0: 2, 3: 1, 5: 4}, {4: 7}]
    pick, sz, x0, y0 = 2, 3.2, 8, 5
    for r, row in enumerate(rows):
        y = y0 + (3 - r) * sz
        text(ax, x0 - 1, y + sz / 2, f"cell {r}", size=7.5, ha="right",
             color="ink" if r == pick else "muted")
        for c in range(6):
            v = row.get(c)
            fill = tint("data", 0.45 if r == pick else 0.16) if v else T["surface"]
            rect(ax, x0 + c * sz, y, sz, sz, fill, edge="grid", lw=0.6)
            if v:
                text(ax, x0 + c * sz + sz / 2, y + sz / 2, str(v), size=8, color="ink")
    outline(ax, x0, y0, 6 * sz, 4 * sz)
    text(ax, x0 + 3 * sz, y0 + 4 * sz + 1.6, "genes", size=7.5)

    owner = [0, 0, 1, 2, 2, 2, 3]
    arrays = [("indptr", "u32", [0, 2, 3, 6, 7], [False, False, True, True, False]),
              ("indices", "u32", [1, 4, 2, 0, 3, 5, 4], [o == pick for o in owner]),
              ("data", "f32", [3, 1, 5, 2, 1, 4, 7], [o == pick for o in owner])]
    ax0, w = 42, 4.3
    for k, (name, kind, values, hot) in enumerate(arrays):
        y = 18 - k * 6.5
        text(ax, ax0 - 1, y + 2.1, name, size=8, color="ink", ha="right", bold=True, mono=True)
        text(ax, ax0 - 1, y + 0.4, kind, size=7, ha="right")
        for i, (v, h) in enumerate(zip(values, hot, strict=True)):
            rect(ax, ax0 + i * w, y, w, 3.2, tint("data", 0.45) if h else T["surface"], lw=0.7)
            text(ax, ax0 + i * w + w / 2, y + 1.6, str(v), size=8, color="ink")
    bracket(ax, ax0 + 3 * w, ax0 + 6 * w, 4.6, "cell 2: from indptr[2] to indptr[3]")
    return fig


def store():
    """The compact block store: rows in chunks, a chunk's byte planes, the three files."""
    fig, ax = canvas(84, 32)
    mx, my, mw, band = 3, 5, 9, 3.6
    for i in range(6):
        hot = i == 1
        y = my + (5 - i) * band
        rect(ax, mx, y, mw, band, tint("data", 0.4 if hot else 0.12))
        text(ax, mx + mw / 2, y + band / 2, f"chunk {i}", size=7, color="ink" if hot else "muted")
    text(ax, mx + mw / 2, my - 1.8, "1,024 rows each", size=7.5)
    top, bot = my + 5 * band, my + 4 * band
    ax.add_patch(Polygon([(mx + mw, top), (22, 27.5), (22, 6), (mx + mw, bot)], closed=True,
                         facecolor=tint("data", 0.07), edgecolor="none"))
    # Byte planes: dark where the bytes vary, pale where they are nearly always zero.
    y = 26.5
    for name, levels in (("gene ids", [0.55, 0.22, 0.07, 0.04]), ("values", [0.55, 0.08])):
        text(ax, 22, y + 1.1, name, size=7.5, color="ink", ha="left", bold=True)
        y -= 1.2
        for b, lvl in enumerate(levels):
            y -= 2.4
            rect(ax, 22, y, 22, 2, tint("data", lvl), lw=0.6)
            text(ax, 45, y + 1, f"byte {b}", size=6.8, ha="left")
        y -= 3.6
    text(ax, 33, 3.4, "each plane deflated on its own", size=7.5)
    fx, fw = 58, 24
    files = [("{prefix}.indptr", "u64 row pointers"), ("{prefix}.cz", "the chunks"),
             ("{prefix}.czi", "chunk offsets")]
    for k, (name, what) in enumerate(files):
        y = 21 - k * 8
        rect(ax, fx, y, fw, 4.4, tint("off", 0.1), lw=0.9)
        text(ax, fx, y + 5.8, name, size=7.5, color="ink", ha="left", mono=True, bold=True)
        text(ax, fx + fw, y + 5.8, what, size=6.8, ha="right")
        if name.endswith(".cz"):
            xs = fx
            for i, length in enumerate((3.2, 5.4, 2.6, 4.4, 3.8, 3.0)):
                rect(ax, xs, y, length, 4.4, tint("data", 0.4 if i == 1 else 0.14), lw=0.6)
                xs += length
        elif name.endswith(".czi"):
            for xs in (0, 3.2, 8.6, 11.2, 15.6, 19.4, 22.4):
                ax.plot([fx + xs + 0.3] * 2, [y + 0.8, y + 3.6], color=T["data"], linewidth=1)
        else:
            for i in range(12):
                ax.plot([fx + 2 * i + 1] * 2, [y + 1.2, y + 3.2], color=T["off"], linewidth=0.8)
    return fig


def rowblocks():
    """A pass over row blocks: each block's partial result, merged in block order."""
    fig, ax = canvas(58, 30)
    mx, my, mw, band = 3, 3, 14, 3.8
    for i, t in enumerate((2, 1, 3, 1, 2, 3)):
        y = my + (5 - i) * band
        rect(ax, mx, y, mw, band, tint("data", 0.14))
        text(ax, mx + mw / 2, y + band / 2, f"block {i}", size=7.5, color="ink")
        rounded(ax, mx + mw + 1.2, y + 0.8, 5.6, band - 1.6, tint("note", 0.16))
        text(ax, mx + mw + 4, y + band / 2, f"T{t}", size=7, color="ink")
        rect(ax, 26, y + 0.5, band - 1, band - 1, tint("data", 0.4), lw=0.7)
        text(ax, 26 + (band - 1) / 2, y + band / 2, f"p{i}", size=7, color="ink")
    head = my + 6 * band + 1.6
    text(ax, mx + mw / 2, head, "rows", size=7.5)
    text(ax, mx + mw + 4, head, "thread", size=7.5)
    text(ax, 27.4, head, "partial", size=7.5)
    bx, mid = 31, my + 3 * band
    ax.plot([bx, bx + 1, bx + 1, bx], [my + 6 * band - 0.5, my + 6 * band - 0.5, my + 0.5,
                                       my + 0.5], color=T["muted"], linewidth=0.8)
    ax.plot([bx + 1, bx + 2.4], [mid, mid], color=T["muted"], linewidth=0.8)
    rect(ax, bx + 3, mid - 2.5, 5, 5, tint("data", 0.55), lw=0.9)
    text(ax, bx + 5.5, mid, "Σ", size=10, color="ink", bold=True)
    text(ax, bx + 9.5, mid + 1.4, "p0 + p1 + ... + p5", size=7.5, color="ink", ha="left",
         mono=True)
    text(ax, bx + 9.5, mid - 1.4, "in block order", size=7.5, ha="left")
    return fig


def views():
    """A DiskMatrix's views on the counts: gene and cell masks and a scale per file row."""
    fig, ax = canvas(76, 33)
    sz, x0, y0, ncol, nrow = 3, 10, 5, 10, 7
    gene_out, cell_out = {2, 6}, {1, 5}
    scale = [0.30, 0.18, 0.42, 0.25, 0.36, 0.20, 0.48]
    for r in range(nrow):
        y = y0 + (nrow - 1 - r) * sz
        for c in range(ncol):
            off = c in gene_out or r in cell_out
            rect(ax, x0 + c * sz, y, sz, sz,
                 tint("off", 0.16) if off else tint("data", 0.14 + 0.5 * scale[r]),
                 edge="grid", lw=0.6, hatch="////" if off else None)
        rect(ax, x0 + ncol * sz + 2, y, 3, sz, tint("note", 0.12 + scale[r]), edge="grid", lw=0.6)
    outline(ax, x0, y0, ncol * sz, nrow * sz)
    for c in range(ncol):
        out = c in gene_out
        text(ax, x0 + c * sz + sz / 2, y0 + nrow * sz + 1.4, "×" if out else "✓", size=7.5,
             color="muted" if out else "ink")
    for r in range(nrow):
        out = r in cell_out
        text(ax, x0 - 1.4, y0 + (nrow - 1 - r) * sz + sz / 2, "×" if out else "✓", size=7.5,
             color="muted" if out else "ink")
    text(ax, x0 + ncol * sz / 2, y0 + nrow * sz + 4, "gene filter", size=8, color="ink",
         bold=True)
    text(ax, x0 - 4.4, y0 + nrow * sz / 2, "cell filter", size=8, color="ink", bold=True,
         rotation=90)
    sx = x0 + ncol * sz + 3.5
    text(ax, sx, y0 + nrow * sz + 4, "scale", size=8, color="ink", bold=True)
    text(ax, sx, y0 + nrow * sz + 1.4, "per row", size=7)
    text(ax, x0 + ncol * sz / 2, y0 - 2, "the counts on disk", size=7.5)
    lx = sx + 6
    for k, (label, fill, hatch) in enumerate((("read by a step", tint("data", 0.3), None),
                                              ("left out", tint("off", 0.16), "////"))):
        y = 20 - k * 5
        rect(ax, lx, y, 3, 3, fill, edge="grid", lw=0.6, hatch=hatch)
        text(ax, lx + 4.2, y + 1.5, label, size=7.5, ha="left")
    return fig


def crates():
    """The crates as layers."""
    fig, ax = canvas(72, 28)
    layers = [
        (24, [(2, 68, "polariseq", "Python API", "off")]),
        (17, [(2, 68, "polariseq-python", "Python bindings", "off")]),
        (10, [(2, 32, "polariseq-io", "file reading", "note"),
              (36, 34, "polariseq-hdf5-raw", "chunk reads", "off")]),
        (3, [(2, 68, "polariseq-core", "algorithms", "data")]),
    ]
    for y, parts in layers:
        for x, w, name, what, color in parts:
            rounded(ax, x, y - 3, w, 5.6, tint(color, 0.16))
            text(ax, x + 1.4, y - 0.2, name, size=8, color="ink", ha="left", bold=True,
                 mono=True)
            text(ax, x + w - 1.4, y - 0.2, what, size=7, ha="right")
    return fig


DIAGRAMS = {"anndata": anndata, "csr": csr, "store": store, "rowblocks": rowblocks,
            "views": views, "crates": crates}


def main() -> None:
    global T
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.svg"):
        if old.stem.rsplit("-", 1)[0] not in DIAGRAMS:
            old.unlink()  # a diagram no longer drawn
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "polariseq",
                         "font.family": "DejaVu Sans", "hatch.linewidth": 0.5})
    for mode, theme in THEMES.items():
        T = theme
        plt.rcParams["hatch.color"] = T["line"]
        for name, draw in DIAGRAMS.items():
            fig = draw()
            fig.savefig(OUT / f"{name}-{mode}.svg", transparent=True, metadata={"Date": None},
                        bbox_inches="tight", pad_inches=0.04)  # trimmed to what is drawn
            plt.close(fig)
    print(f"{len(DIAGRAMS) * len(THEMES)} diagrams in {OUT}")


if __name__ == "__main__":
    main()
