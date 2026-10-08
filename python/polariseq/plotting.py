"""Polariseq plotting namespace (`ps.pl`) — v3 design system.

Calm, washed-but-semi-vibrant presentation graphics:

- Theme engine: ``ps.pl.set_theme("paper" | "notebook" | "talk")``.
- Washed categorical palettes (Paul Tol schemes blended toward neutral —
  distinguishable but never loud), soft custom sequential/diverging cmaps.
- Typographic hierarchy: light-weight titles, grey subtitles, optional
  italic captions — nothing is bold-by-default.
- Modern axis-label placement on framed charts: y-label flat at the top of
  the axis, x-label right-aligned; hairline spines; no chartjunk.
- Rich per-plot attributes: ``title / subtitle / caption / palette /
  legend_loc / legend_title / figsize / alpha / colorbar / frame / dpi``.
- Auto-annotation integration: ``umap(..., on_data=True)`` prints cluster
  or cell-type names (e.g. ``obs["cell_type"]`` from ``ps.tl.auto_annotate``)
  directly on the clusters, with a white halo, replacing the legend.

All v1/v2 signatures keep working; new attributes are keyword-only adds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from polariseq import AnnData

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

# Paul Tol schemes, washed toward neutral at draw time (see _wash).
_TOL_VIBRANT = [
    "#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311",
    "#009988", "#BBBBBB",
]
_TOL_MUTED = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
]
_TOL_RAW = _TOL_VIBRANT + _TOL_MUTED

# Light-theme colour values. These names are kept because external code may
# import them, but nothing inside this module reads them directly — every
# draw site goes through `_c()` so that a theme switch actually changes the
# colours. See `_THEME` and `set_theme`.
TEXT = "#2E3340"        # near-black with a calm blue cast
TEXT_SUB = "#828A96"    # subtitles / axis metadata
TEXT_CAP = "#9AA1AC"    # captions
SPINE = "#D8DCE2"
GRID = "#ECEEF1"
BG_SOFT = "#E7EBF0"     # fill for "other" points

WASH = 0.32             # blend factor toward neutral for categorical colors


def _wash(hexc: str, f: float = WASH) -> str:
    """Blend a color toward a soft neutral so palettes stop shouting."""
    r = int(hexc[1:3], 16) / 255
    g = int(hexc[3:5], 16) / 255
    b = int(hexc[5:7], 16) / 255
    nr, ng, nb = 0.72, 0.74, 0.77          # soft neutral target
    r, g, b = (r * (1 - f) + nr * f, g * (1 - f) + ng * f, b * (1 - f) + nb * f)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


CATEGORICAL_PALETTE = [_wash(c) for c in _TOL_RAW]
VEGA_20_SCANPY = CATEGORICAL_PALETTE    # backwards-compat alias

ACCENT = _wash("#0077BB", 0.18)         # primary series (calm steel blue)
ACCENT_2 = _wash("#CC3311", 0.25)       # secondary series (soft brick)

CONTINUOUS_CMAP = "cividis"


def _soft_cmap(colors: list[str], name: str):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


SEQ_CMAP = _soft_cmap(
    [_wash("#0077BB", 0.55), _wash("#0077BB", 0.30), _wash("#0077BB", 0.05)],
    "polariseq_seq")
DIV_CMAP = _soft_cmap(
    [_wash("#3A5F8A", 0.10), "#FFFFFF", _wash("#B0554A", 0.10)],
    "polariseq_div")
DOT_CMAP = _soft_cmap(
    ["#F3F1EE", _wash("#CC6655", 0.45), _wash("#A63D2F", 0.05)],
    "polariseq_dot")

#: Gene expression on an embedding. Not SEQ_CMAP, which runs washed-blue to
#: saturated-blue: in scRNA-seq most cells are at or near zero for any given
#: gene, so a ramp that starts at a *colour* spends most of its range on the
#: cells that carry no signal and leaves little contrast for the ones that
#: do. This starts at a near-background neutral so zero recedes, then climbs
#: through warm mid-tones to a saturated end — the expressing cells are what
#: the eye lands on, which is the entire point of a marker plot.
EXPR_CMAP = _soft_cmap(
    ["#EFF1F4", "#CBD9E8", "#7FA8D4", "#3E6FB0", "#7A4A9C", "#8C2F62"],
    "polariseq_expr")

#: The same ramp for dark backgrounds: it has to *start* dark or every
#: zero-expression cell becomes a bright dot on a black field.
EXPR_CMAP_DARK = _soft_cmap(
    ["#20252C", "#2E4A6B", "#3E6FB0", "#5B8FD6", "#A87CC8", "#E86FA6"],
    "polariseq_expr_dark")

#: The live theme. Sizes *and colours* — colours live here rather than in
#: module constants precisely so a theme can change them; when they were
#: constants, `set_theme` could only resize things and a dark theme was
#: impossible to express.
_THEME = {
    "name": "paper",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "base": 8.5,
    "title": 9.5,
    "label": 8.0,
    "tick": 6.8,
    "legend": 6.8,
    "dpi": 300,
    "linewidth": 0.8,
    "point_scale": 1.0,
    # colours
    "text": TEXT,
    "text_sub": TEXT_SUB,
    "text_cap": TEXT_CAP,
    "spine": SPINE,
    "grid": GRID,
    "bg_soft": BG_SOFT,
    "figure_bg": "#FFFFFF",
    "axes_bg": "#FFFFFF",
    "halo": "#FFFFFF",      # outline behind on-data labels
    "wash": WASH,
}

_PRESETS = {
    "paper": {},
    "notebook": {"base": 10.5, "title": 11.5, "label": 10.0, "tick": 8.5,
                 "legend": 8.5, "dpi": 150, "point_scale": 1.3},
    "talk": {"base": 13.0, "title": 14.5, "label": 12.5, "tick": 11.0,
             "legend": 11.0, "dpi": 120, "point_scale": 1.8},
    # Dark is a full colour inversion, not a background swap: text, spines,
    # grid and the halo behind on-data labels all move, and categorical
    # colours are washed *less* (0.18 against 0.32) because the same blend
    # that keeps a palette calm on white makes it muddy on near-black.
    "dark": {
        "text": "#E8EAEE", "text_sub": "#9BA3AF", "text_cap": "#7C838F",
        "spine": "#3A414D", "grid": "#2B313B",
        "bg_soft": "#333A45", "figure_bg": "#181C22", "axes_bg": "#181C22",
        "halo": "#181C22", "wash": 0.18,
        "dpi": 150, "point_scale": 1.2,
    },
    # Maximum-legibility print: no wash, heavier lines, pure black text.
    "print": {
        "text": "#000000", "text_sub": "#444444", "text_cap": "#666666",
        "spine": "#000000", "grid": "#DDDDDD", "wash": 0.0,
        "linewidth": 1.1, "dpi": 600,
    },
}

#: Theme names, for error messages and for anyone enumerating them.
THEMES = tuple(_PRESETS)


def _c(key: str) -> Any:
    """A theme value. Every draw site reads colours through this."""
    return _THEME[key]


def set_theme(name: str = "paper", **overrides: Any) -> None:
    """Set the global theme, optionally overriding individual keys.

    Presets: ``paper`` (default, dense and small for figures), ``notebook``,
    ``talk`` (large type for slides), ``dark``, ``print``.

    Any theme key may be overridden::

        ps.pl.set_theme("dark", point_scale=2.0, dpi=200)

    Themes reset from the ``paper`` baseline each time, so switching between
    them does not accumulate the previous one's overrides.
    """
    if name not in _PRESETS:
        raise ValueError(f"unknown theme {name!r}; choose from {list(_PRESETS)}")
    base = {
        "name": "paper", "base": 8.5, "title": 9.5, "label": 8.0, "tick": 6.8,
        "legend": 6.8, "dpi": 300, "linewidth": 0.8, "point_scale": 1.0,
        "text": TEXT, "text_sub": TEXT_SUB, "text_cap": TEXT_CAP,
        "spine": SPINE, "grid": GRID, "bg_soft": BG_SOFT,
        "figure_bg": "#FFFFFF", "axes_bg": "#FFFFFF", "halo": "#FFFFFF",
        "wash": WASH,
    }
    _THEME.update(base)
    _THEME.update(_PRESETS[name], name=name)
    unknown = set(overrides) - set(_THEME)
    if unknown:
        raise ValueError(f"unknown theme keys {sorted(unknown)}; "
                         f"valid keys: {sorted(_THEME)}")
    _THEME.update(overrides)


class theme:
    """Use a theme for a block, then restore the previous one.

    ::

        with ps.pl.theme("dark"):
            ps.pl.umap(adata, color="leiden", save="dark.png")
    """

    def __init__(self, name: str = "paper", **overrides: Any) -> None:
        self._name, self._overrides = name, overrides
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> dict:
        self._saved = dict(_THEME)
        set_theme(self._name, **self._overrides)
        return _THEME

    def __exit__(self, *exc: object) -> bool:
        _THEME.clear()
        _THEME.update(self._saved)
        return False


def get_theme() -> dict:
    return dict(_THEME)


def categorical_palette(n: int, washed: bool = True) -> list[str]:
    """Palette for n classes: washed Tol, or golden-angle HSL beyond 22."""
    # Wash comes from the theme: the blend that keeps a palette calm on white
    # makes it muddy on near-black, so `dark` washes less (0.18 vs 0.32) and
    # `print` not at all.
    base = [_wash(c, _c("wash")) for c in _TOL_RAW] if washed else _TOL_RAW
    if n <= len(base):
        return base[:n]
    import colorsys
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        s = 0.42 + 0.16 * ((i % 3) / 2.0)
        lightness = 0.55 + 0.14 * (((i // 3) % 3) / 2.0)
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        out.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return out


def _categorical_colors(labels: np.ndarray, palette: list[str] | None = None):
    cats = sorted(set(labels.tolist()))
    pal = palette if palette is not None else categorical_palette(len(cats))
    cat_to_color = dict(zip(cats, pal, strict=False))
    colors = np.array([cat_to_color[v] for v in labels])
    return colors, cats, cat_to_color


# ---------------------------------------------------------------------------
# Text hierarchy + framing helpers
# ---------------------------------------------------------------------------

def _heading(ax, title: str | None = None, subtitle: str | None = None, lift: float = 0.0):
    """Left-aligned light-weight title with grey subtitle floating above.
    The subtitle is a fixed distance above the axes, in points, so it keeps
    clear of the title whatever size the axes is (a fraction of the axes
    height put it on the title once the canvas grew). ``lift`` raises it, in
    points, for plots that put a y-axis label at the top of the axes, which it
    would otherwise touch."""
    t = _THEME
    if title:
        ax.set_title(title, loc="left", fontsize=t["title"],
                     fontweight="normal", color=_c("text"),
                     pad=(24 + lift) if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 9 + lift), textcoords="offset points",
                    fontsize=t["tick"], color=_c("text_sub"), va="bottom", ha="left")


def _caption(ax, text: str | None):
    """Optional description tag, italic grey, tucked under the axes."""
    if text:
        t = _THEME
        ax.text(0.0, -0.22, text, transform=ax.transAxes,
                fontsize=t["tick"] - 0.4, color=_c("text_cap"), style="italic",
                va="top", ha="left")


def _axis_labels(ax, xlabel: str | None = None, ylabel: str | None = None):
    """Modern placement: y-label flat at the axis top, x-label right-aligned."""
    t = _THEME
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=t["label"], color=_c("text"), labelpad=3,
                      loc="right")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=t["label"], color=_c("text"), rotation=0,
                      ha="left", va="bottom", labelpad=8)
        ax.yaxis.set_label_coords(-0.02, 0.998)


def _apply_axes_style(ax, frame: bool = False, grid: str | None = None):
    # Paint the canvas first so every later draw sits on the themed ground.
    ax.set_facecolor(_c("axes_bg"))
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(_c("figure_bg"))
    t = _THEME
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not frame:
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_c("spine"))
            ax.spines[side].set_linewidth(t["linewidth"] * 0.75)
        ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"), length=2.5,
                       width=t["linewidth"] * 0.6)
        if grid:
            ax.grid(axis=grid, color=_c("grid"), linewidth=0.5)
            ax.set_axisbelow(True)


def _auto_point_size(n_cells: int) -> float:
    return max(1.2, min(60_000.0 / n_cells, 24.0)) * _THEME["point_scale"]


def _density_order(coords: np.ndarray, bins: int = 64) -> np.ndarray:
    """Sparse-first draw order via 2D histogram density (dense lands on top)."""
    try:
        H, xe, ye = np.histogram2d(coords[:, 0], coords[:, 1], bins=bins)
        ix = np.clip(np.searchsorted(xe, coords[:, 0]) - 1, 0, bins - 1)
        iy = np.clip(np.searchsorted(ye, coords[:, 1]) - 1, 0, bins - 1)
        dens = H[ix, iy]
    except Exception:
        dists = np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)
        return np.argsort(dists)
    return np.argsort(dens, kind="stable")


#: Above this many cells an embedding is drawn as an image of binned cells
#: (``render="auto"``): every cell counts, nothing is sampled, and drawing
#: takes seconds where a scatter of tens of millions of points takes minutes
#: and turns into a solid blot.
DENSITY_ABOVE = 1_000_000

#: More groups than this are named on the data, at their centres, instead of
#: in a legend that would run off the figure.
MAX_LEGEND = 40

#: With that many groups, only the largest this many are named, and a name
#: that would overlap one already placed is left out.
MAX_NAMED = 40
#: Names on the data wrap at this many characters, so a long cell-type name
#: takes a few short lines instead of one line wider than the plot.
LABEL_WRAP = 18

#: A column of whole numbers with at most this many distinct values is drawn
#: as groups (cluster labels), not as a gradient.
MAX_INTEGER_GROUPS = 500


def _use_density(render: str, n: int) -> bool:
    if render not in ("auto", "points", "density"):
        raise ValueError(f"render must be 'auto', 'points' or 'density', not {render!r}")
    return render == "density" or (render == "auto" and n > DENSITY_ABOVE)


def _raster_grid(coords: np.ndarray, bins: int | None = None):
    """Each cell's pixel on a grid covering the embedding, ``bins`` pixels
    along its longer side: (pixel index per cell, (nx, ny), extent)."""
    if bins is None:   # finer grids only once there are cells enough to fill them
        bins = int(np.clip(np.sqrt(len(coords)) / 2, 300, 900))
    lo, hi = np.nanmin(coords, axis=0), np.nanmax(coords, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    lo, hi = lo - 0.01 * span, hi + 0.01 * span
    w, h = hi - lo
    if w >= h:
        nx, ny = bins, max(1, int(round(bins * h / w)))
    else:
        nx, ny = max(1, int(round(bins * w / h))), bins
    ix = np.clip(((coords[:, 0] - lo[0]) / w * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip(((coords[:, 1] - lo[1]) / h * ny).astype(np.int64), 0, ny - 1)
    return iy * nx + ix, (nx, ny), (lo[0], hi[0], lo[1], hi[1])


def _density_alpha(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Opacity per pixel: empty pixels clear, the rest from faint to full on a
    log scale of the cells they hold, so sparse regions stay visible."""
    a = np.zeros(counts.shape)
    nz = counts > 0
    if nz.any():
        a[nz] = 0.5 + 0.5 * np.log1p(counts[nz]) / np.log1p(counts.max())
    return a * alpha


def _show_raster(ax, rgba: np.ndarray, shape, extent, zorder: int = 1) -> None:
    nx, ny = shape
    ax.imshow(rgba.reshape(ny, nx, 4), extent=extent, origin="lower",
              interpolation="nearest", aspect="equal", zorder=zorder)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def _raster_plain(ax, pix, shape, extent, color: str, alpha: float, zorder: int = 1) -> None:
    from matplotlib.colors import to_rgb
    nx, ny = shape
    counts = np.bincount(pix, minlength=nx * ny).astype(float)
    rgba = np.zeros((nx * ny, 4))
    rgba[:, :3] = to_rgb(color)
    rgba[:, 3] = _density_alpha(counts, alpha)
    _show_raster(ax, rgba, shape, extent, zorder)


def _raster_categorical(ax, pix, codes, colors_rgb: np.ndarray, shape, extent,
                        alpha: float, zorder: int = 2) -> None:
    """Each pixel in the colour of the group most of its cells belong to.
    One sort of (pixel, group) keys, so memory stays linear in cells."""
    nx, ny = shape
    npx, ncat = nx * ny, len(colors_rgb)
    keep = codes >= 0
    pix, codes = pix[keep], codes[keep].astype(np.int64)
    key = np.sort(pix * ncat + codes)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    runs = np.diff(np.r_[starts, key.size])
    first = key[starts]
    p, c = first // ncat, first % ncat
    o = np.lexsort((-runs, p))
    winner = np.r_[True, p[o][1:] != p[o][:-1]]
    rgba = np.zeros((npx, 4))
    rgba[p[o][winner], :3] = colors_rgb[c[o][winner]]
    rgba[:, 3] = _density_alpha(np.bincount(pix, minlength=npx).astype(float), alpha)
    _show_raster(ax, rgba, shape, extent, zorder)


def _raster_continuous(ax, pix, values, cmap, norm, shape, extent, alpha: float,
                       zorder: int = 2):
    """Each pixel coloured by the mean value of its cells; returns a mappable
    for the colourbar."""
    from matplotlib.cm import ScalarMappable
    nx, ny = shape
    npx = nx * ny
    ok = np.isfinite(values)
    counts = np.bincount(pix[ok], minlength=npx).astype(float)
    sums = np.bincount(pix[ok], weights=values[ok], minlength=npx)
    mean = np.divide(sums, counts, out=np.zeros(npx), where=counts > 0)
    rgba = np.asarray(cmap(norm(mean)), dtype=float)
    rgba[:, 3] = _density_alpha(counts, alpha)
    _show_raster(ax, rgba, shape, extent, zorder)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def _codes(vals: np.ndarray, series: Any = None):
    """Groups of a label column as (sorted groups, integer code per cell),
    without a Python object per cell: tens of millions of labels in seconds."""
    if isinstance(series, pd.Categorical):
        cat = series
    elif isinstance(getattr(series, "dtype", None), pd.CategoricalDtype):
        cat = series.array
    else:
        cat = pd.Categorical(vals)
    present = np.unique(np.asarray(cat.codes)[np.asarray(cat.codes) >= 0])
    groups = sorted(cat.categories[present].tolist())
    remap = np.full(len(cat.categories), -1, dtype=np.int64)
    remap[present] = [groups.index(g) for g in cat.categories[present].tolist()]
    codes = np.asarray(cat.codes, dtype=np.int64)
    return groups, np.where(codes >= 0, remap[np.maximum(codes, 0)], -1)


def _group_centres(coords: np.ndarray, codes: np.ndarray, n_groups: int, min_group: int = 8):
    """Median position of each group's cells, by one sort of the codes."""
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    bounds = np.searchsorted(sc, np.arange(n_groups + 1))
    out = {}
    for g in range(n_groups):
        a, b = bounds[g], bounds[g + 1]
        if b - a >= min_group:
            sel = order[a:b]
            out[g] = (b - a, float(np.median(coords[sel, 0])), float(np.median(coords[sel, 1])))
    return out


def _group_anchors(coords: np.ndarray, codes: np.ndarray, n_groups: int) -> dict:
    """Where to label each group: the group's cell nearest the median of its cells. The median
    itself can fall in empty space when a group lies in two places; a cell of the group cannot."""
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    out = {}
    for g in range(n_groups):
        sel = order[bounds[g]:bounds[g + 1]]
        if sel.size:
            pts = coords[sel]
            mx, my = np.median(pts[:, 0]), np.median(pts[:, 1])
            i = int(np.argmin((pts[:, 0] - mx) ** 2 + (pts[:, 1] - my) ** 2))
            out[g] = (int(sel.size), float(pts[i, 0]), float(pts[i, 1]))
    return out


def _labels_at(ax, centres: dict, groups: list, fontsize: float | None = None,
               max_named: int | None = None) -> int:
    """Name groups at their centres, largest first, skipping a name that
    would land on one already placed; at most ``max_named`` names. Returns
    how many were named. Long names are wrapped onto several lines."""
    import textwrap

    import matplotlib.patheffects as pe
    t = _THEME
    many = len(centres) > MAX_LEGEND
    fs = fontsize or (t["label"] - 0.6 if many else t["label"] + 0.3)
    limit = max_named if max_named is not None else (MAX_NAMED if many else len(centres))
    # An equal-aspect axes is shrunk to its final box when the figure is
    # drawn, not before. Names measured against the box it has until then
    # looked clear of each other and landed on top of each other once the
    # axes took its shape (a 164-type UMAP came out with names over names).
    ax.apply_aspect()
    # Overlap is judged on each name's drawn extent, so a long cell-type name
    # keeps the room it really takes.
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    placed: list[Any] = []
    for g, (_, x, y) in sorted(centres.items(), key=lambda kv: -kv[1][0]):
        if len(placed) >= limit:
            break
        name = textwrap.fill(str(groups[g]), width=LABEL_WRAP, break_long_words=False)
        txt = ax.text(x, y, name, fontsize=fs, color=_c("text"), ha="center",
                      va="center", zorder=6,
                      path_effects=[pe.withStroke(linewidth=2.2, foreground=_c("halo"),
                                                  alpha=0.9)])
        box = txt.get_window_extent(renderer).expanded(1.06, 1.25)
        inside = frame.x0 <= box.x0 and box.x1 <= frame.x1
        if not inside or any(box.overlaps(o) for o in placed):
            txt.remove()
            continue
        placed.append(box)
    return len(placed)


def _save(fig, save: str | None, dpi: int | None = None):
    if save:
        # Take the background from the theme, not a hardcoded white: saving a
        # dark-theme figure onto a white canvas produced dark text on white
        # with dark-tuned colours, which is the worst of both.
        fig.savefig(save, dpi=dpi or _THEME["dpi"], bbox_inches="tight",
                    facecolor=_c("figure_bg"))


def _legend_out(ax, cat_to_color: dict, cats: list, loc: str | None = "right",
                legend_title: str | None = None, max_entries: int = 40):
    import matplotlib.lines as mlines
    t = _THEME
    if loc in (None, "none", "off") or loc == "on data":
        return
    show = cats[:max_entries]
    handles = [mlines.Line2D([], [], marker="o", ls="",
                             markersize=t["legend"] * 0.85,
                             markerfacecolor=cat_to_color[c],
                             markeredgecolor="none", label=str(c))
               for c in show]
    kw = dict(frameon=False, fontsize=t["legend"], handletextpad=0.3,
              labelspacing=0.35, borderaxespad=0.0, alignment="left")
    if legend_title:
        kw["title"] = legend_title
        kw["title_fontsize"] = t["legend"]
    if loc in ("right", "right margin"):
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  **kw)
    elif loc in ("top", "upper"):
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                  ncol=min(len(show), 6), **kw)
    elif loc in ("bottom", "lower"):
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.10), ncol=min(len(show), 6), **kw)
    else:  # any matplotlib location string
        ax.legend(handles=handles, loc=loc, **kw)


def _on_data_labels(ax, coords: np.ndarray, vals: np.ndarray, cats: list,
                    fontsize: float | None = None, min_group: int = 8):
    """Print category names at cluster centroids with a soft white halo.

    Repeated labels (e.g. several clusters resolved to the same cell type)
    are drawn once, at the centroid of their largest group.
    """
    import matplotlib.patheffects as pe
    t = _THEME
    fs = fontsize or (t["label"] + 0.3)
    best: dict[str, tuple[int, float, float]] = {}
    for c in cats:
        m = vals == c
        if m.sum() < min_group:
            continue
        text = str(c)
        size = int(m.sum())
        if text not in best or size > best[text][0]:
            best[text] = (size, float(np.median(coords[m, 0])),
                          float(np.median(coords[m, 1])))
    for text, (_, x, y) in best.items():
        ax.text(x, y, text, fontsize=fs, color=_c("text"), ha="center", va="center",
                fontweight="normal", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white",
                                            alpha=0.9)])


LABEL_STYLES = ("number", "name", "callout", "none")


def _cluster_labels(ax, centres: dict, cats: list, colours: dict, *, style: str = "auto",
                    styles: dict | None = None, which: list | None = None, n_cells: int = 0,
                    callout_below: float = 0.02, legend_loc: str | None = "right",
                    legend_title: str | None = None, fontsize: float | None = None) -> dict:
    """Label each group of an embedding in one of three ways, chosen per group.

    - ``"number"``: a numbered disc on the group (at its cell nearest the median of its cells), the
      number and name listed in a legend beside the plot.
    - ``"name"``: the name written on the group, with a halo. A name that would land on another label
      is drawn as a callout instead.
    - ``"callout"``: a dot at the group, a leader line, and the name outside the point cloud, spread
      down the left and right so that no two names overlap. Suited to small groups.
    - ``"none"``: no label.

    ``style`` is the default for every group (``"auto"``: numbers for groups holding at least
    ``callout_below`` of the cells, callouts for smaller ones); ``styles`` overrides it per group,
    ``{"Platelet": "callout"}``; ``which`` limits the labels to those groups. Returns the groups drawn
    in each style, and keeps it on ``ax`` as ``ax.polariseq_labels``.
    """
    import textwrap

    import matplotlib.lines as mlines
    import matplotlib.patheffects as pe
    t = _THEME
    fs = fontsize or t["label"]
    styles = {str(k): v for k, v in (styles or {}).items()}
    for k, v in [("style", style)] + list(styles.items()):
        if v not in LABEL_STYLES + ("auto",):
            raise ValueError(f"label style {v!r} for {k}: use one of {LABEL_STYLES + ('auto',)}")
    names = [str(c) for c in cats]
    unknown = [k for k in list(styles) + [str(w) for w in (which or [])] if k not in names]
    if unknown:
        raise KeyError(f"label groups {sorted(set(unknown))} are not groups of this plot; groups: {names}")
    wanted = None if which is None else {str(w) for w in which}
    total = n_cells or sum(v[0] for v in centres.values()) or 1

    def style_of(g: int) -> str:
        name = names[g]
        if wanted is not None and name not in wanted:
            return "none"
        s = styles.get(name, style)
        if s == "auto":
            s = "number" if centres[g][0] >= callout_below * total else "callout"
        return s

    chosen = {g: style_of(g) for g in centres}
    done: dict[str, list[str]] = {"number": [], "name": [], "callout": []}
    halo = [pe.withStroke(linewidth=2.2, foreground=_c("halo"), alpha=0.9)]
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    placed: list[Any] = []

    # numbers, largest group first
    numbered = sorted((g for g, s in chosen.items() if s == "number"), key=lambda g: -centres[g][0])
    handles = []
    for k, g in enumerate(numbered, 1):
        _, x, y = centres[g]
        ax.plot(x, y, "o", ms=fs * 1.45, mfc="white", mec=_c("text"), mew=0.6, zorder=7)
        txt = ax.text(x, y, str(k), fontsize=fs * 0.82, fontweight="bold", ha="center", va="center",
                      color=_c("text"), zorder=8)
        placed.append(txt.get_window_extent(renderer).expanded(1.6, 1.6))
        handles.append(mlines.Line2D([], [], marker="s", ls="", markersize=t["legend"] * 0.9,
                                     markerfacecolor=colours[g], markeredgecolor="none",
                                     label=f"{k}  {names[g]}"))
        done["number"].append(names[g])

    # names on the groups; one that would overlap a label already drawn becomes a callout
    for g in sorted((g for g, s in chosen.items() if s == "name"), key=lambda g: -centres[g][0]):
        _, x, y = centres[g]
        txt = ax.text(x, y, textwrap.fill(names[g], width=LABEL_WRAP, break_long_words=False),
                      fontsize=fs, color=_c("text"), ha="center", va="center", zorder=8, path_effects=halo)
        box = txt.get_window_extent(renderer).expanded(1.06, 1.2)
        if any(box.overlaps(o) for o in placed):
            txt.remove()
            chosen[g] = "callout"
            continue
        placed.append(box)
        done["name"].append(names[g])

    # callouts, spread down each side of the point cloud
    calls = [g for g, s in chosen.items() if s == "callout"]
    if calls:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        mid = (x0 + x1) / 2
        gap = 0.06 * h
        right_edge = x1
        for side in ("left", "right"):
            items = sorted((g for g in calls if (centres[g][1] < mid) == (side == "left")),
                           key=lambda g: -centres[g][2])
            if not items:
                continue
            ys = [centres[g][2] for g in items]
            for i in range(1, len(ys)):
                ys[i] = min(ys[i], ys[i - 1] - gap)
            if ys[-1] < y0 + 0.03 * h:   # pushed below the frame: move the column up, then squeeze
                shift = (y0 + 0.03 * h) - ys[-1]
                ys = [v + shift for v in ys]
                if ys[0] > y1 - 0.03 * h:
                    top, bottom = y1 - 0.03 * h, y0 + 0.03 * h
                    step = (top - bottom) / max(len(ys) - 1, 1)
                    ys = [top - i * step for i in range(len(ys))]
            lx = x0 - 0.04 * w if side == "left" else x1 + 0.04 * w
            for g, ly in zip(items, ys, strict=True):
                _, x, y = centres[g]
                ax.plot([x, lx], [y, ly], color=_c("text_sub"), lw=0.5, zorder=7, clip_on=False)
                # a hollow ring, so the cells of a small group stay visible under it
                ax.plot(x, y, "o", ms=fs * 0.9, mfc="none", mec=_c("text"), mew=0.7, zorder=8, clip_on=False)
                txt = ax.text(lx + (-0.01 if side == "left" else 0.01) * w, ly, names[g], fontsize=fs,
                              color=_c("text"), ha="right" if side == "left" else "left", va="center",
                              zorder=8, clip_on=False)
                bb = txt.get_window_extent(renderer).transformed(ax.transData.inverted())
                right_edge = max(right_edge, bb.x1)
                done["callout"].append(names[g])
        # The callouts sit in the margins, outside the axes, and the axes keep their scale: widening
        # them instead would shrink the plot after the names on it were checked for overlap. A saved
        # figure (bbox_inches="tight") and a notebook include the margins.
        ax.set_xlim(x0, x1)
        legend_x = 1.02 + (right_edge - x1) / w + 0.03 if right_edge > x1 else 1.02

    if handles and legend_loc not in (None, "none", "off", "on data"):
        kw = dict(frameon=False, fontsize=t["legend"], handletextpad=0.4, labelspacing=0.35,
                  borderaxespad=0.0, alignment="left")
        if legend_title:
            kw.update(title=legend_title, title_fontsize=t["legend"])
        if legend_loc in ("right", "right margin"):
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(legend_x if calls else 1.02, 1.0), **kw)
        elif legend_loc in ("bottom", "lower"):
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                      ncol=min(len(handles), 3), **kw)
        else:
            ax.legend(handles=handles, loc=legend_loc, **kw)
    ax.polariseq_labels = done
    return done


# ---------------------------------------------------------------------------
# Embedding plots (UMAP / PCA)
# ---------------------------------------------------------------------------

def _embedding(
    adata: AnnData,
    basis: str,
    color: str | None = None,
    ax: Any | None = None,
    size: float | None = None,
    title: str | None = None,
    cmap: str | None = None,
    show: bool = True,
    save: str | None = None,
    legend_loc: str | None = "right",
    *,
    subtitle: str | None = None,
    caption: str | None = None,
    legend_title: str | None = None,
    palette: list[str] | None = None,
    alpha: float = 0.9,
    frame: bool = False,
    colorbar: bool = True,
    on_data: bool | str = False,
    groups: str | list[str] | None = None,
    figsize: tuple | None = None,
    dpi: int | None = None,
    render: str = "auto",
    labels: str | None = None,
    label_style: dict | None = None,
    label_groups: list | None = None,
    callout_below: float = 0.02,
) -> Any:
    import matplotlib.pyplot as plt

    if labels is not None and labels not in LABEL_STYLES + ("auto",):
        raise ValueError(f"labels={labels!r}: use one of {LABEL_STYLES + ('auto',)}")
    if basis not in adata.obsm:
        raise KeyError(f"basis {basis!r} not in obsm; run the analysis first")
    coords = np.asarray(adata.obsm[basis][:, :2], dtype=float)
    n = len(coords)
    t = _THEME

    own_fig = ax is None
    if own_fig:
        w = 4.4 if legend_loc in (None, "none", "off", "on data") else 4.9
        size_in = (w, 3.7)
        # Many groups are named on the data, not in a legend, and the names
        # need room: a square canvas, where an embedding is drawn at its own
        # aspect and a wide one would leave the sides empty.
        if isinstance(color, str) and color in adata.obs:
            col = adata.obs[color]
            if not isinstance(col, pd.Series):   # an out-of-core result is a plain array
                col = pd.Series(np.asarray(col))
            if not pd.api.types.is_float_dtype(col.dtype) and col.nunique() > MAX_LEGEND:
                size_in = (6.6, 6.6)
        fig, ax = plt.subplots(figsize=figsize or size_in, dpi=dpi or t["dpi"])
    else:
        fig = ax.figure

    _apply_axes_style(ax, frame=frame)
    ax.set_aspect("equal", adjustable="box")
    dense = _use_density(render, n)
    if dense:
        pix, shape, extent = _raster_grid(coords)
    else:
        order = _density_order(coords)
    s = size if size is not None else _auto_point_size(n)
    label_text = color
    named_note = ""

    if color is None and dense:
        _raster_plain(ax, pix, shape, extent, "#8C96A6", alpha)
    elif color is None:
        ax.scatter(coords[order, 0], coords[order, 1], s=s, c="#C4CBD6",
                   edgecolors="none", rasterized=n > 20_000, alpha=alpha * 0.9)
    elif color in adata.obs:
        series = adata.obs[color]
        vals = np.asarray(series)
        # A categorical column is categorical however its labels happen to
        # look. Deciding from the values alone treated Leiden clusters as a
        # gradient the moment a dataset produced more than 30 of them, because
        # the labels cast to float cleanly: an 800K-cell run with 34 clusters
        # came out as a colourbar from 0 to 33. Ask the dtype first.
        if isinstance(getattr(series, "dtype", None), pd.CategoricalDtype):
            continuous = False
        elif np.issubdtype(np.asarray(vals).dtype, np.integer) and \
                len(np.unique(vals)) <= MAX_INTEGER_GROUPS:
            # Whole-number labels with a modest number of values are groups:
            # an out-of-core run hands back its 66 Leiden clusters as plain
            # integers, which the test below would have drawn as a gradient.
            continuous = False
        else:
            try:
                vv = vals.astype(float)
                continuous = np.isfinite(vv).all() and len(np.unique(vv)) > 30
            except (TypeError, ValueError):
                continuous = False
        if continuous:
            from matplotlib.colors import Normalize
            cm_x = plt.get_cmap(cmap) if isinstance(cmap, str) else (cmap or _expr_cmap())
            norm = Normalize(vmin=np.nanmin(vv), vmax=np.nanmax(vv))
            if dense:
                sc = _raster_continuous(ax, pix, vv, cm_x, norm, shape, extent, alpha)
            else:
                sc = ax.scatter(coords[order, 0], coords[order, 1], s=s,
                                c=vv[order], cmap=cm_x, norm=norm,
                                edgecolors="none", rasterized=n > 20_000,
                                alpha=alpha)
            if colorbar:
                cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02,
                                  shrink=0.55, aspect=18)
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
                cb.set_label(str(color), fontsize=t["tick"], color=_c("text_sub"))
        elif dense:
            from matplotlib.colors import to_rgb
            cats, codes = _codes(vals, series)
            pal = palette if palette is not None else categorical_palette(len(cats))
            c2c = dict(zip(cats, pal, strict=False))
            rgb = np.array([to_rgb(c2c[c]) for c in cats])
            if groups is not None:
                wanted = {str(g) for g in ([groups] if isinstance(groups, str) else groups)}
                unknown = wanted - {str(c) for c in cats}
                if unknown:
                    raise KeyError(f"groups {sorted(unknown)} not found in {color!r}; "
                                   f"available: {[str(c) for c in cats]}")
                keep = np.array([str(c) in wanted for c in cats])
                sel = (codes >= 0) & keep[np.maximum(codes, 0)]
                _raster_plain(ax, pix[~sel], shape, extent, _c("bg_soft"), alpha * 0.8)
                _raster_categorical(ax, pix[sel], codes[sel], rgb, shape, extent, alpha)
                cats_shown = [c for c in cats if str(c) in wanted]
            else:
                _raster_categorical(ax, pix, codes, rgb, shape, extent, alpha)
                cats_shown = cats
            if labels is not None:
                centres = _group_anchors(coords, codes, len(cats))
                shown = {cats.index(c) for c in cats_shown}
                _cluster_labels(ax, {g: v for g, v in centres.items() if g in shown}, cats,
                                {i: c2c[c] for i, c in enumerate(cats)}, style=labels,
                                styles=label_style, which=label_groups, n_cells=n,
                                callout_below=callout_below, legend_loc=legend_loc,
                                legend_title=legend_title or color)
            elif on_data or len(cats_shown) > MAX_LEGEND:
                centres = _group_centres(coords, codes, len(cats))
                shown = {cats.index(c) for c in cats_shown}
                named = _labels_at(ax, {g: v for g, v in centres.items() if g in shown}, cats)
                if named < len(cats_shown):
                    named_note = f"; {named} of {len(cats_shown)} groups named"
            else:
                _legend_out(ax, {c: c2c[c] for c in cats_shown}, cats_shown, legend_loc,
                            legend_title or color)
        else:
            colors, cats, c2c = _categorical_colors(vals, palette)
            if groups is not None:
                # Highlight a subset against the rest. The context cells stay
                # on the plot in a flat neutral rather than being dropped:
                # a cluster shown alone loses the shape of the manifold it
                # sits in, which is most of what the reader is judging.
                wanted = {str(g) for g in (
                    [groups] if isinstance(groups, str) else groups)}
                unknown = wanted - {str(c) for c in cats}
                if unknown:
                    raise KeyError(
                        f"groups {sorted(unknown)} not found in {color!r}; "
                        f"available: {[str(c) for c in cats]}"
                    )
                sel = np.array([str(v) in wanted for v in vals])
                rest = ~sel
                if rest.any():
                    ax.scatter(coords[rest, 0], coords[rest, 1], s=s,
                               c=_c("bg_soft"), edgecolors="none",
                               rasterized=n > 20_000, alpha=alpha * 0.55,
                               zorder=1)
                sel_order = [i for i in order if sel[i]]
                ax.scatter(coords[sel_order, 0], coords[sel_order, 1], s=s,
                           c=colors[sel_order], edgecolors="none",
                           rasterized=n > 20_000, alpha=alpha, zorder=2)
                cats = [c for c in cats if str(c) in wanted]
                c2c = {k: v for k, v in c2c.items() if str(k) in wanted}
            else:
                ax.scatter(coords[order, 0], coords[order, 1], s=s,
                           c=colors[order], edgecolors="none",
                           rasterized=n > 20_000, alpha=alpha)
            if labels is not None:
                gcats, gcodes = _codes(vals, series)
                centres = _group_anchors(coords, gcodes, len(gcats))
                keep = {str(c) for c in cats}
                lookup = {str(k): v for k, v in c2c.items()}
                _cluster_labels(ax, {g: v for g, v in centres.items() if str(gcats[g]) in keep}, gcats,
                                {i: lookup[str(c)] for i, c in enumerate(gcats) if str(c) in lookup},
                                style=labels, styles=label_style, which=label_groups, n_cells=n,
                                callout_below=callout_below, legend_loc=legend_loc,
                                legend_title=legend_title or color)
            elif len(cats) > MAX_LEGEND:
                gcats, gcodes = _codes(vals, series)
                named = _labels_at(ax, _group_centres(coords, gcodes, len(gcats)), gcats)
                if named < len(gcats):
                    named_note = f"; {named} of {len(gcats)} groups named"
            elif on_data:
                _on_data_labels(ax, coords, vals, cats)
            else:
                _legend_out(ax, c2c, cats, legend_loc, legend_title or color)
    elif color in list(adata.var_names):
        gi = list(adata.var_names).index(color)
        col = np.asarray(adata.X[:, gi].todense()).flatten() \
            if hasattr(adata.X, "todense") else np.asarray(adata.X[:, gi])
        from matplotlib.colors import Normalize
        vmax = np.percentile(col, 98) if col.max() > 0 else col.max()
        cm_x = plt.get_cmap(cmap) if isinstance(cmap, str) else (cmap or _expr_cmap())
        norm = Normalize(vmin=0, vmax=max(vmax, 1e-9))
        if dense:
            sc = _raster_continuous(ax, pix, np.asarray(col, dtype=float), cm_x, norm,
                                    shape, extent, alpha)
        else:
            sc = ax.scatter(coords[order, 0], coords[order, 1], s=s, c=col[order],
                            cmap=cm_x, norm=norm, edgecolors="none",
                            rasterized=n > 20_000, alpha=alpha)
        if colorbar:
            cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02,
                              shrink=0.55, aspect=18)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
            cb.set_label(str(color), fontsize=t["tick"], color=_c("text_sub"))
    else:
        hint = ""
        if adata.uns.get("hvg"):
            hint = (
                f" This object was subset to {adata.n_vars:,} "
                f"highly-variable genes, so most genes are no longer in it — "
                f"pass subset=False to highly_variable_genes() to keep them."
            )
        raise KeyError(
            f"color {color!r} is neither a column of .obs nor a gene in "
            f".var_names.{hint}"
        )

    auto_sub = subtitle if subtitle is not None else (
        f"{n:,} cells" + (" as density" if dense else "") + named_note
        if title is not None or dense or named_note else None)
    _heading(ax, title if title is not None else (label_text or basis), auto_sub)
    _caption(ax, caption)
    _save(fig, save, dpi)
    if show and own_fig:
        plt.show()
    return ax


def _resolve_markers(markers, kw, fn: str) -> list[str]:
    """Accept scanpy's `var_names=` as well as our `markers=`.

    People arrive here with scanpy muscle memory, and a `TypeError` about an
    unexpected keyword is a poor way to learn that one library calls the same
    argument something else.
    """
    alias = kw.pop("var_names", None)
    if markers is None and alias is None:
        raise TypeError(f"{fn}() needs the genes to plot, as `markers=[...]`")
    if markers is not None and alias is not None:
        raise TypeError(f"{fn}() got both `markers` and `var_names`; use one")
    chosen = markers if markers is not None else alias
    return [chosen] if isinstance(chosen, str) else list(chosen)


def embedding(
    adata: AnnData, basis: str = "X_umap", color: str | list[str] | None = None,
    **kw,
) -> Any:
    """Scatter any 2-D embedding in `obsm`.

    `umap` and `pca` are thin wrappers over this; use it directly for a basis
    they do not cover (`X_tsne`, a custom projection). The leading `X_` is
    optional: ``basis="umap"`` and ``basis="X_umap"`` both work.
    """
    if basis not in adata.obsm and f"X_{basis}" in adata.obsm:
        basis = f"X_{basis}"
    split_by = kw.pop("split_by", None)
    if split_by is not None:
        return _split_embedding(adata, basis, color, split_by, **kw)
    if isinstance(color, (list, tuple)):
        return _multi_embedding(adata, basis, list(color), **kw)
    return _embedding(adata, basis, color, **kw)


def _split_embedding(adata, basis, color, split_by, *, ncols=None, figsize=None, dpi=None,
                     show=True, save=None, size=None, palette=None, **kw):
    """One panel per level of ``split_by`` (a condition, a donor): that
    level's cells coloured, every other cell in soft grey behind them, the
    colours shared by all panels."""
    import matplotlib.pyplot as plt
    t = _THEME
    coords = np.asarray(adata.obsm[basis][:, :2], dtype=float)
    levels = _levels(adata.obs[split_by])
    sarr = np.asarray(adata.obs[split_by])
    fig, axes = _panel_grid(len(levels), ncols or len(levels), figsize, dpi,
                            aspect=_data_aspect(coords))
    s = size if size is not None else _auto_point_size(len(coords))
    cats = c2c = None
    if color is not None and color in adata.obs:
        colors, cats, c2c = _categorical_colors(np.asarray(adata.obs[color]), palette)
    elif color is not None:
        vals = _values(adata, [color])[color]
        vmax = np.percentile(vals, 98) if vals.max() > 0 else 1.0
        from matplotlib.colors import Normalize
        norm, cm_x = Normalize(vmin=0, vmax=max(vmax, 1e-9)), _expr_cmap()
    order = _density_order(coords)
    for ax, lev in zip(axes, levels, strict=False):
        _apply_axes_style(ax, frame=False)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(coords[:, 0], coords[:, 1], s=s, c=_c("bg_soft"), edgecolors="none",
                   rasterized=len(coords) > 20_000, alpha=0.5, zorder=1)
        idx = [i for i in order if sarr[i] == lev]
        if color is None:
            fg = ACCENT
        elif cats is not None:
            fg = colors[idx]
        else:
            fg = cm_x(norm(vals[idx]))
        ax.scatter(coords[idx, 0], coords[idx, 1], s=s, c=fg, edgecolors="none",
                   rasterized=len(idx) > 20_000, alpha=0.9, zorder=2)
        ax.set_title(f"{lev}", loc="left", fontsize=t["tick"] + 1.5, color=_c("text"), pad=4)
        ax.text(0.0, -0.04, f"{len(idx):,} cells", transform=ax.transAxes, fontsize=t["tick"] - 0.5,
                color=_c("text_sub"), va="top")
    if cats is not None:
        _legend_out(axes[len(levels) - 1], c2c, cats, kw.get("legend_loc", "right"),
                    kw.get("legend_title", color))
    fig.tight_layout()
    _save(fig, save, dpi)
    if show:
        plt.show()
    return axes


def _expr_cmap():
    """The expression ramp for the active theme."""
    return EXPR_CMAP_DARK if _THEME["name"] == "dark" else EXPR_CMAP


def _data_aspect(coords: np.ndarray) -> float:
    """Width / height of the point cloud, ignoring extreme outliers.

    Percentiles rather than min/max: a handful of stray cells far from the
    manifold would otherwise set the shape of every panel.
    """
    lo = np.percentile(coords, 1, axis=0)
    hi = np.percentile(coords, 99, axis=0)
    w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    return w / h if h > 0 else 1.0


def _panel_grid(n: int, ncols: int | None, figsize, dpi, aspect: float | None = None):
    """A figure sized for `n` panels, and the flattened axes to fill.

    `aspect` is the width/height of the *data*. Embeddings are drawn with
    equal aspect (distances mean something), so a panel whose shape does not
    match the data's leaves the difference as blank margin — a wide, short
    UMAP in a near-square panel wastes most of the canvas vertically. Sizing
    the panel to the data removes it.
    """
    import matplotlib.pyplot as plt

    ncols = ncols or min(n, 3 if n > 4 else 2 if n > 1 else 1)
    nrows = -(-n // ncols)
    if figsize is None and aspect:
        # Clamp: a pathological aspect should not produce a 40-inch figure.
        a = min(max(aspect, 0.45), 2.6)
        panel_w = 4.0
        w, h = panel_w * ncols, (panel_w / a) * nrows
    else:
        w, h = figsize or (4.3 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h),
                             dpi=dpi or _THEME["dpi"], squeeze=False)
    fig.patch.set_facecolor(_c("figure_bg"))
    flat = [a for row in axes for a in row]
    for extra in flat[n:]:
        extra.set_visible(False)
    return fig, flat[:n]


def _multi_embedding(adata, basis, colors, *, ncols=None, figsize=None,
                     dpi=None, show=True, save=None, **kw):
    """One panel per entry in `color`, on a shared grid.

    Asking for several colourings of one embedding is the most common thing
    anyone does with a UMAP — cluster labels beside three marker genes — and
    doing it by hand means building the grid, threading `ax=` through, and
    remembering to turn off the extra axes. Passing a list does it here.
    """
    import matplotlib.pyplot as plt

    coords = np.asarray(adata.obsm[basis][:, :2], dtype=float)
    fig, axes = _panel_grid(len(colors), ncols, figsize, dpi,
                            aspect=_data_aspect(coords))
    for ax, col in zip(axes, colors, strict=True):
        _embedding(adata, basis, col, ax=ax, show=False, **kw)
    fig.tight_layout()
    _save(fig, save, dpi)
    if show:
        plt.show()
    return axes


def umap(
    adata: AnnData, color: str | None = None, ax: Any | None = None,
    size: float | None = None, title: str | None = None, cmap: str | None = None,
    show: bool = True, save: str | None = None, legend_loc: str | None = "right",
    **kw,
) -> Any:
    """UMAP scatter — calm palette, density-ordered, optional on-data labels.

    `color` accepts an obs column, a gene name, or a **list of either**, in
    which case a grid of panels is drawn (`ncols=` to control the shape)::

        ps.pl.umap(adata, color=["leiden", "CD3E", "LYZ", "MS4A1"])

    Extra attributes: ``subtitle, caption, legend_title, palette, alpha,
    frame, colorbar, on_data (print category names on clusters), groups
    (highlight a subset, grey the rest), ncols, figsize, dpi``.

    Labelling groups (``labels=``), chosen per group::

        ps.pl.umap(adata, color="cell_type", labels="number")     # numbered discs, names in the legend
        ps.pl.umap(adata, color="cell_type", labels="name")       # names on the groups
        ps.pl.umap(adata, color="cell_type", labels="callout")    # names outside, joined by lines
        ps.pl.umap(adata, color="cell_type", labels="auto",       # numbers, callouts for small groups
                   label_style={"Platelet": "callout", "B cell": "name"},  # per-group choice
                   label_groups=["B cell", "CD4 T cell", "Platelet"])      # only these labelled

    ``callout_below`` sets the size (share of cells) under which ``"auto"`` uses a callout.
    Auto-annotation workflow::

        ps.tl.auto_annotate(adata)                # writes obs['cell_type']
        ps.pl.umap(adata, color='cell_type', on_data=True)
    """
    split_by = kw.pop("split_by", None)
    if split_by is not None:
        return _split_embedding(adata, "X_umap", color, split_by, show=show, save=save,
                                size=size, **kw)
    if isinstance(color, (list, tuple)):
        if ax is not None:
            raise ValueError(
                "pass a single `color` when supplying `ax`; a list of colours "
                "builds its own grid of panels and cannot draw into one axis"
            )
        return _multi_embedding(
            adata, "X_umap", list(color), show=show, save=save, size=size,
            title=title, cmap=cmap, legend_loc=legend_loc, **kw)
    return _embedding(adata, "X_umap", color, ax, size, title, cmap, show,
                      save, legend_loc, **kw)


def pca(
    adata: AnnData, color: str | None = None, ax: Any | None = None,
    size: float | None = None, title: str | None = None, cmap: str | None = None,
    show: bool = True, save: str | None = None,
    legend_loc: str | None = "right", **kw,
) -> Any:
    """PCA scatter (PC1 vs PC2). Same extra attributes as :func:`umap`."""
    return _embedding(adata, "X_pca", color, ax, size, title, cmap, show,
                      save, legend_loc, **kw)


def pca_variance_ratio(adata: AnnData, ax: Any | None = None, show: bool = True,
                       save: str | None = None, n_comps: int = 30, **kw) -> Any:
    """Variance explained per PC + cumulative curve on one calm axis."""
    import matplotlib.pyplot as plt
    t = _THEME
    vr = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)[:n_comps]
    x = np.arange(1, len(vr) + 1)
    cum = np.cumsum(vr) * 100.0

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=kw.pop("dpi", t["dpi"]))
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=True, grid="y")
    ax.bar(x, vr * 100.0, color=ACCENT, alpha=0.85, width=0.78, edgecolor="none")
    ax.plot(x, cum, "o-", color=ACCENT_2, markersize=2.4, linewidth=1.1)
    ax.set_ylim(0, 105)
    _axis_labels(ax, xlabel="principal component",
                 ylabel="variance explained (%)")
    _heading(ax, kw.pop("title", "PCA variance explained"),
             kw.pop("subtitle",
                    f"top {len(vr)} PCs carry {cum[-1]:.0f}% of variance"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def _values_for(adata: AnnData, key: str) -> np.ndarray:
    """Per-cell values for `key`, from `obs` or from gene expression.

    scRNA-seq plots are asked for both kinds interchangeably — `n_genes` is an
    obs column, `CD3E` is a gene — and a plotting function that only knows
    about `obs` cannot draw the plot people actually want (a violin of marker
    expression per cluster). `obs` wins on a name collision, matching scanpy.
    """
    if key in adata.obs:
        return np.asarray(adata.obs[key], dtype=float)

    return _gene_values(adata, [key])[:, 0]


def _gene_values(adata: AnnData, genes: list[str]) -> np.ndarray:
    """Every cell's value of each gene (cells × genes), reading only those
    genes: a column gather in the Rust core in memory, one streamed pass out
    of core."""
    names = list(map(str, adata.var_names))
    where = {g: i for i, g in enumerate(names)}
    missing = [g for g in genes if str(g) not in where]
    if missing:
        raise KeyError(
            f"{missing[0]!r} is neither a column of .obs nor a gene in .var_names. "
            f"Available obs columns: {sorted(adata.obs)}"
        )
    # a gene asked for twice (two clusters sharing a top marker) is read once
    idx, inverse = np.unique(np.array([where[str(g)] for g in genes], dtype=np.int64),
                             return_inverse=True)
    x = adata.X
    if hasattr(x, "column_values"):  # out of core: one pass for all genes
        vals = np.asarray(x.column_values([int(i) for i in idx]), dtype=float)
    elif hasattr(x, "to_scipy"):  # SparseMatrix: gather the columns first
        vals = np.asarray(x[:, idx].to_scipy().toarray(), dtype=float)
    elif hasattr(x, "toarray"):  # scipy sparse
        vals = np.asarray(x[:, idx].toarray(), dtype=float)
    else:
        vals = np.asarray(x, dtype=float)[:, idx]
    return vals[:, inverse.ravel()]


def _values(adata: AnnData, keys: list[str]) -> dict[str, np.ndarray]:
    """Per-cell values of obs columns and genes, the genes read together."""
    genes = [k for k in keys if k not in adata.obs]
    got = dict(zip(genes, _gene_values(adata, genes).T, strict=True)) if genes else {}
    return {k: (np.asarray(adata.obs[k], dtype=float) if k in adata.obs else got[k])
            for k in keys}


def _levels(values: Any) -> list:
    """The categories of a label column, in their own order."""
    if isinstance(values, pd.Categorical) or isinstance(
            getattr(values, "dtype", None), pd.CategoricalDtype):
        return [c for c in pd.Categorical(values).categories if (np.asarray(values) == c).any()]
    vals = np.asarray(values)
    return sorted(set(vals.tolist()), key=lambda g: (str(type(g)), g))


def _group_order(values: Any) -> list:
    """Groups in a reader's order: a categorical's own order; otherwise
    numbers (and numeric labels such as clusters "0" to "12") numerically,
    then text alphabetically."""
    if isinstance(values, pd.Categorical) or isinstance(
            getattr(values, "dtype", None), pd.CategoricalDtype):
        return _levels(values)

    def key(g):
        try:
            return (0, float(g), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(g))
    return sorted(set(np.asarray(values).tolist()), key=key)


def _place_labels(ax, points: list[tuple[float, float, str]], fontsize: float, color) -> None:
    """Name points without overlapping labels: each label takes the first of a
    few offsets around its point whose box is free, with a leader line when
    it had to move away."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [(4, 2), (4, -9), (-4, 2), (-4, -9), (8, 12), (8, -18), (-8, 12), (-8, -18),
               (14, 22), (14, -28), (-14, 22), (-14, -28), (20, 34), (-20, 34)]
    for x, y, text in points:
        for k, (dx, dy) in enumerate(offsets):
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=color,
                            ha="left" if dx >= 0 else "right",
                            arrowprops=(dict(arrowstyle="-", color=color, lw=0.4,
                                             shrinkA=0, shrinkB=2) if k >= 4 else None))
            box = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            t.remove()


def _kde(v: np.ndarray, n_grid: int = 96, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian kernel density over the data's range (Silverman's rule)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) > max_points:
        v = np.random.default_rng(0).choice(v, max_points, replace=False)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.array([lo - 0.02, lo + 0.02]), np.array([1.0, 1.0])
    sd = v.std()
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = 1.06 * spread * len(v) ** (-0.2) or (hi - lo) / 50
    grid = np.linspace(lo, hi, n_grid)
    dens = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(axis=1)
    return grid, dens / dens.max()


def violin(
    adata: AnnData, keys: list[str], groupby: str | None = None,
    ax: Any | None = None, show: bool = True, save: str | None = None,
    *, split_by: str | None = None, **kw,
) -> Any:
    """Soft violins with median dots; calm palette; rich text attributes.

    With ``split_by`` (for example ``"condition"``) each group's violin is
    split by that column: two conditions as the left and right halves of one
    violin, more as violins side by side, so a gene's change between
    conditions reads within each cell type.
    """
    if split_by is not None:
        if groupby is None:
            raise ValueError("split_by needs a groupby")
        return _split_violin(adata, [keys] if isinstance(keys, str) else list(keys), groupby,
                             split_by, ax=ax, show=show, save=save, **kw)
    import matplotlib.pyplot as plt
    t = _THEME
    title = kw.pop("title", None)
    subtitle = kw.pop("subtitle", None)
    caption = kw.pop("caption", None)
    palette = kw.pop("palette", None)

    if groupby is not None:
        groups = sorted(set(np.asarray(adata.obs[groupby]).tolist()))
        n_keys = len(keys)
        own = ax is None
        if own:
            fig, axes = plt.subplots(
                1, n_keys, figsize=(max(n_keys * 2.1, 3.0), 3.0), dpi=t["dpi"])
            axes = np.atleast_1d(axes)
        else:
            fig, axes = ax.figure, np.atleast_1d(ax)
        keys = keys[: len(axes)]
        garr = np.asarray(adata.obs[groupby])
        pal = palette or categorical_palette(len(groups))
        for i, key in enumerate(keys):
            vals = _values_for(adata, key)
            data = [vals[garr == g] for g in groups]
            v = axes[i].violinplot(data, positions=range(len(groups)),
                                   showextrema=False, widths=0.75)
            for j, pc in enumerate(v["bodies"]):
                pc.set_facecolor(pal[j % len(pal)])
                pc.set_alpha(0.55)
                pc.set_edgecolor("none")
            qmed = [np.median(d) for d in data]
            q1 = [np.percentile(d, 25) for d in data]
            axes[i].vlines(range(len(groups)), q1, qmed, color="#FFFFFF",
                           linewidth=1.0, alpha=0.9)
            axes[i].scatter(range(len(groups)), qmed, s=13, color="white",
                            zorder=3, edgecolor=_c("text"), linewidth=0.7)
            _apply_axes_style(axes[i], frame=True, grid="y")
            axes[i].set_xticks(range(len(groups)))
            rot = 45 if len(groups) > 12 else 0
            axes[i].set_xticklabels([str(g) for g in groups],
                                    fontsize=t["tick"], rotation=rot,
                                    ha="right" if rot else "center")
            axes[i].set_xlabel(str(groupby), fontsize=t["tick"], color=_c("text_sub"))
            axes[i].set_ylabel("")
        _axis_labels(axes[0], ylabel=str(keys[0]))
        _heading(axes[0], title or " / ".join(keys), subtitle)
    else:
        n_keys = len(keys)
        own = ax is None
        if own:
            fig, axes = plt.subplots(1, n_keys, figsize=(n_keys * 1.9, 2.7),
                                     dpi=t["dpi"])
            axes = np.atleast_1d(axes)
        else:
            fig, axes = ax.figure, np.atleast_1d(ax)
        keys = keys[: len(axes)]
        for i, key in enumerate(keys):
            vals = _values_for(adata, key)
            v = axes[i].violinplot([vals], positions=[0], showextrema=False,
                                   widths=0.7)
            for pc in v["bodies"]:
                pc.set_facecolor(ACCENT)
                pc.set_alpha(0.5)
                pc.set_edgecolor("none")
            med = np.median(vals)
            axes[i].scatter([0], [med], s=14, color="white", zorder=3,
                            edgecolor=_c("text"), linewidth=0.7)
            _apply_axes_style(axes[i], frame=True, grid="y")
            axes[i].set_xticks([])
            axes[i].set_ylabel("")
        _axis_labels(axes[0], ylabel=str(keys[0]))
        _heading(axes[0], title or "distributions", subtitle)
    fig.tight_layout()
    _caption(axes[0], caption)
    _save(fig, save)
    if show and own:
        plt.show()
    return axes[0]


def _split_violin(adata: AnnData, keys: list[str], groupby: str, split_by: str, *,
                  ax: Any | None = None, show: bool = True, save: str | None = None,
                  **kw) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    t = _THEME
    groups = _levels(adata.obs[groupby])
    levels = _levels(adata.obs[split_by])
    garr = np.asarray(adata.obs[groupby])
    sarr = np.asarray(adata.obs[split_by])
    pal = kw.pop("palette", None) or [ACCENT, ACCENT_2, *categorical_palette(max(1, len(levels)))]
    values = _values(adata, keys)
    own = ax is None
    if own:
        fig, axes = plt.subplots(len(keys), 1, dpi=kw.pop("dpi", t["dpi"]), squeeze=False,
                                 figsize=kw.pop("figsize", (max(4.2, 0.62 * len(groups) + 1.2),
                                                            1.9 * len(keys) + 0.6)))
        axes = axes[:, 0]
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)
    two = len(levels) == 2
    slot = 0.84 / (1 if two else len(levels))
    for panel, key in zip(axes, keys, strict=False):
        _apply_axes_style(panel, frame=True, grid="y")
        vals = values[key]
        for gi, g in enumerate(groups):
            for li, lev in enumerate(levels):
                v = vals[(garr == g) & (sarr == lev)]
                if len(v) < 2:
                    continue
                grid, dens = _kde(v)
                if two:
                    side = -1 if li == 0 else 1
                    x0, width = gi, 0.4 * dens
                    panel.fill_betweenx(grid, x0, x0 + side * width, color=pal[li], alpha=0.72,
                                        linewidth=0)
                    xm = x0 + side * 0.07
                else:
                    x0 = gi + (li - (len(levels) - 1) / 2) * slot
                    width = 0.45 * slot * dens
                    panel.fill_betweenx(grid, x0 - width, x0 + width, color=pal[li % len(pal)],
                                        alpha=0.72, linewidth=0)
                    xm = x0
                panel.scatter([xm], [np.median(v)], s=10, color="white", zorder=3,
                              edgecolor=_c("text"), linewidth=0.6)
        panel.set_xticks(range(len(groups)))
        rot = 35 if len(groups) > 5 else 0
        panel.set_xticklabels([str(g) for g in groups], fontsize=t["tick"], rotation=rot,
                              ha="right" if rot else "center")
        panel.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
        panel.set_xlim(-0.6, len(groups) - 0.4)
        _axis_labels(panel, ylabel=str(key))
    for panel in axes[:-1]:
        panel.set_xticklabels([])
    handles = [Patch(color=pal[i % len(pal)], alpha=0.72, label=str(lev))
               for i, lev in enumerate(levels)]
    axes[0].legend(handles=handles, frameon=False, fontsize=t["legend"], ncol=len(levels),
                   loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.0,
                   columnspacing=0.9)
    _heading(axes[0], kw.pop("title", None), kw.pop("subtitle", None))
    fig.tight_layout()
    _caption(axes[-1], kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return axes


def _result_table(result: Any, key: str = "compare_conditions") -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if key not in result.uns:
        raise KeyError(f"{key!r} not in .uns: run ps.tl.compare_conditions first")
    return result.uns[key]["table"]


def volcano(result: Any, group: str | None = None, *, condition: str | None = None,
            n_labels: int = 10, alpha: float = 0.05, min_log2_fold_change: float = 1.0,
            ax: Any | None = None, show: bool = True, save: str | None = None,
            key: str = "compare_conditions", **kw) -> Any:
    """Fold change against significance for one cell type's comparison.

    ``result`` is the table ``ps.tl.compare_conditions`` returns, or the data
    it was run on. Genes past both thresholds (``adj_p_value < alpha`` and
    ``|log2 fold change| >= min_log2_fold_change``) are coloured, up and
    down, and the most significant are named.
    """
    import matplotlib.pyplot as plt
    t = _THEME
    table = _result_table(result, key)
    gcol = table.columns[0]
    if group is None:
        present = table[gcol].unique()
        if len(present) > 1:
            raise ValueError(f"pick a group: {[str(g) for g in present]}")
        group = present[0]
    sel = table[table[gcol].astype(str) == str(group)]
    if condition is not None:
        sel = sel[sel["condition"].astype(str) == str(condition)]
    elif sel["condition"].nunique() > 1:
        raise ValueError(f"pick a condition: {list(sel['condition'].unique())}")
    x = sel["log2_fold_change"].to_numpy(float)
    y = -np.log10(np.clip(sel["adj_p_value"].to_numpy(float), 1e-300, 1.0))
    up = (sel["adj_p_value"] < alpha).to_numpy() & (x >= min_log2_fold_change)
    down = (sel["adj_p_value"] < alpha).to_numpy() & (x <= -min_log2_fold_change)
    rest = ~(up | down)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (4.2, 3.4)), dpi=kw.pop("dpi", t["dpi"]))
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=True, grid="y")
    ax.scatter(x[rest], y[rest], s=6, color=_c("bg_soft"), edgecolors="none", rasterized=True)
    ax.scatter(x[up], y[up], s=9, color=ACCENT_2, edgecolors="none", label=f"up ({up.sum()})")
    ax.scatter(x[down], y[down], s=9, color=ACCENT, edgecolors="none", label=f"down ({down.sum()})")
    for v in (-min_log2_fold_change, min_log2_fold_change):
        ax.axvline(v, color=_c("spine"), linewidth=0.7, linestyle=(0, (3, 2)))
    ax.axhline(-np.log10(alpha), color=_c("spine"), linewidth=0.7, linestyle=(0, (3, 2)))
    top = sel.assign(_y=y)[up | down].sort_values("_y", ascending=False).head(n_labels)
    _place_labels(ax, [(float(r["log2_fold_change"]), float(r["_y"]), str(r["gene"]))
                       for _, r in top.iterrows()], t["tick"] - 0.5, _c("text"))
    ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
    ax.legend(frameon=False, fontsize=t["legend"], loc="upper left", handletextpad=0.2)
    _axis_labels(ax, xlabel="log2 fold change", ylabel="-log10 adjusted p")
    cond = str(sel["condition"].iloc[0]) if len(sel) else ""
    _heading(ax, kw.pop("title", f"{group}"), kw.pop("subtitle", f"{cond}, {len(sel):,} genes tested"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


def fold_changes(result: Any, genes: list[str] | None = None, *, condition: str | None = None,
                 n_genes: int = 6, alpha: float = 0.05, ax: Any | None = None,
                 show: bool = True, save: str | None = None, key: str = "compare_conditions",
                 **kw) -> Any:
    """Which cell types respond: log2 fold changes, genes against cell types.

    ``genes`` defaults to the ``n_genes`` most significant of each cell type.
    A dot marks ``adj_p_value < alpha``; a blank cell, a gene not tested there
    (too little expression in that cell type).
    """
    import matplotlib.pyplot as plt
    t = _THEME
    table = _result_table(result, key)
    gcol = table.columns[0]
    if condition is not None:
        table = table[table["condition"].astype(str) == str(condition)]
    groups = [g for g in _levels(table[gcol]) if (table[gcol] == g).any()]
    if genes is None:
        genes = []
        for g in groups:
            sub = table[(table[gcol] == g) & (table["adj_p_value"] < alpha)]
            for name in sub.sort_values("adj_p_value")["gene"].head(n_genes):
                if name not in genes:
                    genes.append(name)
    lfc = np.full((len(genes), len(groups)), np.nan)
    sig = np.zeros_like(lfc, dtype=bool)
    look = table.set_index([table[gcol].astype(str), "gene"])
    for j, g in enumerate(groups):
        for i, name in enumerate(genes):
            k = (str(g), name)
            if k in look.index:
                row = look.loc[k]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                lfc[i, j] = row["log2_fold_change"]
                sig[i, j] = row["adj_p_value"] < alpha
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (0.55 * len(groups) + 2.2,
                                                           0.24 * len(genes) + 1.4)),
                               dpi=kw.pop("dpi", t["dpi"]))
    else:
        fig = ax.figure
    vmax = float(np.nanpercentile(np.abs(lfc), 98)) if np.isfinite(lfc).any() else 1.0
    im = ax.imshow(np.ma.masked_invalid(lfc), cmap=kw.pop("cmap", DIV_CMAP), vmin=-vmax,
                   vmax=vmax, aspect="auto")
    yy, xx = np.nonzero(sig)
    ax.scatter(xx, yy, s=5, color=_c("text"), edgecolors="none")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], rotation=40, ha="right", fontsize=t["tick"])
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=t["tick"])
    ax.tick_params(length=0, colors=_c("text_sub"))
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=t["tick"] - 1, colors=_c("text_sub"))
    cb.set_label("log2 fold change", fontsize=t["tick"] - 0.5, color=_c("text_sub"))
    cond = str(table["condition"].iloc[0]) if len(table) else ""
    _heading(ax, kw.pop("title", "fold change by cell type"),
             kw.pop("subtitle", f"{cond}; dot: adjusted p < {alpha:g}"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


def scrublet_score_distribution(adata: AnnData, *, log: bool = True, show: bool = True,
                                save: str | None = None, **kw) -> Any:
    """Doublet scores of the cells and of the simulated doublets, per batch.

    The threshold ``ps.pp.scrublet`` chose (dashed) should sit in the valley
    between the simulated doublets' two modes; if it does not, pass one with
    ``ps.pp.scrublet(..., threshold=...)``. Cells are drawn on a log scale,
    where the few doublets would otherwise vanish beside the singlets.
    """
    import matplotlib.pyplot as plt
    t = _THEME
    info = adata.uns.get("scrublet")
    if info is None:
        raise KeyError("run ps.pp.scrublet first")
    batches = info["batches"] if "batches" in info else {"all": info}
    key = info.get("batched_by")
    scores = np.asarray(adata.obs["doublet_score"], dtype=float)
    labels = np.asarray(adata.obs[key]).astype(str) if key else None
    n = len(batches)
    fig, axes = plt.subplots(n, 2, figsize=kw.pop("figsize", (6.2, 2.1 * n + 0.5)),
                             dpi=kw.pop("dpi", t["dpi"]), squeeze=False)
    top = max(float(np.nanmax(scores)),
              max(float(np.max(b["doublet_scores_sim"])) for b in batches.values()))
    bins = np.linspace(0.0, top * 1.02, 60)
    for r, (name, b) in enumerate(batches.items()):
        obs = scores if labels is None else scores[labels == name]
        obs = obs[np.isfinite(obs)]
        sim = np.asarray(b["doublet_scores_sim"], dtype=float)
        thr = b.get("threshold")
        for c, (vals, what, color) in enumerate(((obs, "cells", ACCENT),
                                                 (sim, "simulated doublets", ACCENT_2))):
            ax = axes[r, c]
            _apply_axes_style(ax, frame=True, grid="y")
            ax.hist(vals, bins=bins, color=color, alpha=0.85, density=True, edgecolor="none")
            if log and c == 0:
                ax.set_yscale("log")
            if thr is not None:
                ax.axvline(thr, color=_c("text"), linewidth=0.8, linestyle=(0, (3, 2)))
            ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
            label = what if n == 1 else f"{name}: {what}"
            ax.set_title(label, loc="left", fontsize=t["tick"] + 1, color=_c("text"), pad=4)
            if r == n - 1:
                _axis_labels(ax, xlabel="doublet score")
    _heading(axes[0, 0], kw.pop("title", None), None)
    fig.tight_layout()
    _caption(axes[-1, 0], kw.pop("caption", None))
    _save(fig, save)
    if show:
        plt.show()
    return axes


def highly_variable_genes(adata: AnnData, ax: Any | None = None, show: bool = True,
                          save: str | None = None, **kw) -> Any:
    """Mean–variance relation with HVGs softly highlighted."""
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    t = _THEME
    var = adata.var
    if "means" not in var or "dispersions" not in var:
        raise KeyError("run ps.pp.highly_variable_genes first")
    means = np.asarray(var["means"], dtype=float)
    disp = np.asarray(var["dispersions"], dtype=float)
    hvg = np.asarray(var.get("highly_variable", np.zeros(len(means), bool)), bool)

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=t["dpi"])
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=True)
    ax.scatter(means[~hvg], disp[~hvg], s=3.5, color=_c("bg_soft"), alpha=0.85,
               edgecolors="none", rasterized=True)
    ax.scatter(means[hvg], disp[hvg], s=7, color=ACCENT, alpha=0.75,
               edgecolors="none", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
    h1 = mlines.Line2D([], [], marker="o", ls="", ms=3.5, color=_c("bg_soft"),
                       markeredgecolor=_c("spine"), label="other genes")
    h2 = mlines.Line2D([], [], marker="o", ls="", ms=4.5, color=ACCENT,
                       label=f"HVGs ({int(hvg.sum()):,})")
    ax.legend(handles=[h1, h2], frameon=False, fontsize=t["legend"],
              loc="upper left", handletextpad=0.2, labelspacing=0.3)
    _axis_labels(ax, xlabel="mean expression", ylabel="dispersion")
    _heading(ax, kw.pop("title", "highly variable genes"),
             kw.pop("subtitle",
                    f"{int(hvg.sum()):,} of {len(means):,} selected"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Matrix views
# ---------------------------------------------------------------------------

def dotplot(
    adata: AnnData, markers: list[str] | None = None, groupby: str = "leiden",
    ax: Any | None = None, show: bool = True, save: str | None = None,
    **kw,
) -> Any:
    """Dot plot: size = fraction expressing, colour = mean expression.

    Genes may be given as `markers=` or, for scanpy compatibility,
    `var_names=`.
    """
    markers = _resolve_markers(markers, kw, "dotplot")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = _THEME
    split_by = kw.pop("split_by", None)

    groups = _group_order(adata.obs[groupby])
    valid_markers = [g for g in markers if g in list(adata.var_names)]
    if not valid_markers:
        raise ValueError("none of the requested markers are in var_names "
                         "(after HVG subsetting?) — pass genes that survive filtering")
    garr = np.asarray(adata.obs[groupby])
    vals = _gene_values(adata, valid_markers)
    levels = _levels(adata.obs[split_by]) if split_by is not None else [None]
    sarr = np.asarray(adata.obs[split_by]) if split_by is not None else None

    # One column per group, or per group and level of split_by.
    cols = [(g, lev) for g in groups for lev in levels]
    frac = np.zeros((len(valid_markers), len(cols)))
    mean_expr = np.zeros_like(frac)
    for gj, (g, lev) in enumerate(cols):
        m = garr == g
        if lev is not None:
            m = m & (sarr == lev)
        if m.any():
            frac[:, gj] = (vals[m] > 0).mean(axis=0)
            mean_expr[:, gj] = vals[m].mean(axis=0)

    own = ax is None
    per_group = 0.55 if split_by is None else 0.34 * len(levels) + 0.2
    if own:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (max(len(groups) * per_group + 1.6, 4),
                                                          max(len(valid_markers) * 0.4, 2.5) + 0.6)),
                               dpi=t["dpi"])
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=False)

    cmap = kw.pop("cmap", None) or DOT_CMAP
    norm = Normalize(vmin=0, vmax=max(mean_expr.max(), 1e-9))
    if split_by is None:
        xs, ys = np.meshgrid(range(len(groups)), range(len(valid_markers)))
        ax.scatter(xs.flatten(), ys.flatten(), s=frac.flatten() * 200 + 5,
                   c=cmap(norm(mean_expr.flatten())), edgecolors="white",
                   linewidths=0.5, zorder=3)
    else:
        # The levels side by side within each group, told apart by the
        # colour of their outline.
        from matplotlib.lines import Line2D
        edge = [ACCENT, ACCENT_2, *categorical_palette(max(1, len(levels)))]
        step = 0.72 / len(levels)
        for li, lev in enumerate(levels):
            xs = np.arange(len(groups)) + (li - (len(levels) - 1) / 2) * step
            for gi in range(len(valid_markers)):
                k = [j * len(levels) + li for j in range(len(groups))]
                ax.scatter(xs, np.full(len(groups), gi), s=frac[gi, k] * 150 + 4,
                           c=cmap(norm(mean_expr[gi, k])), edgecolors=edge[li % len(edge)],
                           linewidths=0.9, zorder=3)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, mfc="white",
                                  mec=edge[i % len(edge)], mew=1.1, label=str(lev))
                           for i, lev in enumerate(levels)],
                  frameon=False, fontsize=t["legend"], ncol=len(levels), loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, columnspacing=0.9)
    rotate = len(groups) > 8 or (split_by is not None and max(len(str(g)) for g in groups) > 6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], fontsize=t["tick"],
                       rotation=40 if rotate else 0, ha="right" if rotate else "center")
    ax.set_yticks(range(len(valid_markers)))
    ax.set_yticklabels(valid_markers, fontsize=t["tick"],
                       fontname="DejaVu Sans Mono")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(len(valid_markers) - 0.4, -0.6)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.024, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=t["tick"] - 1, colors=_c("text_sub"))
    cb.set_label("mean expression", fontsize=t["tick"] - 0.5,
                 color=_c("text_sub"), labelpad=2)
    _heading(ax, kw.pop("title", "marker expression"),
             kw.pop("subtitle",
                    "size = fraction expressing · colour = mean expression"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


def heatmap(
    adata: AnnData, markers: list[str] | None = None, groupby: str = "leiden",
    ax: Any | None = None, cmap: Any = None, show: bool = True,
    save: str | None = None, **kw,
) -> Any:
    """Heatmap of z-scored marker expression across groups.

    Genes may be given as `markers=` or, for scanpy compatibility,
    `var_names=`.
    """
    markers = _resolve_markers(markers, kw, "heatmap")
    import matplotlib.pyplot as plt
    t = _THEME

    groups = sorted(set(np.asarray(adata.obs[groupby]).tolist()),
                    key=lambda g: (len(g), g) if isinstance(g, str) else g)
    valid_markers = [g for g in markers if g in list(adata.var_names)]
    if not valid_markers:
        raise ValueError("none of the requested markers are in var_names")
    gene_indices = [list(adata.var_names).index(g) for g in valid_markers]
    garr = np.asarray(adata.obs[groupby])

    expr = np.zeros((len(valid_markers), len(groups)))
    for gi, gidx in enumerate(gene_indices):
        col = np.asarray(adata.X[:, gidx].todense()).flatten() \
            if hasattr(adata.X, "todense") else np.asarray(adata.X[:, gidx])
        for gj, g in enumerate(groups):
            m = garr == g
            expr[gi, gj] = col[m].mean() if m.any() else 0.0
    mu = expr.mean(axis=1, keepdims=True)
    sd = expr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = np.clip((expr - mu) / sd, -2.5, 2.5)

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(max(len(groups) * 0.55, 3.6),
                                        max(len(valid_markers) * 0.32, 2.1)),
                               dpi=t["dpi"])
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=False)
    im = ax.imshow(z, aspect="auto", cmap=cmap or DIV_CMAP, vmin=-2.5, vmax=2.5,
                   interpolation="nearest")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], fontsize=t["tick"],
                       rotation=45 if len(groups) > 8 else 0,
                       ha="right" if len(groups) > 8 else "center")
    ax.set_yticks(range(len(valid_markers)))
    ax.set_yticklabels(valid_markers, fontsize=t["tick"],
                       fontname="DejaVu Sans Mono")
    ax.set_xticks(np.arange(-0.5, len(groups)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(valid_markers)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=t["tick"] - 1, colors=_c("text_sub"))
    cb.set_label("row z-score", fontsize=t["tick"] - 0.5, color=_c("text_sub"))
    _heading(ax, kw.pop("title", "marker heatmap"),
             kw.pop("subtitle", "row-scaled, clipped at ±2.5"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


def rank_genes_groups(
    adata: AnnData, groups: list[str] | None = None, n_genes: int = 10,
    ax: Any | None = None, show: bool = True, save: str | None = None,
    *, ncols: int | None = None, figsize: tuple | None = None,
    dpi: int | None = None, sharey: bool = False, key: str = "rank_genes_groups",
    **kw,
) -> Any:
    """Top marker genes per cluster, ranked, one panel per group.

    Reads what `ps.tl.rank_genes_groups` wrote. The y axis is the
    t-statistic, so panel heights are comparable within a group but not
    necessarily across them — pass ``sharey=True`` if you want them forced
    onto one scale, at the cost of squashing the weaker clusters.

    ::

        ps.tl.rank_genes_groups(adata, groupby="leiden")
        ps.pl.rank_genes_groups(adata, n_genes=8)
    """
    import matplotlib.pyplot as plt

    if key not in adata.uns:
        raise KeyError(
            f"{key!r} not in .uns — run ps.tl.rank_genes_groups(adata, "
            f"groupby=...) first"
        )
    de = adata.uns[key]
    available = list(de)
    try:
        available.sort(key=int)
    except (TypeError, ValueError):
        available.sort()
    picked = [str(g) for g in (groups or available)]
    missing = [g for g in picked if g not in de]
    if missing:
        raise KeyError(f"groups {missing} not in {key!r}; have {available}")

    t = _THEME
    title = kw.pop("title", None)
    subtitle = kw.pop("subtitle", None)
    caption = kw.pop("caption", None)

    own = ax is None
    if own:
        fig, axes = _panel_grid(len(picked), ncols, figsize, dpi)
    else:
        fig, axes = ax.figure, [ax]
        picked = picked[:1]

    pal = categorical_palette(len(available))
    lim = None
    if sharey:
        lim = max(
            float(np.max(np.abs(np.asarray(de[g]["scores"][:n_genes]))))
            for g in picked
        ) * 1.15

    for a, g in zip(axes, picked, strict=True):
        names = [str(x) for x in de[g]["names"][:n_genes]]
        scores = np.asarray(de[g]["scores"][:n_genes], dtype=float)
        y = np.arange(len(names))[::-1]
        colour = pal[available.index(g) % len(pal)]
        a.barh(y, scores, height=0.68, color=colour, edgecolor="none")
        a.set_yticks(y)
        a.set_yticklabels(names, fontsize=t["tick"], color=_c("text"))
        _apply_axes_style(a, frame=True, grid="x")
        a.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
        if lim:
            a.set_xlim(0, lim)
        _heading(a, f"{g}", None)
        a.set_xlabel("t-statistic", fontsize=t["label"], color=_c("text_sub"))

    if own:
        if title:
            fig.suptitle(title, fontsize=t["title"], color=_c("text"))
        fig.tight_layout()
        _caption(axes[-1], caption)
        if subtitle:
            axes[0].set_ylabel(subtitle, fontsize=t["label"],
                               color=_c("text_sub"))
        _save(fig, save, dpi)
        if show:
            plt.show()
    return axes if own else ax


def stacked_bar(
    adata: AnnData, groupby: str = "leiden", color_by: str = "dataset",
    ax: Any | None = None, show: bool = True, save: str | None = None,
    **kw,
) -> Any:
    """Horizontal 100% composition bars with legend below."""
    import matplotlib.pyplot as plt
    t = _THEME

    groups = sorted(set(np.asarray(adata.obs[groupby]).tolist()),
                    key=lambda g: (len(g), g) if isinstance(g, str) else g)
    cats = sorted(set(np.asarray(adata.obs[color_by]).tolist()))
    garr = np.asarray(adata.obs[groupby])
    carr = np.asarray(adata.obs[color_by])
    comp = np.zeros((len(groups), len(cats)))
    for gi, g in enumerate(groups):
        m = garr == g
        for ci, c in enumerate(cats):
            comp[gi, ci] = (carr[m] == c).sum()
    comp = comp / np.maximum(comp.sum(axis=1, keepdims=True), 1) * 100

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, max(len(groups) * 0.30, 1.5)),
                               dpi=t["dpi"])
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=True)
    palette = kw.pop("palette", None) or categorical_palette(len(cats))
    left = np.zeros(len(groups))
    for ci, c in enumerate(cats):
        ax.barh(range(len(groups)), comp[:, ci], left=left, height=0.7,
                color=palette[ci], edgecolor="white", linewidth=0.4,
                label=str(c)[:26])
        left += comp[:, ci]
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([(str(g)[:24] + "...") if len(str(g)) > 26
                        else str(g) for g in groups], fontsize=t["tick"])
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
    ax.invert_yaxis()
    _axis_labels(ax, xlabel="composition (%)")
    ncol = 1 if len(cats) <= 6 else 2
    ax.legend(frameon=False, fontsize=t["legend"],
              title=kw.pop("legend_title", str(color_by)),
              title_fontsize=t["legend"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol,
              handlelength=1.0, handleheight=1.0, columnspacing=1.2,
              alignment="left")
    _heading(ax, kw.pop("title", f"{groupby} composition"),
             kw.pop("subtitle", f"by {color_by}"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


def scatter_qc(adata: AnnData, ax: Any | None = None, show: bool = True,
               save: str | None = None, **kw) -> Any:
    """QC scatter: genes vs counts with density rendering at scale."""
    import matplotlib.pyplot as plt
    t = _THEME

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=t["dpi"])
    else:
        fig = ax.figure
    _apply_axes_style(ax, frame=True)

    n_counts = np.asarray(
        adata.obs.get("n_counts", adata.obs.get("total_counts",
                                                np.zeros(adata.n_obs))),
        dtype=float)
    n_genes = np.asarray(
        adata.obs.get("n_genes", adata.obs.get("n_genes_by_counts",
                                               np.zeros(adata.n_obs))),
        dtype=float)

    if len(n_counts) > 50_000:
        hb = ax.hexbin(n_counts, n_genes, gridsize=60, cmap="Blues",
                       mincnt=1, edgecolors="none", linewidths=0.0)
        cb = fig.colorbar(hb, ax=ax, fraction=0.042, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=t["tick"], colors=_c("text_sub"))
        cb.set_label("cells per bin", fontsize=t["tick"], color=_c("text_sub"))
    else:
        ax.scatter(n_counts, n_genes, s=_auto_point_size(len(n_counts)),
                   color=ACCENT, alpha=0.3, edgecolors="none", rasterized=True)
    _axis_labels(ax, xlabel="counts per cell", ylabel="genes per cell")
    _heading(ax, kw.pop("title", "QC overview"),
             kw.pop("subtitle",
                    f"{len(n_counts):,} cells · median "
                    f"{int(np.nanmedian(n_genes)):,} genes/cell"))
    _caption(ax, kw.pop("caption", None))
    _save(fig, save)
    if show and own:
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Figures that adapt to the size of the data
# ---------------------------------------------------------------------------

_QC_LABELS = {"n_genes": "genes detected per cell", "n_counts": "counts per cell",
              "pct_counts_mt": "mitochondrial counts (%)", "total_counts": "counts per cell",
              "n_genes_by_counts": "genes detected per cell"}


def _compact(v: float, _pos: Any = None) -> str:
    """1,234 -> 1.2k, 3,400,000 -> 3.4M: tick labels that fit at any scale."""
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            x = v / div
            return f"{x:.0f}{suf}" if x == round(x) or abs(x) >= 10 else f"{x:.1f}{suf}"
    return f"{v:g}"


def _plain(v: float, _pos: Any = None) -> str:
    return f"{v:,.0f}" if abs(v) >= 1 else f"{v:g}"


def _log_axis(ax, lo: float, hi: float, axis: str = "x") -> None:
    """A log axis with few, readable labels: 1-2-5 steps over a narrow range,
    powers of ten over a wide one; never overlapping 5000 and 10000."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    decades = np.log10(max(hi, 1e-12) / max(lo, 1e-12))
    subs = (1.0, 2.0, 5.0) if decades <= 2.2 else (1.0,)
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(_compact if hi >= 1e5 else _plain))
    a.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=axis, which="minor", colors=_c("spine"), length=1.6,
                   width=_THEME["linewidth"] * 0.5)


def qc_distributions(adata: AnnData, keys: tuple | list = ("n_genes", "n_counts"), *,
                     thresholds: dict | None = None, before: dict | None = None,
                     bins: int = 64, show: bool = True,
                     save: str | None = None, figsize: tuple | None = None,
                     dpi: int | None = None, **kw) -> Any:
    """Distributions of per-cell quality metrics, from every cell.

    Counts and detected genes span orders of magnitude, so the bins are
    spaced evenly on a log scale (a linear histogram on a log axis collapses
    the bulk of the cells into a few wide blocks). Nothing is sampled: a
    histogram of tens of millions of values is one pass. The median is
    marked, and so is any filter threshold passed in ``thresholds``
    (``{"n_genes": 200}``).

    ``adata`` holds the cells that were kept. To show the ones a threshold
    removed, pass their metrics from before filtering in ``before``
    (``{"n_genes": values, "n_counts": values}``, one value per cell, read
    right after ``calculate_qc_metrics``): they are drawn in grey under the
    kept cells, and the subtitle counts what was removed. Without it a
    threshold line sits at the edge of the data and cuts nothing.
    """
    import matplotlib.pyplot as plt
    t = _THEME
    keys = [keys] if isinstance(keys, str) else list(keys)
    thresholds = thresholds or {}
    fig, axes = plt.subplots(1, len(keys), figsize=figsize or (3.5 * len(keys), 2.5),
                             dpi=dpi or t["dpi"], squeeze=False)
    axes = list(axes[0])
    n_total = n_kept = None
    everyone: dict[str, np.ndarray] = {}
    for ax, key in zip(axes, keys, strict=True):
        v = np.asarray(_values_for(adata, key), dtype=float)
        v = v[np.isfinite(v)]
        n_kept = len(v) if n_kept is None else n_kept
        pos = v[v > 0]
        # Every cell before filtering when given, else the cells kept.
        a = v
        if before is not None and key in before:
            a = np.asarray(before[key], dtype=float)
            a = a[np.isfinite(a)]
        everyone[key] = a
        n_total = len(a) if n_total is None else n_total
        pos_all = a[a > 0]
        _apply_axes_style(ax, frame=True, grid="y")
        if pos_all.size == 0:
            ax.text(0.5, 0.5, "no positive values", transform=ax.transAxes, ha="center",
                    color=_c("text_sub"), fontsize=t["tick"])
            continue
        lo, hi = float(pos_all.min()), float(pos_all.max())
        logx = hi / max(lo, 1e-12) >= 20
        edges = np.geomspace(lo, hi, bins + 1) if logx else np.linspace(lo, hi, bins + 1)
        if pos_all.size > pos.size:
            h_all, _ = np.histogram(pos_all, edges)
            ax.stairs(h_all, edges, fill=True, color=_c("bg_soft"), linewidth=0, zorder=1)
        h, _ = np.histogram(pos, edges)
        ax.stairs(h, edges, fill=True, color=ACCENT, alpha=0.85, linewidth=0, zorder=2)
        if logx:
            _log_axis(ax, lo, hi, "x")
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(_compact))
        med = float(np.median(pos))
        ax.axvline(med, color=_c("text_sub"), linewidth=0.8, linestyle=(0, (2, 2)), zorder=3)
        ax.text(med, 1.0, f" median {_plain(med)}", transform=ax.get_xaxis_transform(),
                fontsize=t["tick"] - 0.3, color=_c("text_sub"), va="top", ha="left")
        if key in thresholds:
            thr = float(thresholds[key])
            ax.axvline(thr, color=ACCENT_2, linewidth=0.9, zorder=3)
            below = int((a < thr).sum())
            x0 = ax.get_xlim()[0]
            if below and x0 < thr:
                # The cells a threshold removes are a sliver of the total, too
                # few to see as bars: shade where they lie and say how many.
                ax.axvspan(x0, thr, color=ACCENT_2, alpha=0.07, linewidth=0, zorder=0)
                centre = float(np.sqrt(x0 * thr)) if logx else (x0 + thr) / 2
                ax.text(centre, 0.62, f"{below:,} cells\nbelow {_plain(thr)}\nremoved",
                        transform=ax.get_xaxis_transform(), fontsize=t["tick"] - 0.3,
                        color=ACCENT_2, va="center", ha="center", linespacing=1.3)
            else:
                ax.text(thr, 0.86, f"kept from {_plain(thr)} ", transform=ax.get_xaxis_transform(),
                        fontsize=t["tick"] - 0.3, color=ACCENT_2, va="top", ha="right")
        _axis_labels(ax, xlabel=_QC_LABELS.get(key, key), ylabel="cells")
    zeros = {k: int((a <= 0).sum()) for k, a in everyone.items()}
    extra = "".join(f", {z:,} with no {k}" for k, z in zeros.items() if z)
    removed = (n_total or 0) - (n_kept or 0)
    if removed > 0:
        what = (f"{n_total:,} cells, every cell counted; {removed:,} removed "
                f"({removed / n_total:.2%}, shaded), {n_kept:,} kept{extra}")
    else:
        what = f"{n_total:,} cells, every cell counted{extra}"
    _heading(axes[0], kw.pop("title", "Quality control"), kw.pop("subtitle", what), lift=9)
    _caption(axes[0], kw.pop("caption", None))
    fig.tight_layout()
    _save(fig, save, dpi)
    if show:
        plt.show()
    return axes


def cluster_sizes(adata: AnnData, key: str = "leiden", *, log: bool | None = None,
                  show: bool = True, save: str | None = None, figsize: tuple | None = None,
                  dpi: int | None = None, **kw) -> Any:
    """Cells per cluster, largest first. The axis turns logarithmic when sizes
    span more than fifty-fold, and cluster names are written out only while
    they fit (at most 40); labels are plain numbers at every scale."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    t = _THEME
    sizes = pd.Series(np.asarray(adata.obs[key])).value_counts()
    k, n = len(sizes), int(sizes.sum())
    fig, ax = plt.subplots(figsize=figsize or (min(9.0, max(4.2, 2.8 + 0.035 * k)), 2.6),
                           dpi=dpi or t["dpi"])
    _apply_axes_style(ax, frame=True, grid="y")
    x = np.arange(k)
    ax.bar(x, sizes.to_numpy(), width=0.86 if k <= 60 else 1.0, color=ACCENT, alpha=0.85,
           edgecolor="none")
    if log if log is not None else sizes.max() / max(sizes.min(), 1) > 50:
        _log_axis(ax, float(max(sizes.min(), 1)), float(sizes.max()), "y")
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    if k <= MAX_LEGEND:
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in sizes.index], fontsize=t["tick"] - 0.5,
                           rotation=90 if k > 15 else 0)
        _axis_labels(ax, xlabel=key, ylabel="cells")
    else:
        ax.set_xticks([])
        _axis_labels(ax, xlabel=f"{k} clusters, largest first", ylabel="cells")
    ax.set_xlim(-0.6, k - 0.4)
    _heading(ax, kw.pop("title", f"{k} clusters"),
             kw.pop("subtitle", f"{n:,} cells; largest {int(sizes.max()):,}, median "
                                f"{int(sizes.median()):,}, smallest {int(sizes.min()):,}"),
             lift=9)
    _caption(ax, kw.pop("caption", None))
    _save(fig, save, dpi)
    if show:
        plt.show()
    return ax


def overview_data(adata: AnnData, *, cluster_key: str = "leiden", qc_before: dict | None = None,
                  thresholds: dict | None = None, bins: int = 64) -> dict:
    """The numbers behind the figures of :func:`overview`, as plain Python values.

    ``qc``: for each metric, the bin edges (evenly spaced on a log scale, as drawn), the number of cells
    in each bin before filtering (when ``qc_before`` is given) and among the cells kept, the threshold
    and how many cells fall below it; ``cluster_sizes``; ``pca_variance_ratio``; and the number of
    cells in the embedding. Written next to the figures as ``figure_data.json``."""
    thresholds = thresholds or {}
    data: dict = {"n_cells": int(adata.n_obs), "qc": {}}
    for key in [k for k in ("n_genes", "n_counts") if k in adata.obs]:
        kept = np.asarray(_values_for(adata, key), dtype=float)
        kept = kept[np.isfinite(kept)]
        everyone = kept
        if qc_before is not None and key in qc_before:
            everyone = np.asarray(qc_before[key], dtype=float)
            everyone = everyone[np.isfinite(everyone)]
        pos = everyone[everyone > 0]
        if pos.size == 0:
            continue
        edges = np.geomspace(float(pos.min()), float(pos.max()), bins + 1)
        entry = {"bin_edges": edges.tolist(),
                 "cells_before_filtering": np.histogram(pos, edges)[0].tolist(),
                 "cells_kept": np.histogram(kept[kept > 0], edges)[0].tolist(),
                 "cells_with_none": int((everyone <= 0).sum()),
                 "median_kept": float(np.median(kept[kept > 0]))}
        if key in thresholds:
            entry["threshold"] = float(thresholds[key])
            entry["cells_below_threshold"] = int((everyone < float(thresholds[key])).sum())
        data["qc"][key] = entry
    if cluster_key in adata.obs:
        sizes = pd.Series(np.asarray(adata.obs[cluster_key])).value_counts()
        data["cluster_sizes"] = {str(k): int(v) for k, v in sizes.items()}
    pca = getattr(adata, "uns", {}).get("pca", {})
    if "variance_ratio" in pca:
        data["pca_variance_ratio"] = [float(v) for v in np.asarray(pca["variance_ratio"])]
    return data


def overview(adata: AnnData, out: str, *, cluster_key: str = "leiden",
             color_by: list[str] | tuple = (), thresholds: dict | None = None,
             qc_before: dict | None = None, prefix: str = "") -> dict:
    """The standard figures of an analysis, written to ``out``, each drawn the
    way that suits the size of the data: quality-control distributions over
    every cell, variance explained by the principal components, cluster sizes,
    and the UMAP coloured by cluster and by any obs column in ``color_by``
    (drawn as density above :data:`DENSITY_ABOVE` cells). ``qc_before`` is
    the per-cell metrics from before filtering (see
    :func:`qc_distributions`), so the cells a threshold removed are shown.
    Also writes ``figure_data.json``, the numbers behind the quality-control, cluster-size and
    variance plots (:func:`overview_data`). Returns ``{figure: path}`` for the figures that could
    be drawn."""
    import os

    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    path = lambda name: os.path.join(out, f"{prefix}{name}.png")  # noqa: E731
    files: dict[str, str] = {}
    qc_keys = [k for k in ("n_genes", "n_counts") if k in adata.obs]
    if qc_keys:
        qc_distributions(adata, qc_keys, thresholds=thresholds, before=qc_before, show=False,
                         save=path("qc"))
        files["qc"] = path("qc")
    if "pca" in getattr(adata, "uns", {}) and "variance_ratio" in adata.uns["pca"]:
        pca_variance_ratio(adata, show=False, save=path("pca"))
        files["pca"] = path("pca")
    if cluster_key in adata.obs:
        cluster_sizes(adata, cluster_key, show=False, save=path("clusters"))
        files["clusters"] = path("clusters")
        if "X_umap" in adata.obsm:
            umap(adata, color=cluster_key, show=False, save=path("umap"), legend_loc="right",
                 title=f"UMAP, {cluster_key}")
            files["umap"] = path("umap")
    for key in color_by:
        if key in adata.obs and "X_umap" in adata.obsm:
            umap(adata, color=key, show=False, save=path(f"umap_{key}"), title=f"UMAP, {key}")
            files[f"umap_{key}"] = path(f"umap_{key}")
    import json

    with open(path("figure_data").replace(".png", ".json"), "w") as fh:
        json.dump(overview_data(adata, cluster_key=cluster_key, qc_before=qc_before, thresholds=thresholds), fh)
    files["figure_data"] = path("figure_data").replace(".png", ".json")
    plt.close("all")
    return files


__all__ = [
    "umap", "pca", "pca_variance_ratio", "violin", "highly_variable_genes",
    "dotplot", "heatmap", "stacked_bar", "scatter_qc",
    "qc_distributions", "cluster_sizes", "overview", "overview_data", "DENSITY_ABOVE",
    "set_theme", "get_theme", "categorical_palette",
    "VEGA_20_SCANPY", "CATEGORICAL_PALETTE",
]
