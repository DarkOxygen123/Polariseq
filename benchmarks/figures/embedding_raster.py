"""Embeddings of millions of cells as density rasters, with the arrays kept as Source Data.

The pixels are computed exactly as ``ps.pl.umap(..., render="density")`` computes them (the library's
own grid, majority rule and opacity), but returned as arrays so that the figure can be redrawn, and
checked, from them:

  count (ny, nx) int32     cells in each pixel
  group (ny, nx) int32     the group most of a pixel's cells belong to (-1: empty, or no group)
  extent (4,)              x0, x1, y0, y1 of the grid in embedding coordinates
  centres (k, 4)           group, cells, median x, median y, for naming groups on the data

Every cell counts; nothing is sampled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from polariseq.plotting import _density_alpha, _group_centres, _raster_grid


def raster(coords: np.ndarray, codes: np.ndarray, bins: int = 600) -> dict:
    coords = np.asarray(coords, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    pix, (nx, ny), extent = _raster_grid(coords, bins)
    npx = nx * ny
    count = np.bincount(pix, minlength=npx)
    group = np.full(npx, -1, dtype=np.int32)
    keep = codes >= 0
    ncat = int(codes[keep].max()) + 1 if keep.any() else 1
    key = np.sort(pix[keep] * ncat + codes[keep])
    if key.size:
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        runs = np.diff(np.r_[starts, key.size])
        first = key[starts]
        p, c = first // ncat, first % ncat
        o = np.lexsort((-runs, p))
        winner = np.r_[True, p[o][1:] != p[o][:-1]]
        group[p[o][winner]] = c[o][winner]
    cen = _group_centres(coords[keep], codes[keep], ncat)
    centres = np.array([[g, n, x, y] for g, (n, x, y) in sorted(cen.items())], dtype=float).reshape(-1, 4)
    return {"count": count.reshape(ny, nx).astype(np.int32), "group": group.reshape(ny, nx),
            "extent": np.asarray(extent, dtype=float), "centres": centres}


def rgba(r: dict, colours_rgb: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """The image ``ps.pl.umap`` would draw for these arrays."""
    g = r["group"].ravel()
    out = np.zeros((g.size, 4))
    has = g >= 0
    out[has, :3] = colours_rgb[g[has]]
    out[:, 3] = _density_alpha(r["count"].ravel().astype(float), alpha)
    return out.reshape(*r["group"].shape, 4)


def save(path: Path, r: dict, group_names, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, count=r["count"], group=r["group"], extent=r["extent"], centres=r["centres"],
                        group_names=np.asarray([str(x) for x in group_names]),
                        **{k: np.asarray(v) for k, v in meta.items()})


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
