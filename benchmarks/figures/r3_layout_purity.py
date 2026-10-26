"""How well a 2-D layout keeps cell types together: the pixel purity used by r4_sweep.py, r4_tools_fetal.py and
r3_umap_cause_4m.py.

On a grid of 600 pixels on the longer side (the library's density grid), the share of cells whose pixel holds
mostly cells of their own type. 1.0 would mean no pixel mixes types.
"""
from __future__ import annotations

import numpy as np
from polariseq.plotting import _raster_grid


def purity(xy: np.ndarray, codes: np.ndarray, bins: int = 600) -> float:
    pix, _, _ = _raster_grid(np.asarray(xy, dtype=float), bins)
    k = int(codes.max()) + 1
    key, count = np.unique(pix.astype(np.int64) * k + codes, return_counts=True)
    p = key // k
    o = np.lexsort((-count, p))
    first = np.r_[True, p[o][1:] != p[o][:-1]]
    return float(count[o][first].sum() / len(codes))

