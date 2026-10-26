"""Comparing a UMAP start with the graph's eigenvectors (shared by the convergence scripts).

A start is a 2-D layout whose axes estimate the two leading non-trivial eigenvectors v2, v3 of the
normalized adjacency S = D^-1/2 W D^-1/2, each axis shifted and scaled to a box. Scaling does not
matter (spans are compared); the shift does, and is undone without changing a true eigenvector: the
non-trivial eigenvectors are orthogonal to the trivial one t = sqrt(deg), so each axis x is replaced
by x - c 1 with c = (x . t) / (1 . t), the one constant offset that makes it orthogonal to t. An exact
eigenvector (already orthogonal to t) is left as it is.

(An earlier version removed the constant vector 1 itself as well as t. 1 is not an eigenvector of S,
so that changed true eigenvectors too: exact eigenvectors scored an energy of 6.8 instead of 1.)
"""
from __future__ import annotations

import numpy as np


def unshift(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    c = (m.T @ t) / t.sum()
    return m - c[None, :]


def span(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the unshifted axes."""
    return np.linalg.qr(unshift(m, t))[0]


def angle(layout: np.ndarray, ref_span: np.ndarray, t: np.ndarray) -> float:
    """Largest principal angle (degrees) between a layout's unshifted span and a reference span."""
    sv = np.linalg.svd(span(layout, t).T @ ref_span, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(sv.min(), -1.0, 1.0))))


def energy_ratio(s_mat, layout: np.ndarray, lam: np.ndarray, t: np.ndarray) -> float:
    """Laplacian energy of the layout's two orthonormal axes (1 - mean Rayleigh quotient of S) over
    that of the two leading non-trivial eigenvectors (1 - (l2 + l3) / 2); 1.0 is optimal, and the
    measure does not depend on the gap below l3. `lam`: eigenvalues of S in decreasing order."""
    q = span(layout, t)
    rq = np.trace(q.T @ (s_mat @ q)) / q.shape[1]
    return float((1 - rq) / (1 - (lam[1] + lam[2]) / 2))
