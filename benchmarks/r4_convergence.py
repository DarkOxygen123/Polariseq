"""How close each UMAP start comes to the graph's eigenvectors, and at what cost.

    python benchmarks/r4_convergence.py --graph DIR --out OUT [--threads 8]

DIR holds a cell graph saved by benchmarks/r3_umap_cause_4m.py: connectivities.npz (the symmetric fuzzy
graph UMAP lays out), pcs.npy (principal components) and leiden.npy (clusters).

Reference: the two leading non-trivial eigenvectors of the normalized adjacency S = D^-1/2 W D^-1/2,
computed by scipy's ARPACK (eigsh), the solver umap-learn's spectral start uses (tolerance 1e-4 as there,
and 1e-8 as the reference). Starts measured, each against the reference as the largest principal angle
between 2-D spans (after removing the constant and trivial directions, so that scaling and shifting an
axis do not count), with its time:

  spectral     Polariseq's current start (subspace iteration, at most 128 iterations)
  multilevel   Polariseq's multilevel start through k-means groups of the principal components
               (250, 1,000, 4,000 groups) and through the Leiden clusters, after 0 to 300 smoothing steps

Writes OUT/convergence.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import polariseq as ps
    import r4_spectral as r4s
    import scipy.sparse as sp
    from polariseq._polariseq import (
        umap_kmeans_labels,
        umap_multilevel_layout,
        umap_spectral_layout,
    )
    from scipy.sparse.linalg import eigsh

    ps.set_budget(threads=a.threads)
    g = Path(a.graph)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"graph": str(g)}
    path = out / "convergence.json"

    def save():
        path.write_text(json.dumps(res, indent=1))

    w = sp.load_npz(g / "connectivities.npz").tocsr()
    w.sort_indices()
    n = w.shape[0]
    pcs = np.ascontiguousarray(np.load(g / "pcs.npy"), dtype=np.float32)
    leiden = np.load(g / "leiden.npy").astype(np.int64)
    deg = np.asarray(w.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    s = (dinv @ w @ dinv).astype(np.float64).tocsr()
    t = np.sqrt(deg)
    res["cells"], res["edges"] = int(n), int(w.nnz)
    res["measure"] = "benchmarks/r4_spectral.py (shift undone along the trivial eigenvector only)"

    def span(m):
        return r4s.span(m, t)

    def angle(layout, ref):
        return r4s.angle(layout, ref, t)

    refs = {}
    for tol in (1e-4, 1e-8):
        t0 = time.perf_counter()
        vals, vecs = eigsh(s, k=4, which="LA", tol=tol, v0=np.sqrt(deg), maxiter=n * 5)
        dt = time.perf_counter() - t0
        order = np.argsort(vals)[::-1]
        refs[tol] = span(vecs[:, order[1:3]])
        refs[f"vecs{tol}"] = vecs[:, order[1:3]]
        res["eigsh_tol1e-4" if tol == 1e-4 else "eigsh_tol1e-8"] = {"seconds": round(dt, 1), "eigenvalues": [float(v) for v in vals[order]]}
        save()
        print(f"eigsh tol {tol:g}: {dt:.1f} s, eigenvalues {vals[order]}", flush=True)
    ref = refs[1e-8]
    lam = np.sort(np.array(res["eigsh_tol1e-8"]["eigenvalues"]))[::-1]
    res["gap_ratio"] = float((lam[2] - lam[3]) / (1 - lam[2]))

    def energy(layout):
        return round(r4s.energy_ratio(s, layout, lam, t), 4)
    res["eigsh_tol1e-4"]["angle_to_reference"] = round(angle(refs[1e-4], ref), 3)
    res["eigsh_tol1e-4"]["energy_ratio"] = energy(refs["vecs0.0001"])

    csr = (w.indptr.astype(np.uint32), w.indices.astype(np.uint32), w.data.astype(np.float32), n)
    t0 = time.perf_counter()
    sp_start = umap_spectral_layout(*csr, 0)
    res["spectral_current"] = {"seconds": round(time.perf_counter() - t0, 1), "angle": round(angle(sp_start, ref), 3),
                               "energy_ratio": energy(sp_start)}
    save()
    print("current spectral start:", res["spectral_current"], flush=True)

    groupings = {}
    for k in (250, 1000, 4000):
        t0 = time.perf_counter()
        lab = np.asarray(umap_kmeans_labels(pcs, k, 0), dtype=np.uint32)
        groupings[f"kmeans{k}"] = (lab, int(lab.max()) + 1, time.perf_counter() - t0)
    _, codes = np.unique(leiden, return_inverse=True)
    groupings["leiden"] = (codes.astype(np.uint32), int(codes.max()) + 1, 0.0)
    res["multilevel"] = {}
    for name, (lab, k, t_group) in groupings.items():
        rows = []
        for steps in (0, 10, 30, 100, 300):
            t0 = time.perf_counter()
            lay = umap_multilevel_layout(*csr, lab, k, steps)
            rows.append({"smoothing_steps": steps, "seconds": round(time.perf_counter() - t0, 2),
                         "angle": round(angle(lay, ref), 3), "energy_ratio": energy(lay)})
            print(name, rows[-1], flush=True)
        res["multilevel"][name] = {"groups": k, "grouping_seconds": round(t_group, 1), "by_steps": rows}
        save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
