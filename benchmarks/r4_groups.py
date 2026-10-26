"""How many groups the multilevel start needs, by dataset size.

    python benchmarks/r4_groups.py OUT.json DATA [DATA ...]

For each dataset: the recommended pipeline (seurat ranking with a 1% detection floor, Scarf's
normalization, PCA 50, 15 neighbors); the reference eigenvectors by ARPACK (tolerance 1e-8); the angle
of the multilevel start to them for 25 to 4,000 k-means groups after 10 smoothing steps, with its time.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import polariseq as ps
    import r4_spectral as r4s
    import scipy.sparse as sp
    from polariseq._polariseq import (
        umap_kmeans_labels,
        umap_multilevel_layout,
        umap_spectral_layout,
    )
    from scipy.sparse.linalg import eigsh

    out, paths = sys.argv[1], sys.argv[2:]
    ps.set_budget(ram_gb=6, threads=4)
    res = {}
    for path in paths:
        a = ps.pp._ensure_materialized(ps.read_h5ad(path, mode="memory"))
        raw = a.copy()
        ps.pp.calculate_qc_metrics(a)
        keep = np.asarray(a.obs["n_genes"]) >= 200
        a, raw = a[keep], raw[keep]
        ps.pp.normalize_total(a, target_sum=1e4)
        ps.pp.log1p(a)
        ps.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat", min_cells=int(0.01 * a.n_obs), subset=True)
        ps.pp.renormalize(a, target_sum=1000, counts=raw)
        ps.pp.pca(a, 50, scale=True, seed=0)
        ps.pp.neighbors(a, n_neighbors=15, seed=0)
        ps.tl.leiden(a, seed=0)
        w = a.obsp["connectivities"].tocsr()
        w.sort_indices()
        n = w.shape[0]
        deg = np.asarray(w.sum(axis=1)).ravel()
        dinv = sp.diags(1.0 / np.sqrt(deg))
        s_mat = (dinv @ w @ dinv).astype(np.float64).tocsr()
        vals, vecs = eigsh(s_mat, k=4, which="LA", tol=1e-8, v0=np.sqrt(deg))
        order = np.argsort(vals)[::-1]
        t = np.sqrt(deg)

        def span(m):
            return r4s.span(m, t)

        def energy_ratio(s_mat, q, lam):   # q: a layout (its span is taken inside)
            return r4s.energy_ratio(s_mat, q, lam, t)
        ref = span(vecs[:, order[1:3]])
        pcs = np.ascontiguousarray(a.obsm["X_pca"], dtype=np.float32)
        csr = (w.indptr.astype(np.uint32), w.indices.astype(np.uint32), w.data.astype(np.float32), n)
        rows = []
        for k in (25, 50, 100, 250, 500, 1000, 2000, 4000):
            if k > n // 2:
                continue
            t0 = time.perf_counter()
            lab = np.asarray(umap_kmeans_labels(pcs, k, 0), dtype=np.uint32)
            lay = umap_multilevel_layout(*csr, lab, int(lab.max()) + 1, 10)
            dt = time.perf_counter() - t0
            sv = np.linalg.svd(span(lay).T @ ref, compute_uv=False)
            rows.append({"groups": k, "cells_per_group": round(n / k, 1), "seconds": round(dt, 2),
                         "angle": round(float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1)))), 3),
                         "energy_ratio": round(energy_ratio(s_mat, lay, vals[order]), 4)})
            print(path.split("/")[-1], rows[-1], flush=True)
        lam = vals[order]
        t0 = time.perf_counter()
        cur = umap_spectral_layout(*csr, 0)
        cur_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        v4, w4 = eigsh(s_mat, k=3, which="LA", tol=1e-4, v0=np.sqrt(deg))
        arp_s = time.perf_counter() - t0
        o4 = np.argsort(v4)[::-1]
        k_default = int(min(1000, max(50, n // 20)))
        lab = np.asarray(umap_kmeans_labels(pcs, k_default, 0), dtype=np.uint32)
        steps = []
        for n_steps in (0, 10, 30, 100, 300):
            t0 = time.perf_counter()
            lay = umap_multilevel_layout(*csr, lab, int(lab.max()) + 1, n_steps)
            steps.append({"smoothing_steps": n_steps, "seconds": round(time.perf_counter() - t0, 3),
                          "energy_ratio": round(energy_ratio(s_mat, lay, lam), 4)})
        extra = {"current_spectral": {"seconds": round(cur_s, 3), "energy_ratio": round(energy_ratio(s_mat, cur, lam), 4)},
                 "arpack_tol1e-4": {"seconds": round(arp_s, 3),
                                    "energy_ratio": round(energy_ratio(s_mat, w4[:, o4[1:3]], lam), 4)},
                 f"multilevel_{k_default}_groups_by_steps": steps}
        print(path.split("/")[-1], json.dumps(extra), flush=True)
        res[path] = {**extra, "cells": int(n), "eigenvalues": [float(v) for v in lam],
                     "gap_ratio": float((lam[2] - lam[3]) / (1 - lam[2])), "by_groups": rows}
        open(out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
