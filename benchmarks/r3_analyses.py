"""Follow-up analyses of the benchmark results: agreement, UMAP, subsampling, budgets.

Run after the laptop campaign (they are untimed, but they load the machine):

    python benchmarks/r3_analyses.py agreement      # seed spread, kNN and Leiden ablation
    python benchmarks/r3_analyses.py umap           # Polariseq UMAP vs umap-learn, same graph
    python benchmarks/r3_analyses.py organ_pcs      # PCs of every fetal organ (for sketching)
    python benchmarks/r3_analyses.py subsample      # what a subsample loses, fetal atlas
    python benchmarks/r3_analyses.py scanpy_organs  # scanpy running the authors' per-organ procedure
    python benchmarks/r3_analyses.py budget         # the budget sweep, per-process footprint

Outputs go to benchmarks/results/r3/<analysis>/, each with a summary.json
that the figures and the numbers read.

agreement
    Four public datasets, the fixed pipeline. Scanpy (tuned) and Polariseq
    at ten seeds each: how far two runs of one tool are apart is the scale
    against which the two tools' distance is read. Then two ablations that
    locate the remaining difference: both tools with exact nearest
    neighbours, and both Leiden implementations on one shared graph.
umap
    Polariseq's UMAP against umap-learn's (through scanpy) on the same
    neighbour graph, at two seeds each, on PBMC 10k and on the fetal atlas's
    published balanced sample (the authors' procedure). Three measures: how
    many of each cell's 15 nearest neighbours in PCA space stay among its 15
    in the layout (local), the share of its 15 layout neighbours with its own
    type (grouping), and the rank correlation of type-centroid distances in
    PCA space and in the layout (global arrangement).
subsample
    The authors' per-organ clustering, repeated on random subsamples and
    geometric sketches (Hie et al. 2019) of 1-25% of each organ, scored by
    the same recovery rule as the full analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
OUT = ROOT / "benchmarks/results/r3"
VAL = ROOT / "benchmarks/results/validation_data"
PUBLIC = {"pbmc3k": "pbmc3k.h5ad", "pbmc10k": "pbmc10k.h5ad",
          "neuron10k": "neuron10k.h5ad", "pbmc20k": "pbmc20k.h5ad"}
SEEDS = range(10)
P = dict(min_genes=200, target_sum=1e4, n_hvg=2000, n_pcs=50, n_neighbors=15,
         resolution=1.0)


def _save(d: Path, obj: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1)[:3000])


# ---------------------------------------------------------------- pipelines
def scanpy_prep(path: str):
    import scanpy as sc

    sc.settings.n_jobs = 4
    a = sc.read_h5ad(path)
    sc.pp.filter_cells(a, min_genes=P["min_genes"])
    sc.pp.normalize_total(a, target_sum=P["target_sum"])
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=P["n_hvg"], flavor="seurat", subset=True)
    return a


def scanpy_run(a, seed: int, *, exact: bool = False) -> np.ndarray:
    import scanpy as sc

    b = a.copy()
    sc.pp.pca(b, n_comps=P["n_pcs"], svd_solver="arpack", random_state=seed)
    kw = {"transformer": "sklearn"} if exact else {}
    sc.pp.neighbors(b, n_neighbors=P["n_neighbors"], random_state=seed, **kw)
    sc.tl.leiden(b, resolution=P["resolution"], random_state=seed, flavor="igraph",
                 n_iterations=2, directed=False)
    return b


def polariseq_prep(path: str):
    import polariseq as ps

    ps.set_budget(threads=4)
    a = ps.read_h5ad(path, lazy=False)
    ps.pp.calculate_qc_metrics(a)  # filter_cells reads n_genes from these
    ps.pp.filter_cells(a, min_genes=P["min_genes"])
    ps.pp.normalize_total(a, target_sum=P["target_sum"])
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=P["n_hvg"], subset=True)
    recipe = ps.workflow() == "polariseq"   # the default's second normalization and scaling
    if recipe:
        ps.pp.renormalize(a, target_sum=1000)
    ps.pp.pca(a, P["n_pcs"], seed=0, scale=recipe)
    return a


def polariseq_graph(a, seed: int, *, exact: bool = False) -> None:
    import polariseq as ps

    for k in ("distances", "connectivities"):
        a.obsp.pop(k, None) if hasattr(a.obsp, "pop") else None
    ps.pp.neighbors(a, n_neighbors=P["n_neighbors"], seed=seed,
                    approximate=False if exact else None)


def polariseq_leiden(a, seed: int) -> np.ndarray:
    import polariseq as ps

    ps.tl.leiden(a, resolution=P["resolution"], seed=seed)
    return np.asarray(a.obs["leiden"]).astype(int)


# ---------------------------------------------------------------- agreement
def agreement() -> None:
    from itertools import combinations

    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score as ari

    # R3_AGREEMENT=brain100k,brain200k runs other datasets and merges them
    # into the existing summary (the ladder's single-seed ARIs needed a floor).
    want = os.environ.get("R3_AGREEMENT")
    sets = {n: f"{n}.h5ad" for n in want.split(",")} if want else PUBLIC
    prev = OUT / "agreement" / "summary.json"
    out: dict = json.loads(prev.read_text()) if want and prev.exists() else {}
    for name, fn in sets.items():
        path = str(VAL / fn)
        t0 = time.perf_counter()
        base = scanpy_prep(path)
        sc_lab, sc_exact = {}, {}
        for s in SEEDS:
            b = scanpy_run(base, s)
            sc_lab[s] = np.asarray(b.obs["leiden"]).astype(int)
            b = scanpy_run(base, s, exact=True)
            sc_exact[s] = np.asarray(b.obs["leiden"]).astype(int)
            if s == 0:
                shared = b  # scanpy's exact graph, for the Leiden ablation
        cells = list(base.obs_names)

        pa = polariseq_prep(path)
        # Both tools keep the same cells; put Polariseq's labels in Scanpy's
        # cell order (by name when names are unique, else by file order).
        q = pd.Index([str(c) for c in pa.obs_names])
        ref = pd.Index([str(c) for c in cells])
        if q.is_unique and ref.is_unique:
            assert set(q) == set(ref), "the two tools kept different cells"
            order = q.get_indexer(ref)
        else:
            assert len(q) == len(ref), "the two tools kept different numbers of cells"
            order = np.arange(len(q))
        ps_lab, ps_exact = {}, {}
        for s in SEEDS:
            polariseq_graph(pa, s)
            ps_lab[s] = polariseq_leiden(pa, s)[order]
            polariseq_graph(pa, s, exact=True)
            ps_exact[s] = polariseq_leiden(pa, s)[order]

        # Leiden only: both implementations on scanpy's exact graph.
        import polariseq as ps
        g = shared.obsp["connectivities"].tocsr()
        pg = ps.AnnData(X=np.zeros((g.shape[0], 1), dtype=np.float32))
        pg.obsp["distances"] = shared.obsp["distances"].tocsr()
        pg.obsp["connectivities"] = g
        leiden_ps = {s: polariseq_leiden(pg, s) for s in SEEDS}
        leiden_sc = {}
        for s in SEEDS:
            sc.tl.leiden(shared, resolution=P["resolution"], random_state=s,
                         flavor="igraph", n_iterations=2, directed=False, key_added="l")
            leiden_sc[s] = np.asarray(shared.obs["l"]).astype(int)

        def within(d):
            return [ari(d[i], d[j]) for i, j in combinations(SEEDS, 2)]

        def between(d1, d2):
            return [ari(d1[i], d2[j]) for i in SEEDS for j in SEEDS]

        out[name] = {
            "cells": len(cells),
            "default": {"scanpy_vs_scanpy": within(sc_lab), "polariseq_vs_polariseq": within(ps_lab),
                        "polariseq_vs_scanpy": between(ps_lab, sc_lab)},
            "exact_knn": {"scanpy_vs_scanpy": within(sc_exact),
                          "polariseq_vs_polariseq": within(ps_exact),
                          "polariseq_vs_scanpy": between(ps_exact, sc_exact)},
            "shared_graph": {"scanpy_vs_scanpy": within(leiden_sc),
                             "polariseq_vs_polariseq": within(leiden_ps),
                             "polariseq_vs_scanpy": between(leiden_ps, leiden_sc)},
            "n_clusters": {"scanpy": [int(len(set(v))) for v in sc_lab.values()],
                           "polariseq": [int(len(set(v))) for v in ps_lab.values()]},
            "seconds": round(time.perf_counter() - t0, 1),
        }
        m = {k: {kk: round(float(np.median(vv)), 3) for kk, vv in v.items()}
             for k, v in out[name].items() if isinstance(v, dict) and k != "n_clusters"}
        print(name, json.dumps(m))
    _save(OUT / "agreement", out)


# ---------------------------------------------------------------- umap
def _umap_metrics(pcs, emb, groups, rng, n_query=20_000, k=15) -> dict:
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    n = len(emb)
    q = rng.choice(n, min(n_query, n), replace=False)
    nn_hi = NearestNeighbors(n_neighbors=k + 1).fit(pcs).kneighbors(pcs[q], return_distance=False)[:, 1:]
    nn_lo = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb[q], return_distance=False)[:, 1:]
    local = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nn_hi, nn_lo)]))
    purity = float(np.mean(groups[nn_lo] == groups[q][:, None]))
    labs = pd.unique(groups)
    c_hi = np.array([pcs[groups == g].mean(axis=0) for g in labs])
    c_lo = np.array([emb[groups == g].mean(axis=0) for g in labs])
    iu = np.triu_indices(len(labs), 1)
    d_hi = np.linalg.norm(c_hi[:, None] - c_hi[None], axis=-1)[iu]
    d_lo = np.linalg.norm(c_lo[:, None] - c_lo[None], axis=-1)[iu]
    return {"knn_preservation": round(local, 4), "type_purity": round(purity, 4),
            "centroid_spearman": round(float(spearmanr(d_hi, d_lo).statistic), 4),
            "n_groups": int(len(labs))}


def _both_umaps(a, groups, min_dist: float, metric_note: str, name: str) -> dict:
    """Polariseq and umap-learn layouts of one graph (`a` holds X_pca and
    obsp distances/connectivities from Polariseq), two seeds each."""
    import anndata as ad
    import polariseq as ps
    import scanpy as sc

    pcs = np.asarray(a.obsm["X_pca"], dtype=np.float32)
    rng = np.random.default_rng(0)
    res, layouts = {}, {}
    for s in (0, 1):
        t0 = time.perf_counter()
        ps.tl.umap(a, min_dist=min_dist, seed=s)
        layouts[f"polariseq_seed{s}"] = (np.asarray(a.obsm["X_umap"]), time.perf_counter() - t0)
    b = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(pcs))]))
    b.obsm["X_pca"] = pcs
    b.obsp["distances"] = a.obsp["distances"].tocsr()
    b.obsp["connectivities"] = a.obsp["connectivities"].tocsr()
    b.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"n_neighbors": int(a.obsp["distances"][0].nnz + 1),
                                     "method": "umap", "metric": metric_note}}
    for s in (0, 1):
        t0 = time.perf_counter()
        if min_dist is None:   # match Polariseq's default layout settings
            sc.tl.umap(b, min_dist=0.5, maxiter=100, random_state=s)
        else:
            sc.tl.umap(b, min_dist=min_dist, random_state=s)
        layouts[f"umap_learn_seed{s}"] = (np.asarray(b.obsm["X_umap"]), time.perf_counter() - t0)
    d = OUT / "umap" / name
    d.mkdir(parents=True, exist_ok=True)
    for k, (emb, secs) in layouts.items():
        np.save(d / f"{k}.npy", emb.astype(np.float32))
        res[k] = {**_umap_metrics(pcs, emb, groups, rng), "seconds": round(secs, 1)}
    np.save(d / "groups.npy", groups)
    return res


def umap_pbmc10k() -> None:
    """PBMC 10k only, Polariseq's default layout against umap-learn on the same graph."""
    import polariseq as ps

    ps.set_budget(ram_gb=6, threads=4)
    a = polariseq_prep(str(VAL / PUBLIC["pbmc10k"]))
    polariseq_graph(a, 0)
    groups = polariseq_leiden(a, 0)
    out = {"pbmc10k": _both_umaps(a, groups.astype(str), None, "euclidean", "pbmc10k")}
    d = OUT / "umap_pbmc10k"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1))


def umap() -> None:
    import polariseq as ps

    ps.set_budget(ram_gb=6, threads=4)
    out = {}   # the fetal sample only (PBMC 10k is not redone here)

    # The fetal atlas's published balanced sample, the authors' procedure.
    import fetal_matched as fm

    x, meta, *_ = fm.matched_sample()
    a = ps.AnnData(X=x)
    ps.pp.pca(a, fm.N_PCS, seed=fm.SEED)
    ps.pp.neighbors(a, n_neighbors=fm.N_NEIGHBORS, metric="cosine", seed=fm.SEED)
    ps.tl.leiden(a, seed=0)  # builds obsp['connectivities']
    groups = meta["Main_cluster_name"].to_numpy().astype(str)
    out["fetal_matched"] = _both_umaps(a, groups, fm.MIN_DIST, "cosine", "fetal_matched")
    pub = meta[["Global_umap_1", "Global_umap_2"]].to_numpy(dtype=np.float32)
    out["fetal_matched"]["published"] = _umap_metrics(
        np.asarray(a.obsm["X_pca"]), pub, groups, np.random.default_rng(0))
    _save(OUT / "umap", out)


# ---------------------------------------------------------------- fetal subsampling
FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.25)
RARE = 1000


def _organ_rows(path: Path, rows: np.ndarray):
    """Sorted rows of an organ file as CSR (all genes)."""
    import h5py
    import scipy.sparse as sp

    with h5py.File(path, "r") as f:
        indptr = f["X/indptr"][:]
        n_genes = int(f["X"].attrs["shape"][1])
        data, idx = f["X/data"], f["X/indices"]
        blocks, W = [], 20_000
        for w0 in range(0, len(indptr) - 1, W):
            w1 = min(w0 + W, len(indptr) - 1)
            sel = rows[(rows >= w0) & (rows < w1)]
            if not len(sel):
                continue
            a, b = int(indptr[w0]), int(indptr[w1])
            blk = sp.csr_matrix((data[a:b], idx[a:b], indptr[w0:w1 + 1] - a),
                                shape=(w1 - w0, n_genes))[sel - w0]
            blocks.append(blk)
    return sp.vstack(blocks, format="csr").astype(np.float32)


def _organ_obs(path: Path) -> np.ndarray:
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as f:
        return read_elem(f["obs"]).index.astype(str).to_numpy()


def _cluster_in_memory(x) -> tuple[np.ndarray, np.ndarray]:
    """The fixed pipeline in memory; returns (kept row mask, labels)."""
    import polariseq as ps

    a = ps.AnnData(X=x)
    n0 = a.n_obs
    ps.pp.calculate_qc_metrics(a)
    keep = np.asarray(a.obs["n_genes"]) >= P["min_genes"]
    ps.pp.filter_cells(a, min_genes=P["min_genes"])
    ps.pp.normalize_total(a, target_sum=P["target_sum"])
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=P["n_hvg"], subset=True)
    recipe = ps.workflow() == "polariseq"   # the default's second normalization and scaling
    if recipe:
        ps.pp.renormalize(a, target_sum=1000)
    ps.pp.pca(a, min(P["n_pcs"], a.n_obs - 1), seed=0, scale=recipe)
    ps.pp.neighbors(a, n_neighbors=P["n_neighbors"], seed=0)
    ps.tl.leiden(a, resolution=P["resolution"], seed=0)
    assert len(keep) == n0 and keep.sum() == a.n_obs
    return keep, np.asarray(a.obs["leiden"]).astype(np.int64)


def score_types(obs: np.ndarray, labels: np.ndarray) -> dict:
    """The recovery rule of fetal_biology.run_compare (split-tolerant form):
    a published type is recovered when the clusters it dominates hold at
    least half its cells and are at least half made of it. Types are counted
    against all 77, so a type missing from a subsample is not recovered."""
    import fetal_biology as fb

    cells = pd.read_csv(fb.CELLS, index_col="cell", usecols=["cell", "Main_cluster_name"])
    all_types = cells["Main_cluster_name"].value_counts()
    pub = cells.loc[obs, "Main_cluster_name"].to_numpy()
    ct = pd.crosstab(pd.Series(pub, name="t"), pd.Series(labels, name="c"))
    majority = ct.idxmax(axis=0)
    size_t, size_c = ct.sum(axis=1), ct.sum(axis=0)
    rec = []
    for t in ct.index:
        own = [c for c in ct.columns if majority[c] == t]
        held = ct.loc[t, own].sum() if own else 0
        if own and held / size_t[t] >= fb.RECOVERY and held / size_c[own].sum() >= fb.RECOVERY:
            rec.append(t)
    rare = set(all_types[all_types < RARE].index)
    same = (pd.Series(labels).map(majority).to_numpy() == pub)
    return {"recovered": len(rec), "of": int(len(all_types)),
            "n_clusters": int(len(np.unique(labels))),
            "rare_recovered": len(set(rec) & rare), "rare_of": len(rare),
            "cells_under_own_type": round(float(same.mean()), 4),
            "recovered_types": sorted(rec)}


def organ_pcs() -> None:
    """Every organ's out-of-core PCs (all cells), for geometric sketching."""
    import polariseq as ps
    import fetal_biology as fb
    import fetal_matched as fm

    manifest = json.loads((fm.ORGAN_FILES / "manifest.json").read_text())
    ps.set_budget(ram_gb=6, threads=4)
    d = OUT / "organ_pcs"
    d.mkdir(parents=True, exist_ok=True)
    for name, m in manifest.items():
        if (d / f"{name}.npy").exists():
            continue
        a = ps.process_diskbacked(str(ROOT / m["path"]), **fb.PARAMS)
        np.save(d / f"{name}.npy", np.asarray(a.obsm["X_pca"], dtype=np.float32))
        np.savetxt(d / f"{name}.obs.txt", np.asarray(a.obs_names, dtype=object), fmt="%s")
        print(f"[organ_pcs] {name}: {a.n_obs:,} cells")


def subsample() -> None:
    import fetal_matched as fm
    from geosketch import gs

    manifest = json.loads((fm.ORGAN_FILES / "manifest.json").read_text())
    d = OUT / "subsample"
    d.mkdir(parents=True, exist_ok=True)
    done = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    for method in ("random", "geosketch"):
        for f in FRACTIONS:
            key = f"{method}_{f}"
            if key in done:
                continue
            obs_all, lab_all, off, t0 = [], [], 0, time.perf_counter()
            for name, m in manifest.items():
                path = ROOT / m["path"]
                obs = _organ_obs(path)
                n = len(obs)
                k = max(int(round(f * n)), P["n_neighbors"] + 2)
                if method == "random":
                    rows = np.sort(np.random.default_rng(0).permutation(n)[:k])
                else:
                    pcs = np.load(OUT / "organ_pcs" / f"{name}.npy")
                    kept = np.loadtxt(OUT / "organ_pcs" / f"{name}.obs.txt", dtype=str)
                    pos = pd.Index(obs).get_indexer(kept)
                    k = min(k, len(kept))
                    rows = np.sort(pos[np.asarray(gs(pcs, k, replace=False, seed=0))])
                x = _organ_rows(path, rows)
                keep, lab = _cluster_in_memory(x)
                obs_all.append(obs[rows][keep])
                lab_all.append(lab + off)
                off += int(lab.max()) + 1
            res = score_types(np.concatenate(obs_all), np.concatenate(lab_all))
            res.update({"cells": int(sum(len(o) for o in obs_all)),
                        "seconds": round(time.perf_counter() - t0, 1)})
            done[key] = res
            (d / "summary.json").write_text(json.dumps(done, indent=1))
            print(key, {k: v for k, v in res.items() if k != "recovered_types"})
    # The full data, same rule, from the organ-by-organ run.
    org = ROOT / "benchmarks/results/fetal/4062k_by_organ"
    done["all_1.0"] = score_types(np.loadtxt(org / "obs_names.txt", dtype=str),
                                  np.load(org / "labels.npy").astype(np.int64))
    _save(d, done)


# ---------------------------------------------------------------- scanpy per organ
def _scanpy_organ(path: str, out: str) -> None:
    """Child process: scanpy's fixed pipeline on one organ file."""
    import scanpy as sc

    sc.settings.n_jobs = 4
    a = sc.read_h5ad(path)
    if a.X.indptr.dtype != a.X.indices.dtype:
        a.X.indices = a.X.indices.astype(a.X.indptr.dtype)
    sc.pp.filter_cells(a, min_genes=P["min_genes"])
    sc.pp.normalize_total(a, target_sum=P["target_sum"])
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=P["n_hvg"], flavor="seurat", subset=True)
    sc.pp.pca(a, n_comps=P["n_pcs"], svd_solver="arpack", random_state=0)
    sc.pp.neighbors(a, n_neighbors=P["n_neighbors"], random_state=0)
    sc.tl.leiden(a, resolution=P["resolution"], random_state=0, flavor="igraph",
                 n_iterations=2, directed=False)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    np.save(o / "labels.npy", np.asarray(a.obs["leiden"]).astype(np.int64))
    np.savetxt(o / "obs_names.txt", np.asarray(a.obs_names, dtype=object), fmt="%s")


def scanpy_organs() -> None:
    import fetal_matched as fm

    manifest = json.loads((fm.ORGAN_FILES / "manifest.json").read_text())
    d = OUT / "scanpy_organs"
    status = {}
    for name, m in sorted(manifest.items(), key=lambda kv: kv[1]["cells"]):
        o = d / name
        if not (o / "labels.npy").exists():
            t0 = time.perf_counter()
            # Same limits as the benchmark: 60 minutes and a footprint of
            # 1.5x the laptop's memory (r3_campaign.MEMORY_LIMIT_GB).
            from telemetry.footprint import rusage  # noqa: PLC0415

            proc = subprocess.Popen([sys.executable, __file__, "_scanpy_organ",
                                     str(ROOT / m["path"]), str(o)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            status[name] = None
            while proc.poll() is None:
                r_ = rusage(proc.pid)
                if r_ and r_["lifetime_max_phys_footprint"] > 1.5 * 8.59e9:
                    proc.kill()
                    status[name] = "exceeds memory (12.9 GB)"
                elif time.perf_counter() - t0 > 3600:
                    proc.kill()
                    status[name] = "timeout (60 min)"
                time.sleep(0.2)
            if status[name] is None:
                status[name] = "ok" if proc.returncode == 0 else f"failed ({proc.returncode})"
            print(f"[scanpy_organs] {name}: {status[name]} in {time.perf_counter() - t0:.0f} s")
        else:
            status[name] = "ok"
    ok = [n for n, s in status.items() if s == "ok" and (d / n / "labels.npy").exists()]
    obs = np.concatenate([np.loadtxt(d / n / "obs_names.txt", dtype=str) for n in ok])
    labs, off = [], 0
    for n in ok:
        lab = np.load(d / n / "labels.npy")
        labs.append(lab + off)
        off += int(lab.max()) + 1
    res = score_types(obs, np.concatenate(labs))
    res["organ_status"] = status
    _save(d, res)


# ---------------------------------------------------------------- budget sweep
def budget() -> None:
    """Out-of-core at 800K cells, budgets 1-8 GB, through the harness; the
    budgets below the planner's floor are expected to be declined."""
    for gb in (1, 2, 3, 4, 6, 8):
        subprocess.run([sys.executable, str(ROOT / "benchmarks/run_telemetry.py"),
                        "--datasets", "brain800k", "--arms", "polariseq-spill",
                        "--threads", "4", "--ram-gb", str(gb), "--no-umap", "--cache", "warm",
                        "--gate", "calibrate", "--reps", "1", "--resume",
                        "--results", str(OUT / f"budget/{gb}gb")], check=False)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_scanpy_organ":
        _scanpy_organ(sys.argv[2], sys.argv[3])
        return 0
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["agreement", "umap", "umap_pbmc10k", "organ_pcs", "subsample",
                                     "scanpy_organs", "budget"])
    a = ap.parse_args()
    {"agreement": agreement, "umap": umap, "umap_pbmc10k": umap_pbmc10k, "organ_pcs": organ_pcs, "subsample": subsample,
     "scanpy_organs": scanpy_organs, "budget": budget}[a.what]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
