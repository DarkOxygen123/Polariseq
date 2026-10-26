"""Validate the Paris port on real data, against scikit-network and Scarf.

The unit tests (``python/polariseq/tests/test_paris.py``) pin the port on
small stress graphs. This script checks the port on real data:

    .venv/bin/python        benchmarks/validate_paris.py export     # Polariseq graphs
    .venv-scarf/bin/python  benchmarks/validate_paris.py reference  # scikit-network on them
    .venv-scarf/bin/python  benchmarks/validate_paris.py scarf      # Scarf end to end, PBMC 3k
    .venv/bin/python        benchmarks/validate_paris.py compare    # port vs both, bit for bit
    .venv/bin/python        benchmarks/validate_paris.py fetal      # 4.06M cells: time, memory, types

``export`` builds Polariseq's connectivity graph (the benchmark parameters:
2,000 HVGs, 50 PCs, 15 neighbours) for PBMC 3k, PBMC 10k and the 100K-cell
brain subset. ``reference`` runs scikit-network's ``Paris`` on each and
times it. ``scarf`` runs Scarf's own pipeline on PBMC 3k and saves the
directed graph ``run_clustering`` clusters, its dendrogram and its labels
(straight and balanced cuts). ``compare`` runs the port on the same graphs
(NumPy 1.x summation, as the references ran) and requires identical
dendrograms and cuts. ``fetal`` runs the port on the full atlas's graph in a
child process, records time and peak footprint, and scores cuts against the
published cell types beside the Leiden runs. Outputs go to
``benchmarks/results/paris_validation/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/paris_validation"
VD = ROOT / "benchmarks/results/validation_data"
GRAPHS = ("pbmc3k", "pbmc10k", "brain100k")
PARAMS = {"n_hvg": 2000, "n_pcs": 50, "n_neighbors": 15, "min_genes": 200, "target_sum": 1e4}
SCARF_CUTS = {"straight": 10, "balanced": (100, 10, 2.0)}  # (max_size, min_size, fc)


def _save_csr(path: Path, m) -> None:
    m = m.tocsr()
    m.sum_duplicates()
    m.sort_indices()
    np.savez_compressed(path, indptr=m.indptr.astype(np.int64), indices=m.indices.astype(np.int64),
                        data=m.data.astype(np.float64), n=np.array(m.shape[0]))


def _load_csr(path: Path):
    import scipy.sparse as sp

    z = np.load(path)
    n = int(z["n"])
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))


def export() -> None:
    import polariseq as ps

    (OUT / "graphs").mkdir(parents=True, exist_ok=True)
    for name in GRAPHS:
        a = ps.read_h5ad(str(VD / f"{name}.h5ad"), lazy=False)
        ps.pp.filter_cells(a, min_genes=PARAMS["min_genes"])
        ps.pp.normalize_total(a, target_sum=PARAMS["target_sum"])
        ps.pp.log1p(a)
        ps.pp.highly_variable_genes(a, n_top_genes=PARAMS["n_hvg"], subset=True)
        ps.pp.pca(a, PARAMS["n_pcs"], seed=0)
        ps.pp.neighbors(a, n_neighbors=PARAMS["n_neighbors"], seed=0)
        _save_csr(OUT / "graphs" / f"{name}.npz", a.obsp["connectivities"])
        print(f"[export] {name}: {a.n_obs:,} cells, {a.obsp['connectivities'].nnz:,} entries")
    # Scarf's reader needs string indices, not anndata 0.13's nullable-string
    # groups: the benchmark arm's no-copy shim.
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from telemetry.arms.seurat_arm import _bpcells_readable

    (OUT / "scarf").mkdir(parents=True, exist_ok=True)
    link = OUT / "scarf" / "pbmc3k_readable.h5ad"
    if not link.exists():
        _bpcells_readable(str(VD / "pbmc3k.h5ad"), link)


def reference() -> None:
    from sknetwork.hierarchy import Paris

    res = {}
    for name in GRAPHS:
        g = _load_csr(OUT / "graphs" / f"{name}.npz")
        t = time.perf_counter()
        d = Paris(reorder=False).fit_transform(g)
        s = time.perf_counter() - t
        np.save(OUT / "graphs" / f"{name}.sknetwork.npy", d)
        res[name] = {"seconds": round(s, 2), "n": g.shape[0]}
        print(f"[reference] {name}: scikit-network Paris {s:.1f} s")
    (OUT / "reference_times.json").write_text(json.dumps(res, indent=1))


def scarf() -> None:
    """Scarf's pipeline on PBMC 3k; keep the graph run_clustering sees."""
    import scarf as sc_
    import zarr

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from telemetry.arms.scarf_pipeline import _consume_rows

    d = OUT / "scarf"
    d.mkdir(parents=True, exist_ok=True)
    zloc = d / "pbmc3k.zarr"
    if not zloc.exists():
        sc_.readers.H5adReader.consume_group = _consume_rows
        sc_.H5adToZarr(sc_.H5adReader(str(d / "pbmc3k_readable.h5ad")), zarr_loc=str(zloc),
                       chunk_size=(2000, 1000)).dump(batch_size=2000)
    ds = sc_.DataStore(str(zloc), nthreads=4, min_features_per_cell=0, min_cells_per_feature=0)
    ds.filter_cells(attrs=["RNA_nFeatures"], lows=[PARAMS["min_genes"] - 1], highs=[np.inf])
    ds.mark_hvgs(top_n=PARAMS["n_hvg"], blacklist="^$", show_plot=False)
    ds.make_graph(feat_key="hvgs", dims=PARAMS["n_pcs"], k=PARAMS["n_neighbors"], rand_state=0)
    g = ds.load_graph(from_assay="RNA", cell_key="I", feat_key="hvgs", symmetric=False, upper_only=False)
    _save_csr(d / "graph.npz", g)
    ds.run_clustering(n_clusters=SCARF_CUTS["straight"], label="straight", force_recalc=True)
    mx, mn, fc = SCARF_CUTS["balanced"]
    ds.run_clustering(balanced_cut=True, max_size=mx, min_size=mn, max_distance_fc=fc, label="balanced")
    loc = ds.zw[ds._get_latest_graph_loc("RNA", "I", "hvgs")].attrs["latest_dendrogram"]
    np.save(d / "dendrogram.npy", np.asarray(ds.zw[loc][:]))
    keep = ds.cells.fetch("I")
    for lab in ("straight", "balanced"):
        np.save(d / f"{lab}.npy", np.asarray(ds.cells.fetch(f"RNA_{lab}"))[keep])
    print(f"[scarf] PBMC 3k: {g.shape[0]:,} cells, directed graph {g.nnz:,} entries, "
          f"dtype {g.dtype}; zarr {zarr.__version__}")


def _same_partition(a, b) -> bool:
    pairs = set(zip(np.asarray(a).tolist(), np.asarray(b).tolist(), strict=True))
    return len(pairs) == len(set(np.asarray(a).tolist())) == len(set(np.asarray(b).tolist()))


def compare() -> None:
    from polariseq._polariseq import dendrogram_cut_balanced, dendrogram_cut_straight, paris_dendrogram

    ref_t = json.loads((OUT / "reference_times.json").read_text())
    report = {}
    for name in GRAPHS:
        z = np.load(OUT / "graphs" / f"{name}.npz")
        t = time.perf_counter()
        d = paris_dendrogram(z["indptr"].astype(np.uint32), z["indices"].astype(np.uint32), z["data"],
                             int(z["n"]), numpy_sum_block=8192)
        s = time.perf_counter() - t
        want = np.load(OUT / "graphs" / f"{name}.sknetwork.npy")
        same = bool(d.shape == want.shape and np.array_equal(d, want))
        report[name] = {"identical": same, "port_s": round(s, 3), "scikit_network_s": ref_t[name]["seconds"],
                        "n": int(z["n"])}
        print(f"[compare] {name}: identical={same}  port {s:.2f} s vs scikit-network "
              f"{ref_t[name]['seconds']:.1f} s")
    # Scarf end to end: its directed float64 graph, its stored dendrogram
    # (infinite heights set to 0), its straight and balanced labels.
    sd = OUT / "scarf"
    z = np.load(sd / "graph.npz")
    d = paris_dendrogram(z["indptr"].astype(np.uint32), z["indices"].astype(np.uint32), z["data"],
                         int(z["n"]), numpy_sum_block=8192)
    d[d == np.inf] = 0
    want = np.load(sd / "dendrogram.npy")
    straight = dendrogram_cut_straight(d, SCARF_CUTS["straight"]) + 1
    mx, mn, fc = SCARF_CUTS["balanced"]
    balanced = dendrogram_cut_balanced(d, mx, mn, fc, numpy_sum_block=8192)
    report["scarf_pbmc3k"] = {
        "dendrogram_identical": bool(np.array_equal(d, want)),
        "straight_same_partition": _same_partition(straight, np.load(sd / "straight.npy")),
        "balanced_identical_labels": bool(np.array_equal(balanced, np.load(sd / "balanced.npy"))),
        "n": int(z["n"]),
    }
    print(f"[compare] Scarf PBMC 3k: {report['scarf_pbmc3k']}")
    (OUT / "compare.json").write_text(json.dumps(report, indent=1))


def _fetal_child(out: str) -> None:
    """Child process: Paris on the full fetal atlas graph, alone, so its
    peak footprint is Paris's (plus the loaded graph)."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import resource
    from polariseq._polariseq import paris_dendrogram

    z = np.load(Path(out) / "fetal_graph.npz")
    ip, ix, dx, n = z["indptr"].astype(np.uint32), z["indices"].astype(np.uint32), z["data"], int(z["n"])
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # Linux, KiB -> bytes
    t = time.perf_counter()
    d = paris_dendrogram(ip, ix, dx, n)
    s = time.perf_counter() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    np.save(Path(out) / "fetal_dendrogram.npy", d)
    json.dump({"seconds": round(s, 1), "peak_footprint_gb": round(peak / 2**30, 2),
               "memory_measure": "peak resident memory (Linux getrusage), compute node",
               "graph_loaded_footprint_gb": round(base / 2**30, 2), "n": n, "nnz": int(dx.size)},
              open(Path(out) / "fetal_paris.json", "w"), indent=1)


def fetal() -> None:
    """Paris on the 4.06M-cell graph; cuts scored like the Leiden runs."""
    import polariseq as ps

    sys.path.insert(0, str(ROOT / "benchmarks"))
    import fetal_biology as fb
    from r3_analyses import score_types

    out = OUT / "fetal"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "fetal_graph.npz").exists():
        ps.set_budget(ram_gb=fb.BUDGET_GB, threads=fb.THREADS)
        a = ps.process_diskbacked(str(fb.FILES["4062k"]), **fb.PARAMS)
        _save_csr(out / "fetal_graph.npz", a.obsp["connectivities"])
        np.savetxt(out / "obs_names.txt", np.asarray(a.obs_names, dtype=object), fmt="%s")
        del a
    subprocess.run([sys.executable, __file__, "_fetal_child", str(out)], check=True)
    info = json.loads((out / "fetal_paris.json").read_text())
    print(f"[fetal] Paris on {info['n']:,} cells: {info['seconds']:.0f} s, "
          f"peak footprint {info['peak_footprint_gb']:.2f} GB")
    from polariseq._polariseq import dendrogram_cut_balanced, dendrogram_cut_straight

    d = np.load(out / "fetal_dendrogram.npy")
    obs = np.loadtxt(out / "obs_names.txt", dtype=str)
    scores = {}
    for k in (80, 132, 185, 223, 300, 400):  # Leiden's cluster counts at res 1/2/3/4, and more
        scores[f"straight_{k}"] = score_types(obs, dendrogram_cut_straight(d, k).astype(np.int64))
    d0 = d.copy()
    d0[d0 == np.inf] = 0
    for mx, mn in ((50_000, 5_000), (20_000, 2_000)):
        scores[f"balanced_{mx}_{mn}"] = score_types(obs, dendrogram_cut_balanced(d0, mx, mn, 2.0).astype(np.int64))
    info["cuts"] = {k: {kk: vv for kk, vv in v.items() if kk != "recovered_types"} for k, v in scores.items()}
    (out / "fetal_paris.json").write_text(json.dumps(info, indent=1))
    for k, v in info["cuts"].items():
        print(f"[fetal] {k:22s} recovered {v['recovered']}/{v['of']}, rare {v['rare_recovered']}/{v['rare_of']}, "
              f"own-type cells {v['cells_under_own_type']:.3f}, clusters {v['n_clusters']}")


def _type_scorer(obs):
    """Types recovered (the recovery rule, splits allowed) for many labelings
    of the same cells, without re-reading the 4M-row annotation each time."""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import fetal_biology as fb

    cells = pd.read_csv(fb.CELLS, index_col="cell", usecols=["cell", "Main_cluster_name", "Organ"])
    sub = cells.loc[obs]
    types = pd.Categorical(sub["Main_cluster_name"])
    organs = pd.Categorical(sub["Organ"])
    t_code, n_types = types.codes.astype(np.int64), len(types.categories)

    def score(labels):
        labels = np.asarray(labels, dtype=np.int64)
        k = int(labels.max()) + 1
        ct = np.bincount(t_code * k + labels, minlength=n_types * k).reshape(n_types, k)
        maj = ct.argmax(0)
        top = ct[maj, np.arange(k)]
        held = np.bincount(maj, weights=top, minlength=n_types)
        dom = np.bincount(maj, weights=ct.sum(0), minlength=n_types)
        size_t = ct.sum(1)
        rec = (held / np.maximum(size_t, 1) >= fb.RECOVERY) & (held / np.maximum(dom, 1) >= fb.RECOVERY)
        return {"clusters": k, "recovered": int(rec.sum()), "own_type_cells": round(float(top.sum() / len(labels)), 4)}

    return score, types, organs


def fetal_tree() -> None:
    """Paris on the whole fetal atlas: types recovered along the tree, and the
    tree's top 30 branches annotated by published type and organ."""
    from polariseq._polariseq import dendrogram_cut_straight

    out = OUT / "fetal"
    d = np.load(out / "fetal_dendrogram.npy")
    n = d.shape[0] + 1
    obs = np.loadtxt(out / "obs_names.txt", dtype=str)
    score, types, organs = _type_scorer(obs)

    # The scorer reproduces the comparison's number for Leiden (resolution 1).
    import fetal_biology as fb
    lei = fb.OUT / "4062k"
    lobs = np.loadtxt(lei / "obs_names.txt", dtype=str)
    assert np.array_equal(lobs, obs), "Leiden and Paris runs kept different cells"
    check = score(np.load(lei / "labels.npy"))
    want = json.loads((lei / "comparison.json").read_text())["types_recovered_allowing_splits"]
    assert check["recovered"] == want, (check, want)
    print(f"[fetal_tree] scorer check: Leiden res 1 -> {check['recovered']} types (comparison: {want})")

    curve = []
    for k in (10, 20, 30, 46, 60, 80, 100, 132, 150, 185, 223, 260, 300, 350, 400, 500, 650, 800):
        s = score(dendrogram_cut_straight(d, k))
        curve.append(s)
        print(f"[fetal_tree] cut {k:4d}: {s}")

    # Top of the tree: cut into 30 clusters; the merges above the cut form a
    # tree over them (heights never decrease towards the root).
    k = 30
    labels = dendrogram_cut_straight(d, k).astype(np.int64)
    cut = np.sort(d[:, 2])[n - k]
    top = np.flatnonzero(d[:, 2] >= cut)
    top_set = set((top + n).tolist())

    def leaf_of(node):
        while node >= n:
            node = int(d[node - n, 0])
        return node

    roots = [int(c) for t_ in top for c in d[t_, :2] if int(c) < n or int(c) not in top_set]
    cluster_of_root = {r: int(labels[leaf_of(r)]) for r in roots}
    order = top[np.argsort(d[top, 2], kind="stable")]
    new_id = {}
    Z = []
    for i, t_ in enumerate(order):
        a, b = (int(c) for c in d[t_, :2])
        ia = cluster_of_root.get(a, new_id.get(a))
        ib = cluster_of_root.get(b, new_id.get(b))
        new_id[n + int(t_)] = k + i
        Z.append([ia, ib, float(d[t_, 2]), 0.0])
    assert len(Z) == k - 1 and all(z[0] is not None and z[1] is not None for z in Z)
    leaves = []
    for c in range(k):
        m = labels == c
        tc = np.bincount(types.codes[m], minlength=len(types.categories))
        oc = np.bincount(organs.codes[m], minlength=len(organs.categories))
        leaves.append({"cluster": c, "cells": int(m.sum()),
                       "type": str(types.categories[tc.argmax()]), "type_share": round(float(tc.max() / m.sum()), 3),
                       "organ": str(organs.categories[oc.argmax()]), "organ_share": round(float(oc.max() / m.sum()), 3)})
    info = json.loads((out / "fetal_paris.json").read_text())
    (out / "tree.json").write_text(json.dumps({
        "n_cells": n, "paris_seconds": info["seconds"], "paris_peak_footprint_gb": info["peak_footprint_gb"],
        "curve": curve, "tree_k": k, "Z": Z, "leaves": leaves,
        "leiden": [{"resolution": r, **{kk: json.loads((fb.OUT / ("4062k" if r == "1" else f"4062k_res{r}")
                                                         / "comparison.json").read_text())[kk]
                                         for kk in ("n_clusters", "types_recovered_allowing_splits",
                                                    "cells_in_cluster_of_own_type")}}
                   for r in ("0.5", "1", "2", "3", "4")],
    }, indent=1))
    for lf in leaves:
        print(f"[fetal_tree] branch {lf['cluster']:2d}: {lf['cells']:8,d} cells, {lf['type']} ({lf['type_share']:.0%}), "
              f"{lf['organ']} ({lf['organ_share']:.0%})")


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "_fetal_child":
        _fetal_child(sys.argv[2])
        return 0
    steps = {"export": export, "reference": reference, "scarf": scarf, "compare": compare, "fetal": fetal,
             "fetal_tree": fetal_tree}
    if what not in steps:
        print(__doc__)
        return 2
    steps[what]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
