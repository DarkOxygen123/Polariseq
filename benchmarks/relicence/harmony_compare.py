#!/usr/bin/env python
"""Harmony of Polariseq (0.1.x and the replacement) against harmonypy 2.0.2, on the same embeddings.

    python harmony_compare.py embed DATASET --out E.npz            # the default pipeline up to PCA
    python harmony_compare.py run E.npz --vars condition[,donor] --out R.json [--seeds 5]
                                        [--arms harmonypy,new]

embed   saves the PCA embedding, the batch variables and the published labels of a dataset.
run     corrects it with every arm and seed (harmonypy's defaults, which are Polariseq's) and reports:
        `diff`, how far two corrected embeddings are apart relative to the size of the correction,
        ||Z_a - Z_b|| / ||Z_b - Z_input||, for every arm against every harmonypy run (next to
        harmonypy's own seed-to-seed value); `knn_shared`, the share of each cell's 15 nearest
        neighbours two runs have in common (5,000 cells); `mixing`, the batch entropy among each cell's
        30 neighbours scaled to [0, 1] (the higher the better mixed); `type_kept`, the share of the 30
        neighbours of the same published type; and the seconds. The arm `old` (0.1.x) ran on the
        development build that held both implementations behind the POLARISEQ_HARMONY_IMPL switch; the
        release has the new Harmony only, so with it the arms are `harmonypy` and `new`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "examples/data/kang_2018"
F = ROOT / "benchmarks/results/fetal_data"
DATASETS = {   # files, label column, PCA components, extra batch columns
    "kang": ({"control": K / "control.h5ad", "IFN-beta": K / "ifn_beta.h5ad"}, "published_type", 30),
    "fetal60k": (F / "fetal60k.h5ad", "fetal", 50),
}


def embed(a) -> int:
    import pandas as pd
    import polariseq as ps
    files, label, n_pcs = DATASETS[a.dataset]
    ps.set_budget(ram_gb=6, threads=4)
    if isinstance(files, dict):
        ad = ps.read_h5ad({k: str(v) for k, v in files.items()}, batch_key="condition")
    else:
        ad = ps.read_h5ad(str(files))
    ps.pp.calculate_qc_metrics(ad)
    ps.pp.filter_cells(ad, min_genes=200)
    ps.pp.filter_genes(ad, min_cells=3)
    ps.pp.normalize_total(ad, target_sum=1e4)
    ps.pp.log1p(ad)
    ps.pp.highly_variable_genes(ad, n_top_genes=2000)
    ps.pp.pca(ad, n_components=n_pcs)
    out = {"pca": np.ascontiguousarray(ad.obsm["X_pca"], dtype=np.float32)}
    if label == "fetal":
        meta = pd.read_csv(F / "fetal_cells.csv.gz", usecols=["cell", "Main_cluster_name", "Fetus_id", "Organ"])
        meta = meta.set_index("cell").reindex(np.asarray(ad.obs_names).astype(str))
        out["labels"] = meta["Main_cluster_name"].fillna("NA").to_numpy().astype(str)
        out["batch_fetus"] = meta["Fetus_id"].fillna("NA").to_numpy().astype(str)
        out["batch_organ"] = meta["Organ"].fillna("NA").to_numpy().astype(str)
    else:
        out["labels"] = np.asarray(ad.obs[label]).astype(str)
        out["batch_condition"] = np.asarray(ad.obs["condition"]).astype(str)
        out["batch_donor"] = np.asarray(ad.obs["donor"]).astype(str)
    np.savez(a.out, **out)
    print(f"{a.dataset}: {ad.n_obs:,} cells, {n_pcs} components -> {a.out}")
    return 0


def run_arm(arm, x, batches, seed):
    if arm == "harmonypy":
        import harmonypy
        import pandas as pd
        meta = pd.DataFrame({f"v{i}": b for i, b in enumerate(batches)})
        t0 = time.perf_counter()
        ho = harmonypy.run_harmony(x, meta, list(meta.columns), verbose=False, random_state=seed)
        sec = time.perf_counter() - t0
        z = np.asarray(ho.Z_corr)
        return (z.T if z.shape[0] != x.shape[0] else z).astype(np.float32), sec
    from polariseq._polariseq import harmony_correct
    if arm == "old":
        raise SystemExit("arm 'old' needs the development build with the 0.1.x Harmony; the release has the "
                         "new Harmony only (arms: harmonypy, new)")
    os.environ["POLARISEQ_HARMONY_IMPL"] = arm
    codes, levels = [], []
    for b in batches:
        u, c = np.unique(b, return_inverse=True)
        codes.append(c.astype(np.uint32))
        levels.append(len(u))
    out = np.empty_like(x)
    t0 = time.perf_counter()
    harmony_correct(x, codes, levels, out, theta=[2.0], sigma=0.1, lamb=None, n_clusters=None, tau=0.0,
                    block_size=0.05, max_iter_harmony=10, max_iter_kmeans=4, epsilon_cluster=1e-3,
                    epsilon_harmony=1e-2, alpha=0.2, batch_prop_cutoff=1e-5, seed=seed)
    return out, time.perf_counter() - t0


def neighbours(z, k, query):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    return nn.kneighbors(z[query], return_distance=False)[:, 1:]


def run(a) -> int:
    e = np.load(a.embedding, allow_pickle=False)
    x = np.ascontiguousarray(e["pca"], dtype=np.float32)
    names = a.vars.split(",")
    batches = [e[f"batch_{v}"] for v in names]
    labels = e["labels"]
    rng = np.random.default_rng(0)
    query = rng.choice(len(x), size=min(5000, len(x)), replace=False)
    _, lab = np.unique(labels, return_inverse=True)
    _, bat = np.unique(batches[0], return_inverse=True)
    n_bat = bat.max() + 1
    runs: dict[str, list[np.ndarray]] = {}
    rep = {"cells": len(x), "dims": x.shape[1], "vars": names, "arms": {}}
    for arm in a.arms.split(","):
        runs[arm], rows = [], []
        for s in range(a.seeds):
            z, sec = run_arm(arm, x, batches, s)
            runs[arm].append(z)
            nb = neighbours(z, 30, query)
            share = np.stack([np.bincount(row, minlength=n_bat) for row in bat[nb]]) / 30.0
            ent = -(np.where(share > 0, share * np.log(share), 0)).sum(1) / np.log(min(n_bat, 30))
            rows.append({"seed": s, "seconds": round(sec, 2), "mixing": round(float(np.median(ent)), 4),
                         "type_kept": round(float((lab[nb] == lab[query, None]).mean()), 4)})
            print(f"  {arm} seed {s}: {rows[-1]}", flush=True)
        rep["arms"][arm] = {"runs": rows}
    ref = runs["harmonypy"]
    nb_ref = [neighbours(z, 15, query) for z in ref]

    def compare(zs, self_pairs):
        pairs = list(combinations(range(len(zs)), 2)) if self_pairs else [(i, j) for i in range(len(zs))
                                                                           for j in range(len(ref))]
        d, kn = [], []
        for i, j in pairs:
            za, zb = zs[i], (zs[j] if self_pairs else ref[j])
            d.append(float(np.linalg.norm(za - zb) / np.linalg.norm(zb - x)))
            na = neighbours(za, 15, query)
            nbb = nb_ref[j] if not self_pairs else neighbours(zb, 15, query)
            kn.append(float(np.mean([len(set(p) & set(q)) / 15 for p, q in zip(na, nbb)])))
        return {"diff": [round(min(d), 4), round(float(np.median(d)), 4), round(max(d), 4)],
                "knn_shared": [round(min(kn), 4), round(float(np.median(kn)), 4), round(max(kn), 4)]}

    rep["harmonypy_seed_to_seed"] = compare(ref, True)
    for arm, zs in runs.items():
        if arm != "harmonypy":
            rep["arms"][arm]["vs_harmonypy"] = compare(zs, False)
            rep["arms"][arm]["seed_to_seed"] = compare(zs, True)
    for arm, v in rep["arms"].items():
        r = v["runs"]
        v["median"] = {k: float(np.median([q[k] for q in r])) for k in ("seconds", "mixing", "type_kept")}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("cells", "vars", "harmonypy_seed_to_seed")}))
    for arm, v in rep["arms"].items():
        print(arm, v["median"], {k: v[k] for k in ("vs_harmonypy", "seed_to_seed") if k in v})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("embed")
    g.add_argument("dataset")
    g.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("embedding")
    r.add_argument("--vars", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--arms", default="harmonypy,new")
    a = ap.parse_args()
    return embed(a) if a.cmd == "embed" else run(a)


if __name__ == "__main__":
    sys.exit(main())
