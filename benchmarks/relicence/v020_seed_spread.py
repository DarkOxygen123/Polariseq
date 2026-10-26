"""Leiden's seed-to-seed agreement on the graphs of the final 0.2.0's compute-node runs (Section 2.1).

    python v020_seed_spread.py OUT_JSON           # on the cluster, env py-polariseq-020-final

For the mouse brain and the fetal atlas, in memory and out of core (4 threads, runs/v020): the neighbour graph is
built again from the run's principal components with the run's settings (15 neighbours, seed 0), which gives the
run's own clusters at Leiden seed 0 (checked: adjusted Rand index 1 against the run's labels); then Leiden at
resolution 1.0 with seeds 0 to 9, and the adjusted Rand index of every pair of seeds.
"""
import os
import glob
import itertools
import json
import statistics
import sys
import time

import anndata
import numpy as np
import polariseq as ps


def ari(a, b) -> float:
    """Adjusted Rand index (Hubert and Arabie 1985), from the contingency table."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.bincount(ai * (bi.max() + 1) + bi, minlength=(ai.max() + 1) * (bi.max() + 1)).astype(np.float64)

    def comb2(x):
        return float((x * (x - 1) / 2).sum())

    n = comb2(np.array([len(a)], dtype=np.float64))
    sum_ij = comb2(table)
    sum_a = comb2(np.bincount(ai).astype(np.float64))
    sum_b = comb2(np.bincount(bi).astype(np.float64))
    expected = sum_a * sum_b / n
    return (sum_ij - expected) / (0.5 * (sum_a + sum_b) - expected)

V020 = os.path.join(os.environ["B"], "runs/v020")
out = {}
for ds in ("brain1306k", "fetal4062k"):
    for arm in ("polariseq-ram", "polariseq-spill"):
        d = glob.glob(f"{V020}/{ds}_{arm}_t4/artifacts/*/")[0]
        pca = np.load(d + "pca.npy").astype(np.float32)
        run_labels = np.load(d + "labels.npy")
        a = anndata.AnnData(np.zeros((len(pca), 1), dtype=np.float32))
        a.obsm["X_pca"] = pca
        t0 = time.perf_counter()
        ps.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", seed=0)
        labels = {}
        for seed in range(10):
            ps.tl.leiden(a, resolution=1.0, seed=seed)
            labels[seed] = np.asarray(a.obs["leiden"]).astype(int)
        pairs = [ari(labels[i], labels[j]) for i, j in itertools.combinations(range(10), 2)]
        key = f"{ds} {arm}"
        out[key] = {"run_reproduced_at_seed_0": float(ari(labels[0], run_labels)),
                    "clusters": [int(len(set(v))) for v in labels.values()],
                    "seed_pairs_ari_median": statistics.median(pairs), "seed_pairs_ari_min": min(pairs),
                    "seed_pairs_ari_max": max(pairs), "seed_pairs_ari_quartiles": statistics.quantiles(pairs, n=4),
                    "seconds": round(time.perf_counter() - t0, 1)}
        print(key, out[key], flush=True)
json.dump({"polariseq": ps.__version__, "runs": out}, open(sys.argv[1], "w"), indent=1)
