"""The fetal atlas by one clustering of all cells, at five resolutions, from the final 0.2.0's node run
Run on the cluster (env py-polariseq-020-final):

    python v020_fetal_resolutions.py OUT_DIR

The out-of-core run of 4,062,980 fetal cells (runs/v020/fetal4062k_polariseq-spill_t4) gives the
principal components, the UMAP and the cells kept; the neighbour graph is built again from its components with
the run's settings (15 neighbours, seed 0), which at resolution 1 gives the run's own clusters (checked). Leiden
then runs at resolutions 0.5, 1, 2, 3 and 4 (seed 0), and for each the per-cluster statistics of every gene are
streamed from the file (normalized to 10,000 counts per cell) and the top Welch markers ranked, as
benchmarks/fetal_biology.py does. Writes, per resolution, the files fetal_biology.py writes (OUT_DIR/4062k for
resolution 1, OUT_DIR/4062k_res<r> for the others), so that `fetal_biology.py compare` scores them unchanged;
and the clusters at resolution 1 with seeds 0, 1 and 2 (OUT_DIR/seeds), for the seed-to-seed spread of the
agreement with the published types.
"""
import os
import glob
import json
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import polariseq as ps
from polariseq._polariseq import streaming_cluster_stats_normalized

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_analysis import welch_markers  # noqa: E402

B = os.environ["B"]  # the working folder (REPRODUCE.md)
RUN = glob.glob(f"{B}/runs/v020/fetal4062k_polariseq-spill_t4/artifacts/*/")[0]
FILE = f"{B}/data/paper/fetal_data/fetal4062k.h5ad"
PARAMS = dict(n_hvg=2000, n_pcs=50, n_neighbors=15, target_sum=1e4, seed=0, min_genes=200)
RESOLUTIONS = (1.0, 0.5, 2.0, 3.0, 4.0)


def main() -> None:
    out_root = Path(sys.argv[1])
    obs = np.array([s for s in open(RUN + "obs_names.txt").read().split("\n") if s], dtype=object)
    run_labels = np.load(RUN + "labels.npy")
    umap = np.load(RUN + "umap.npy").astype(np.float32)
    hvg = open(RUN + "hvg.txt").read().split()
    a = anndata.AnnData(np.zeros((len(obs), 1), dtype=np.float32))
    a.obsm["X_pca"] = np.load(RUN + "pca.npy").astype(np.float32)
    t0 = time.perf_counter()
    ps.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", seed=0)
    t_graph = time.perf_counter() - t0
    print(f"graph of {len(obs):,} cells: {t_graph:.0f}s", flush=True)
    backed = anndata.read_h5ad(FILE, backed="r")
    full = pd.Index(np.asarray(backed.obs_names, dtype=object))
    var_names = [str(v) for v in backed.var_names]
    backed.file.close()
    pos = full.get_indexer(pd.Index(obs))
    assert (pos >= 0).all()
    seeds = out_root / "seeds"
    seeds.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2):   # Section 2.2: the agreement with the published types across three seeds
        ps.tl.leiden(a, resolution=1.0, seed=seed)
        np.save(seeds / f"labels_res1_seed{seed}.npy", np.asarray(a.obs["leiden"]).astype(np.uint32))
    np.savetxt(seeds / "obs_names.txt", obs, fmt="%s")
    for res in RESOLUTIONS:
        out = out_root / ("4062k" if res == 1.0 else f"4062k_res{res:g}")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        ps.tl.leiden(a, resolution=res, seed=PARAMS["seed"])
        labels = np.asarray(a.obs["leiden"]).astype(np.uint32)
        t_leiden = time.perf_counter() - t0
        if res == 1.0:
            assert (pd.crosstab(labels, run_labels).astype(bool).sum(axis=1) == 1).all(), "not the run's clusters"
        n_clusters = int(labels.max()) + 1
        labels_full = np.full(len(full), n_clusters, dtype=np.uint32)
        labels_full[pos] = labels
        t0 = time.perf_counter()
        stats = streaming_cluster_stats_normalized(FILE, labels_full, n_clusters, PARAMS["target_sum"])
        t_stats = time.perf_counter() - t0
        sizes = np.asarray(stats["cluster_sizes"], dtype=np.int64)
        np.savez_compressed(out / "stats.npz", sums=stats["sums"], sq_sums=stats["sq_sums"], det=stats["counts"],
                            sizes=sizes)
        (out / "var_names.txt").write_text("\n".join(var_names))
        welch_markers(stats, var_names, n_top=50).to_csv(out / "markers.csv", index=False)
        np.save(out / "labels.npy", labels)
        np.save(out / "umap.npy", umap)
        np.savetxt(out / "obs_names.txt", obs, fmt="%s")
        (out / "hvg.txt").write_text("\n".join(sorted(hvg)))
        (out / "summary.json").write_text(json.dumps({
            "path": "benchmarks/results/fetal_data/fetal4062k.h5ad", "params": {**PARAMS, "resolution": res},
            "budget_gb": 6.0, "threads": 4, "polariseq": ps.__version__,
            "source_run": RUN, "n_cells_file": int(len(full)), "n_cells_kept": int(len(obs)),
            "n_clusters": n_clusters, "leiden_quality": float(a.uns["leiden"]["quality"]),
            "timing_s": {"graph": round(t_graph, 1), "leiden": round(t_leiden, 1),
                         "streaming_stats": round(t_stats, 1)}}, indent=1))
        print(f"resolution {res:g}: {n_clusters} clusters, leiden {t_leiden:.0f}s, stats {t_stats:.0f}s", flush=True)


if __name__ == "__main__":
    main()
