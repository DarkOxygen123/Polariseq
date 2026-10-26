#!/usr/bin/env python
"""Polariseq arm of the showdown, installed from the repository (benchmarks/atlas/envs/py-polariseq.txt).

    python drv_polariseq.py convert --input SHARDS --project DIR --out ART --threads 16 --ram-gb 160
    python drv_polariseq.py run     --input SHARDS --project DIR --out ART --threads 16 --ram-gb 160 \
                                    [--doublets flag|remove|none] [--batch-cols dataset_id,donor_id]

convert  reads the files and writes Polariseq's copy of the counts (the block store) into PROJECT
run      the pipeline of examples/06_large_run.py under the 'polariseq' workflow (the defaults of the
         paper), with Scrublet after quality control. The copy of the counts made by convert is reused.

Doublets: ps.pp.scrublet scores and flags every cell, per group of --batch-cols (doublets form within
one capture; dataset and donor stand in for it). 'flag' keeps every cell and writes the flags for review,
which is what a researcher does first; 'remove' drops the flagged cells with the public subset
(adata[mask]) before normalization; 'none' skips the step.

Artifacts in --out after the timed steps: global_rows.npy (each analyzed cell's row in the
concatenated input), labels.npy, umap.npy, hvg.txt, and for doublets qc_rows.npy, doublet_score.npy,
predicted_doublet.npy (over the cells that passed quality control).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, Steps, emit, shard_files, shard_sizes  # noqa: E402


def global_rows(adata, files: list[str], offsets: dict[str, int]):
    """Each cell's row in the concatenated input: file offset + row within the file. Polariseq
    names a cell by its row in its file and records the file stem in obs['dataset']."""
    import numpy as np
    names = np.asarray(adata.obs_names, dtype=np.int64)
    if len(files) == 1:
        return names
    ds = adata.obs["dataset"]
    off = np.array([offsets[str(c)] for c in ds.categories], dtype=np.int64)
    return off[np.asarray(ds.codes)] + names


def batch_column(adata, cols: list[str]):
    """One categorical naming each cell's (dataset, donor) group, from the obs columns given."""
    import numpy as np
    import pandas as pd
    codes = np.zeros(adata.n_obs, dtype=np.int64)
    for c in cols:
        v = pd.Categorical(adata.obs[c])
        codes = codes * (len(v.categories) + 1) + np.asarray(v.codes, dtype=np.int64)
    uniq, inv = np.unique(codes, return_inverse=True)
    return pd.Categorical.from_codes(inv.astype(np.int32), categories=[f"g{i}" for i in range(len(uniq))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["convert", "run"])
    ap.add_argument("--input", required=True, help="folder, pattern or comma-separated .h5ad files")
    ap.add_argument("--project", required=True, help="Polariseq's working folder (the block store)")
    ap.add_argument("--out", required=True, help="artifacts folder")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--ram-gb", type=float, default=160)
    ap.add_argument("--mode", choices=["auto", "memory", "out-of-core"], default="auto")
    ap.add_argument("--doublets", choices=["flag", "remove", "none"], default="flag")
    ap.add_argument("--batch-cols", default="dataset_id,donor_id")
    ap.add_argument("--no-umap", action="store_true")
    a = ap.parse_args()
    st = Steps()
    files = shard_files(a.input)
    sizes = shard_sizes(files)
    offsets = {Path(f).stem: int(sum(sizes[:i])) for i, f in enumerate(files)}

    with st("import"):
        import numpy as np
        import polariseq as ps
    ps.set_budget(ram_gb=a.ram_gb, threads=a.threads)
    ps.set_workflow("polariseq")
    project = ps.project(a.project)
    art = Path(a.out)
    art.mkdir(parents=True, exist_ok=True)
    modes = {} if a.mode == "auto" else {"mode": a.mode}

    with st("load"):
        adata = ps.read_h5ad(files if len(files) > 1 else files[0], **modes)
    summary = {"tool": "polariseq", "version": ps.__version__, "build": os.environ.get("PS_BUILD", "installed release"),
               "phase": a.phase, "mode": adata.mode,
               "cells_in": int(adata.n_obs), "genes_in": int(adata.n_vars), "threads": a.threads,
               "ram_gb": a.ram_gb}

    if a.phase == "convert":
        with st("convert"):
            ps.pp.calculate_qc_metrics(adata)   # out of core this writes the copy of the counts
        summary["project_gb"] = du_gb(project.path)
        emit(summary)
        return 0

    with st("qc"):
        ps.pp.calculate_qc_metrics(adata)
        ps.pp.filter_cells(adata, min_genes=P["min_genes"])
        ps.pp.filter_genes(adata, min_cells=P["min_cells"])
    summary.update(cells_after_qc=int(adata.n_obs), genes_after_qc=int(adata.n_vars))

    qc_rows = global_rows(adata, files, offsets)
    if a.doublets != "none":
        d = P["doublet"]
        with st("doublets"):
            adata.obs["doublet_group"] = batch_column(adata, a.batch_cols.split(","))
            ps.pp.scrublet(adata, batch_key="doublet_group", sim_doublet_ratio=d["sim_ratio"],
                           expected_doublet_rate=d["expected_rate"], n_prin_comps=d["n_pcs"],
                           min_counts=d["min_counts"], min_cells=d["min_cells"],
                           min_gene_variability_pctl=d["variability_pctl"], random_state=P["seed"])
        score = np.asarray(adata.obs["doublet_score"], dtype=np.float32)
        flag = np.asarray(adata.obs["predicted_doublet"], dtype=bool)
        np.save(art / "qc_rows.npy", qc_rows)
        np.save(art / "doublet_score.npy", score)
        np.save(art / "predicted_doublet.npy", flag)
        groups = np.asarray(adata.obs["doublet_group"].codes)
        summary.update(doublets_flagged=int(flag.sum()), doublet_groups=int(groups.max()) + 1,
                       doublet_score_nan=int(np.isnan(score).sum()))
        if a.doublets == "remove":
            with st("doublet_removal"):
                adata = adata[~flag]
            summary["cells_after_doublets"] = int(adata.n_obs)

    with st("normalize"):
        ps.pp.normalize_total(adata, target_sum=P["target_sum"])
        ps.pp.log1p(adata)
    with st("hvg"):
        ps.pp.highly_variable_genes(adata, n_top_genes=P["n_hvg"], subset=True)
    with st("renormalize"):
        ps.pp.renormalize(adata, target_sum=1000)
    with st("pca"):
        ps.pp.pca(adata, n_components=P["n_pcs"], seed=P["seed"], scale=True)
    with st("neighbors"):
        ps.pp.neighbors(adata, n_neighbors=P["neighbors"], seed=P["seed"])
    with st("leiden"):
        ps.tl.leiden(adata, resolution=P["resolution"], seed=P["seed"])
    if not a.no_umap:
        with st("umap"):
            ps.tl.umap(adata, seed=P["seed"], n_epochs=P["umap_epochs"])

    labels = np.asarray(adata.obs["leiden"]).astype(np.int32)
    np.save(art / "global_rows.npy", global_rows(adata, files, offsets))
    np.save(art / "labels.npy", labels)
    if "X_umap" in adata.obsm:
        np.save(art / "umap.npy", np.asarray(adata.obsm["X_umap"], dtype=np.float32))
    (art / "hvg.txt").write_text("\n".join(map(str, adata.var_names)) + "\n")
    summary.update(cells_analyzed=int(adata.n_obs), hvg=int(adata.n_vars),
                   clusters=int(len(np.unique(labels))), project_gb=du_gb(project.path))
    emit(summary)
    return 0


def du_gb(path) -> float | None:
    try:
        return round(int(subprocess.run(["du", "-sk", str(path)], capture_output=True,
                                        text=True).stdout.split()[0]) * 1024 / 1e9, 2)
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
