"""Write the harness's run records (one JSON per run) as one table per campaign.

    python benchmarks/export_tables.py RECORD_FOLDER OUT.csv [--campaign NAME]

One row per run: the dataset and its size, the tool (arm), the repeat, the outcome, the parameters,
the time of every step (pipeline_s: the analysis without process start-up and imports), the peak memory
measures, the machine, and the software versions. The memory time series stay in the records; this table is
what the comparisons are read from.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

#: "preprocess" is the out-of-core arm's single call (ps.process_diskbacked: quality control through clustering),
#: which the harness times as one step in place of the separate steps.
STEPS = ["load", "preprocess", "qc", "filter_cells", "normalize", "log1p", "hvg", "renormalize", "pca", "neighbors",
         "leiden", "umap"]


def row(d: dict, campaign: str) -> dict:
    ds, p, env, out = d.get("dataset", {}), d.get("params", {}), d.get("env", {}), d.get("outputs", {})
    peaks, parent = d.get("peaks", {}), d.get("parent", {})
    steps = {s.get("name"): s.get("wall_s") for s in d.get("steps", [])}
    gb = lambda b: round(b / 1e9, 3) if b else ""
    r = {
        "campaign": campaign, "run_id": d.get("run_id"), "timestamp": d.get("timestamp"),
        "dataset": ds.get("label"), "n_cells": ds.get("n_cells"), "n_genes": ds.get("n_genes"),
        "stored_values": ds.get("nnz"), "tool": d.get("arm"), "tool_label": d.get("arm_label"),
        "workflow": out.get("workflow", ""), "rep": d.get("rep"), "status": d.get("status"),
        "note": d.get("note", ""), "threads": p.get("threads"), "budget_gb": p.get("ram_gb"),
        "with_umap": p.get("with_umap"), "n_hvg": p.get("n_hvg"), "n_pcs": p.get("n_pcs"),
        "n_neighbors": p.get("n_neighbors"), "resolution": p.get("resolution"), "min_genes": p.get("min_genes"),
        "seed": p.get("seed"), "wall_s": d.get("wall_s"),
        "pipeline_s": round(sum(v for k, v in steps.items() if k in STEPS and v), 3) if steps else "",
    }
    for s in STEPS:
        r[f"{s}_s"] = steps.get(s, "")
    r.update({
        "peak_footprint_gb": gb(peaks.get("footprint")), "peak_rss_gb": gb(peaks.get("parent_rss")),
        "peak_uss_gb": gb(peaks.get("child_uss")), "min_free_gb": gb(parent.get("min_sys_available")),
        "cells_kept": out.get("n_obs_after", ""), "genes_kept": out.get("n_vars_after", ""),
        "clusters": out.get("n_clusters", ""), "storage": out.get("storage", ""),
        "threads_used": out.get("threads_used", ""), "cpu": env.get("cpu"),
        "ram_gb": gb(env.get("total_ram_bytes")), "os": f"{env.get('platform', '')} {env.get('release', '')}".strip(),
    })
    for k, v in (env.get("versions") or {}).items():
        r[f"version_{k}"] = v
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records")
    ap.add_argument("out")
    ap.add_argument("--campaign", default=None)
    a = ap.parse_args()
    campaign = a.campaign or os.path.basename(os.path.normpath(a.records))
    rows = [row(json.load(open(f)), campaign) for f in sorted(glob.glob(os.path.join(a.records, "*.json")))]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {a.out}")


if __name__ == "__main__":
    main()
