"""The planner against every measurement of the final 0.2.0 build (2026-10-14).

    .venv/bin/python benchmarks/relicence/planner_v020_check.py

Plans each run's configuration (cells after quality control, threads, budget, mode) and sets the plan beside
what was measured:
- example 06's full analyses (per step): the Mac (benchmarks/results/v020_laptop_final) and Windows
  (benchmarks/platform_runs/windows-v020-final/v020_full);
- the benchmark harness (QC through Leiden as one phase, then UMAP; peaks read from the sampled resident memory):
  the cluster (benchmarks/figures/source_data/v020_cluster), Windows (platform_runs/windows-v020-final/telemetry) and the
  Mac (benchmarks/results/telemetry_v020_final_umap).
Resident memory throughout, which counts the pages of mapped files, as the plan does. On macOS a step's resident
memory also holds what earlier steps freed (the system takes those pages back only when it needs them), so the
Mac's steps are set against the plan's highest step so far.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import polariseq.planning as P

ROOT = Path(__file__).resolve().parents[2]
SHAPES = {"brain1306k": (1306127, 27998, 2624828308), "fetal4062k": (4062980, 63561, 2631157163)}
KEPT = {"brain1306k": 1306127, "fetal4062k": 4000604}  # cells after quality control
GB = 1e9


def plan(ds: str, threads: int, ram_gb: float, storage: str) -> dict[str, float]:
    n, g, z = SHAPES[ds]
    p = P.estimate(n, g, z, P.Budget(int(ram_gb  * 1e9), threads, "/tmp", 10**13), storage=storage)
    steps = {s.name: s.peak / GB for s in p.steps}
    # the graph steps work on the cells kept by quality control
    q = P.estimate(KEPT[ds], g, z * KEPT[ds] // n, P.Budget(int(ram_gb  * 1e9), threads, "/tmp", 10**13),
                   storage=storage)
    for s in q.steps:
        if s.name in ("neighbors", "leiden", "umap"):
            steps[s.name] = s.peak / GB
    steps["whole"] = max(steps.values())
    steps["through leiden"] = max(v for k, v in steps.items() if k not in ("umap", "whole"))
    return steps


def full_analyses():
    for plat, pat in (("mac", "benchmarks/results/v020_laptop_final/*/report.json"),
                      ("windows", "benchmarks/platform_runs/windows-v020-final/v020_full/*/report.json")):
        for f in sorted(glob.glob(str(ROOT / pat))):
            r = json.loads(Path(f).read_text())
            ds = Path(f).parent.name
            pl = plan(ds, r["budget"]["threads"], r["budget"]["ram_gb"], "spill")
            so_far = 0.0
            for s in r["steps"]:
                name = {"neighbors": "neighbors", "leiden": "leiden", "umap": "umap", "pca": "pca",
                        "hvg": "hvg"}.get(s["step"])
                if name:
                    so_far = max(so_far, pl[name])
                    p = so_far if plat == "mac" else pl[name]
                    yield plat, "06 full", ds, name, p, s["peak_rss_gb"], (r.get("graph_inputs") or {}).get(name)


def harness():
    pats = (("param", "benchmarks/figures/source_data/v020_cluster/*.json"),
            ("windows", "benchmarks/platform_runs/windows-v020-final/telemetry/*.json"),
            ("mac", "benchmarks/results/telemetry_v020_final_umap/*.json"),
            # the 0.2.0 laptop series, once run
            ("mac", "benchmarks/results/telemetry_v020/*.json"),
            ("windows", "benchmarks/platform_runs/windows-v020-series/telemetry/*.json"))
    for plat, pat in pats:
        for f in sorted(glob.glob(str(ROOT / pat))):
            r = json.loads(Path(f).read_text())
            if r["arm"] != "polariseq-spill" or r["status"] != "ok":
                continue
            ds = Path(f).name.split("__")[0]
            if ds not in SHAPES:
                d = r["dataset"]
                SHAPES[ds] = (d["n_cells"], d["n_genes"], d["nnz"])
                KEPT[ds] = (r.get("outputs") or {}).get("n_obs_after") or d["n_cells"]
            pl = plan(ds, r["params"]["threads"], r["params"]["ram_gb"], "spill")
            s = r["parent"]["samples"]
            seg = {st["name"]: (st["t_begin"], st["t_end"]) for st in r["steps"]}
            for phase, key in (("preprocess", "through leiden"), ("umap", "umap")):
                if phase not in seg or (phase == "umap" and not r["params"].get("with_umap")):
                    continue
                a, b = seg[phase]
                inside = [x for t, x in zip(s["t"], s["rss"]) if a <= t <= b]
                if not inside:  # a run too short for a sample inside the phase: its peak, for a run without UMAP
                    if r["params"].get("with_umap"):
                        continue
                    inside = [r["parent"].get("peak_rss") or max(s["rss"])]
                peak = max(inside) / GB
                p = pl["whole"] if (plat == "mac" and key == "umap") else pl[key]
                yield plat, f"harness t{r['params']['threads']}", ds, key, p, peak, None


def in_memory():
    """The in-memory runs, step by step (the cluster: every step is timed) and as a whole (the laptop series)."""
    pats = (("param", "benchmarks/figures/source_data/v020_cluster/*polariseq-ram*.json"),
            ("mac", "benchmarks/results/telemetry_v020/*__polariseq-ram__*.json"),
            ("windows", "benchmarks/platform_runs/windows-v020-series/telemetry/*__polariseq-ram__*.json"))
    alias = {"filter_cells": "qc", "log1p": "normalize+log1p", "normalize": "normalize+log1p",
             "renormalize": "hvg subset"}
    for plat, pat in pats:
        for f in sorted(glob.glob(str(ROOT / pat))):
            r = json.loads(Path(f).read_text())
            if r["status"] != "ok":
                continue
            d = r["dataset"]
            p = P.estimate(d["n_cells"], d["n_genes"], d["nnz"],
                           P.Budget(int(r["params"]["ram_gb"]  * 1e9), r["params"]["threads"], "/tmp", 10**13),
                           storage="ram", with_umap=r["params"].get("with_umap", False))
            planned = {x.name: x.peak / GB for x in p.steps}
            s = r["parent"]["samples"]
            src = f"in memory t{r['params']['threads']}"
            yield plat, src, d["label"], "whole", p.peak_gb, max(s["rss"]) / GB, None
            if plat == "mac":
                continue  # macOS keeps freed pages resident; the whole run is the comparison there
            for st in r["steps"]:
                key = alias.get(st["name"], st["name"])
                v = [x for t, x in zip(s["t"], s["rss"]) if st["t_begin"] <= t <= st["t_end"]]
                if v and key in planned:
                    yield plat, src, d["label"], st["name"], planned[key], max(v) / GB, None


def main() -> None:
    print(f"{'platform':8} {'source':12} {'dataset':11} {'step':15} {'plan':>6} {'measured':>8} {'plan/meas':>9}  inputs")
    worst = []
    for plat, src, ds, step, p, m, inp in [*full_analyses(), *harness(), *in_memory()]:
        print(f"{plat:8} {src:12} {ds:11} {step:15} {p:6.2f} {m:8.2f} {p / m:9.2f}  {inp or ''}")
        worst.append((p / m, plat, src, ds, step))
    worst.sort()
    print("lowest plan/measured:", [f"{w[0]:.2f} {w[1]} {w[2]} {w[3]} {w[4]}" for w in worst[:4]])


if __name__ == "__main__":
    main()
