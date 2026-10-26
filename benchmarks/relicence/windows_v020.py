#!/usr/bin/env python
"""Polariseq 0.2.0 on the Windows laptop at 1.3M and 4.06M cells: timings, true memory use, and the plan.

    .venv\\Scripts\\python.exe benchmarks\\relicence\\windows_v020.py           # resumable
    .venv\\Scripts\\python.exe benchmarks\\relicence\\windows_v020.py --publish-only

Run through benchmarks\\windows\\run_v020.ps1, which rebuilds the library and fetches the data first.

1. Timed runs, the harness and settings of the macOS laptop runs (run_telemetry.py: 4 threads, 6 GB budget,
   warm cache, rotating order, calibrated quiet gate, with UMAP): brain 1.3M and fetal 4.06M, in memory
   (1 repeat: the planner declines both on this laptop) and out of core (3 repeats). Windows records the
   memory the run makes the machine hold (resident set, the growth of the compression store and the pagefile
   in use), the supplement's definition. Records: benchmarks/results/telemetry_win_v020.
2. The full analysis of each dataset (examples/06_large_run.py, the same settings, every figure, the run
   summary): benchmarks/results/v020_windows_final/<dataset>.
3. The plan against the measurement: ps.plan()'s peak for every step (out of core, this budget) beside
   the measured peak of that step in 2 and the measured peak of the whole run in 1: plan_vs_measured.md/.json.
   The run summary also says where each graph step read its inputs (into memory, or in place).
4. Everything but the per-cell tables (too large for GitHub) is pushed to the branch
   results/windows-v020-final, after each part, as benchmarks/platform_runs/windows-v020-final/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

import crossplat_run as xp  # noqa: E402

DATASETS = {   # label: (path, bytes of the file, the same file as on the Mac and the cluster)
    "brain1306k": (REPO / "benchmarks/results/validation_data/brain1306k.h5ad", 21_094_963_453),
    "fetal4062k": (REPO / "benchmarks/results/fetal_data/fetal4062k.h5ad", 6_724_796_853),
}
STAGES = [("brain1306k", "polariseq-ram", 1), ("brain1306k", "polariseq-spill", 3),
          ("fetal4062k", "polariseq-ram", 1), ("fetal4062k", "polariseq-spill", 3)]
RESULTS = REPO / "benchmarks/results/telemetry_win_v020_final"
FULL = REPO / "benchmarks/results/v020_windows_final"
LABEL = "windows-v020-final"
#: planner step -> example 06 step
STEP_MAP = {"load": "open", "qc": "qc (and the one-time copy of the counts)", "normalize+log1p": "normalize",
            "hvg": "hvg", "hvg subset": "renormalize", "pca": "pca", "neighbors": "neighbors",
            "leiden": "leiden", "umap": "umap"}


def check(log: xp.Log) -> None:
    import polariseq
    log.line(f"polariseq {polariseq.__version__} from {polariseq.__file__}")
    if not polariseq.__version__.startswith("0.2.0"):
        xp.die(f"this is polariseq {polariseq.__version__}, not 0.2.0: run git pull and run_v020.ps1 again")
    for name, (path, size) in DATASETS.items():
        if not path.exists() or path.stat().st_size != size:
            xp.die(f"{path} is missing or not {size:,} bytes; run_v020.ps1 fetches it")


def timed_runs(log: xp.Log) -> None:
    import psutil
    # Polariseq's budget is 6 GB (--ram-gb 6, as on the Mac), which fits the 35-45% of this laptop's
    # 16 GB that is free with the system running. A run is stopped once it holds more than the installed
    # memory (the Mac campaign allowed 1.5x): beyond that it would only page, and what it pages is
    # measured below that line anyway.
    vm = psutil.virtual_memory()
    limit = vm.total / 1e9
    log.line(f"memory: {vm.total / 1e9:.1f} GB installed, {vm.available / 1e9:.1f} GB available now; "
             f"budget 6 GB; runs stopped above {limit:.1f} GB")
    if vm.available < 6.5e9:
        log.line("WARNING: less than 6.5 GB available; close programs for a fair measurement")
    campaign = [x for x in xp.CAMPAIGN if x != "--no-umap"] + ["--memory-limit-gb", f"{limit:.2f}"]
    log.line(f"harness arguments: {' '.join(campaign)}")
    for ds, arm, reps in STAGES:
        log.line(f"\n===== {reps} x [{arm}] on {ds}")
        rc = xp.stream([sys.executable, str(BENCH / "run_telemetry.py"), "--datasets", ds, "--arms", arm,
                        *campaign, "--timeout", str(xp.TIMEOUT_S), "--reps", str(reps),
                        "--results", str(RESULTS), "--resume"], log)
        if rc != 0:
            log.line(f"run_telemetry.py exited {rc} on {ds} {arm}; continuing")


def full_analyses(log: xp.Log) -> None:
    for ds, (path, _) in DATASETS.items():
        out = FULL / ds
        if (out / "report.json").exists():
            log.line(f"full analysis {ds}: on record")
            continue
        shutil.rmtree(out, ignore_errors=True)
        project = REPO / "benchmarks/results/tmp" / f"v020_{ds}_project"
        log.line(f"\n===== full analysis {ds} (examples/06_large_run.py)")
        rc = xp.stream([sys.executable, "-u", str(REPO / "examples/06_large_run.py"), str(path), "--out", str(out),
                        "--project", str(project), "--ram-gb", "6", "--threads", "4", "--umap"], log)
        shutil.rmtree(project, ignore_errors=True)
        if rc != 0:
            log.line(f"06_large_run.py exited {rc} on {ds}")


def plan_vs_measured(log: xp.Log) -> dict:
    import polariseq as ps
    from telemetry.record import load_records
    ps.set_budget(ram_gb=6, threads=4)
    out: dict = {}
    md = ["# Plan against measurement, Windows, Polariseq 0.2.0 (4 threads, 6 GB budget, out of core)", ""]
    records = load_records(RESULTS)
    for ds, (path, _) in DATASETS.items():
        plan = ps.plan(str(path), n_hvg=2000, n_pcs=50, storage="spill", verbose=False)
        planned = {s.name: round(s.peak / 1e9, 2) for s in plan.steps}
        rep = FULL / ds / "report.json"
        measured_steps = {}
        if rep.exists():
            measured_steps = {s["step"]: round(s["peak_rss_gb"], 2) for s in json.loads(rep.read_text())["steps"]}
        runs = [r for r in records if str(r["dataset"]["label"]).startswith(ds)
                and r["arm"] == "polariseq-spill" and r.get("status") == "ok"]
        # the supplement's measure: resident set + growth of the compression store + pagefile in use
        demand = [r["parent"]["peak_demand_bytes"] / 1e9 for r in runs if r.get("parent", {}).get("peak_demand_bytes")]
        rss = [r["peaks"].get("child_rss", 0) / 1e9 for r in runs]
        private = [r["parent"].get("os_peak_private", 0) / 1e9 for r in runs]
        ram_runs = [r for r in records if str(r["dataset"]["label"]).startswith(ds) and r["arm"] == "polariseq-ram"]
        out[ds] = {"planned_peak_gb": round(plan.peak_gb, 2), "planned_limiting_step": plan.limiting_step,
                   "planned_steps_gb": planned, "measured_steps_peak_rss_gb": measured_steps,
                   "harness_demand_gb": [round(x, 2) for x in demand], "harness_peak_rss_gb": [round(x, 2) for x in rss],
                   "harness_peak_private_gb": [round(x, 2) for x in private],
                   "in_memory_arm": [(r.get("status"), r.get("note", "")[:120]) for r in ram_runs]}
        md += [f"## {ds}", "",
               f"Planned peak {plan.peak_gb:.2f} GB ({plan.limiting_step}); measured, whole run (median of "
               f"{len(demand)}): demand {statistics.median(demand) if demand else float('nan'):.2f} GB, "
               f"resident {statistics.median(rss) if rss else float('nan'):.2f} GB. In-memory arm: "
               f"{', '.join(s for s, _ in out[ds]['in_memory_arm']) or 'not run'}.", "",
               "| step | planned GB | measured GB (full analysis, resident) |", "|---|---:|---:|"]
        for name, gb in planned.items():
            m = measured_steps.get(STEP_MAP.get(name, name))
            md.append(f"| {name} | {gb:.2f} | {m if m is not None else ''} |")
        md.append("")
    (FULL / "plan_vs_measured.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (FULL / "plan_vs_measured.md").write_text("\n".join(md), encoding="utf-8")
    log.line("\n".join(md))
    return out


def publish(log: xp.Log, push: bool) -> None:
    xp.RESULTS, xp.LABEL, xp.ARMS = RESULTS, LABEL, "polariseq-ram,polariseq-spill"
    stage = xp.report(log)
    if FULL.exists():
        for f in FULL.rglob("*"):
            if f.is_file() and f.name != "cells.tsv.gz":
                dst = stage / "v020_full" / f.relative_to(FULL)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    if push:
        xp.publish(stage, f"results/{LABEL}", "origin", log)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish-only", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args(argv)
    xp.LOGS.mkdir(parents=True, exist_ok=True)
    xp.LABEL = LABEL
    log = xp.Log(xp.LOGS / f"crossplat_{LABEL}_{time.strftime('%Y%m%dT%H%M%S')}.log")
    if not a.publish_only:
        check(log)
        xp.use_repo_scratch()
        xp.keep_awake()
        timed_runs(log)
        publish(log, not a.no_publish)
        full_analyses(log)
        plan_vs_measured(log)
    publish(log, not a.no_publish)
    log.line(f"\ndone. Log: {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
