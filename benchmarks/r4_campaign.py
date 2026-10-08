"""The Polariseq runs of the laptop campaign again, with Polariseq's workflow as default.

    .venv/bin/python benchmarks/r4_campaign.py [--allow-dirty]     # resumable
    .venv/bin/python benchmarks/r4_campaign.py --list
    .venv/bin/python benchmarks/r4_campaign.py --stage ladder        # one stage again (resumes)
    .venv/bin/python benchmarks/r4_campaign.py --control             # the control below
    .venv/bin/python benchmarks/r4_campaign.py --umap [--control]    # the same with UMAP

The same harness, datasets, repeats, threads (4), budget (6 GB), memory measure (per-process peak
footprint) and memory ceiling (1.5 x the laptop's memory) as the Polariseq stages of
benchmarks/r3_campaign.py, into benchmarks/results/telemetry_r4. Only Polariseq is re-run: the
other tools are unchanged, and their records in telemetry_r3 stand. Each record states the workflow
it ran under (`outputs.workflow`).

--control runs a few points of the same stages with the published workflow (POLARISEQ_WORKFLOW=scanpy)
into benchmarks/results/telemetry_r4_scanpy, in the same session as the main runs, so that the cost of
the workflow can be told apart from the state of the machine (the first campaign ran on a freshly started
machine, this one with less memory available).

--umap runs the stages (or, with --control, the control) with UMAP, into
benchmarks/results/telemetry_r4_umap (control: telemetry_r4_umap_scanpy). The other tools' runs with
UMAP go to the same folder from ``r3_campaign.py --umap``; they are read together.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crossplat_run as xp  # noqa: E402
import final_campaign as fc  # noqa: E402

REPO = HERE.parent
fc.RESULTS = REPO / "benchmarks/results/telemetry_r4"
MEMORY_LIMIT_GB = 1.5 * 8.59
xp.CAMPAIGN = [*xp.CAMPAIGN, "--memory-limit-gb", f"{MEMORY_LIMIT_GB:.2f}"]

STAGES: list[tuple[str, str, str, int]] = [
    ("public", fc.PUBLIC, "polariseq-ram,polariseq-spill", 5),
    ("ladder", fc.LADDER, fc.POLARISEQ, 3),
    ("fetal", fc.FETAL, "polariseq-spill", 3),
    ("fetal", "fetal1000k,fetal2000k", "polariseq-ram", 3),
]
CONTROL: list[tuple[str, str, str, int]] = [
    ("control", "brain400k,fetal2000k", "polariseq-ram", 3),
    ("control", "brain400k,brain1306k,fetal4062k", "polariseq-spill", 3),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--stage", choices=sorted({s[0] for s in STAGES}), help="run only this stage")
    ap.add_argument("--control", action="store_true", help="the published workflow, a few points")
    ap.add_argument("--umap", action="store_true", help="with UMAP (Fig. 3 of revision 4)")
    a = ap.parse_args(argv)
    stages = [s for s in STAGES if a.stage in (None, s[0])]
    if a.control:
        import os
        os.environ["POLARISEQ_WORKFLOW"] = "scanpy"
        fc.RESULTS = REPO / "benchmarks/results/telemetry_r4_scanpy"
        stages = CONTROL
    if a.umap:
        xp.CAMPAIGN = [x for x in xp.CAMPAIGN if x != "--no-umap"]
        fc.RESULTS = Path(str(fc.RESULTS).replace("telemetry_r4", "telemetry_r4_umap"))
    if a.list:
        for name, ds, arms, reps in STAGES + CONTROL:
            print(f"{name:<7} {reps} x [{arms}] on {ds}")
        return 0
    xp.LOGS.mkdir(parents=True, exist_ok=True)
    log = xp.Log(xp.LOGS / f"r4_campaign_{time.strftime('%Y%m%dT%H%M%S')}.log")
    fc.preflight(log, a.allow_dirty, a.allow_battery)
    xp.use_repo_scratch()
    xp.keep_awake()
    t0 = time.time()
    for name, ds, arms, reps in stages:
        log.line(f"\n===== {name}: {reps} x [{arms}] on {ds}")
        fc.telemetry(ds, arms, reps, log)
    log.line(f"workflow: {__import__('os').environ.get('POLARISEQ_WORKFLOW', 'polariseq (default)')}; "
             f"UMAP: {'yes' if a.umap else 'no'}; results -> {fc.RESULTS.relative_to(REPO)}")
    log.line(f"\nfinished in {(time.time() - t0) / 3600:.1f} h; records in {fc.RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
