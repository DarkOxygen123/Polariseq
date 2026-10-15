#!/usr/bin/env python
"""The timed laptop runs at 1.3M and 4.06M cells again, with Polariseq 0.2.0.

    .venv/bin/python benchmarks/relicence/laptop_v020.py [--allow-dirty]   # resumable
    .venv-011/bin/python benchmarks/relicence/laptop_v020.py --results telemetry_v011_umap_control
    ... --reps-upto 2       # at most this many repeats per stage (to alternate two builds)

The same harness, settings and repeats as the r4 laptop campaign with UMAP
(`benchmarks/r4_campaign.py --umap`: 4 threads, 6 GB budget, warm cache, memory ceiling 1.5 x the
laptop's memory) at the two largest points it ran Polariseq on: brain 1.3M (in memory, 1 repeat, which
the planner declines on this laptop as before, and out of core, 3 repeats) and fetal 4.06M (out of core,
3 repeats). Records go to benchmarks/results/telemetry_v020_umap, beside the r4 ones they replace.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import r4_campaign  # noqa: E402,F401  (sets the r4 campaign's harness arguments and memory ceiling)
import crossplat_run as xp  # noqa: E402
import final_campaign as fc  # noqa: E402

REPO = BENCH.parent
STAGES = [
    ("brain", "brain1306k", "polariseq-ram", 1),
    ("brain", "brain1306k", "polariseq-spill", 3),
    ("fetal", "fetal4062k", "polariseq-spill", 3),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--results", default="telemetry_v020_umap",
                    help="records folder under benchmarks/results (the build is the Python running this)")
    ap.add_argument("--reps-upto", type=int, default=None,
                    help="run each stage only up to this many repeats; with --resume semantics, calling it with "
                         "1, 2, 3 adds one repeat per call, so two builds can alternate")
    a = ap.parse_args(argv)
    xp.CAMPAIGN = [x for x in xp.CAMPAIGN if x != "--no-umap"]   # with UMAP, as r4_campaign --umap
    fc.RESULTS = REPO / "benchmarks/results" / a.results
    xp.LOGS.mkdir(parents=True, exist_ok=True)
    log = xp.Log(xp.LOGS / f"laptop_v020_{time.strftime('%Y%m%dT%H%M%S')}.log")
    fc.preflight(log, a.allow_dirty, a.allow_battery)
    xp.use_repo_scratch()
    xp.keep_awake()
    log.line(f"harness arguments: {' '.join(xp.CAMPAIGN)}")
    t0 = time.time()
    import polariseq
    log.line(f"polariseq {polariseq.__version__} from {polariseq.__file__}")
    for name, ds, arms, reps in STAGES:
        if a.reps_upto is not None:
            reps = min(reps, a.reps_upto)
        log.line(f"\n===== {name}: {reps} x [{arms}] on {ds}")
        fc.telemetry(ds, arms, reps, log)
    log.line(f"\nfinished in {(time.time() - t0) / 3600:.2f} h; records in {fc.RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
