#!/usr/bin/env python
"""The Polariseq laptop series again, with Polariseq 0.2.0.

    .venv/bin/python benchmarks/relicence/laptop_series_v020.py [--allow-dirty]   # resumable
    .venv/bin/python benchmarks/relicence/laptop_series_v020.py --list

1. The stages of the r4 laptop campaign (benchmarks/r4_campaign.py), unchanged: the same harness,
   datasets, repeats (five on the four public datasets, three on every subset of the mouse brain and the
   fetal atlas), threads (4), budget (6 GB), memory measure and ceiling (1.5 x the laptop's memory),
   quality control through clustering without UMAP. Records: benchmarks/results/telemetry_v020 (the r4
   records in telemetry_r4 stay as they are; the other tools' records in telemetry_r3 stand).
2. The budget sweep at 800,000 mouse-brain cells: out of core at 1, 2, 3, 4, 6 and
   8 GB, one run each; the planner declines the budgets below its floor before reading any data.
   Records: benchmarks/results/v020_budget/<N>gb, as r4_budget.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import r4_campaign as r4  # noqa: E402  (sets the r4 campaign's harness arguments and memory ceiling)
import crossplat_run as xp  # noqa: E402
import final_campaign as fc  # noqa: E402

REPO = BENCH.parent
RESULTS = REPO / "benchmarks/results/telemetry_v020"
BUDGETS = REPO / "benchmarks/results/v020_budget"
BUDGET_GB = (1, 2, 3, 4, 6, 8)


def budget_sweep(log: xp.Log) -> None:
    base = list(xp.CAMPAIGN)
    i = base.index("--ram-gb")
    for gb in BUDGET_GB:
        args = base[:i + 1] + [str(gb)] + base[i + 2:]
        out = BUDGETS / f"{gb}gb"
        log.line(f"\n===== budget {gb} GB: 1 x [polariseq-spill] on brain800k")
        rc = xp.stream([fc.PY, str(BENCH / "run_telemetry.py"), "--datasets", "brain800k",
                        "--arms", "polariseq-spill", *args, "--timeout", str(xp.TIMEOUT_S),
                        "--reps", "1", "--results", str(out), "--resume"], log)
        # a budget below the floor is declined: a 'failed' record, which is the result
        log.line(f"budget {gb} GB: run_telemetry exited {rc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for name, ds, arms, reps in r4.STAGES:
            print(f"{name:<7} {reps} x [{arms}] on {ds}")
        print(f"budget  1 x [polariseq-spill] on brain800k at {', '.join(map(str, BUDGET_GB))} GB")
        return 0
    fc.RESULTS = RESULTS
    xp.LOGS.mkdir(parents=True, exist_ok=True)
    log = xp.Log(xp.LOGS / f"laptop_series_v020_{time.strftime('%Y%m%dT%H%M%S')}.log")
    fc.preflight(log, a.allow_dirty, a.allow_battery)
    xp.use_repo_scratch()
    xp.keep_awake()
    import polariseq
    log.line(f"polariseq {polariseq.__version__} from {polariseq.__file__}")
    log.line(f"harness arguments: {' '.join(xp.CAMPAIGN)}")
    t0 = time.time()
    for name, ds, arms, reps in r4.STAGES:
        log.line(f"\n===== {name}: {reps} x [{arms}] on {ds}")
        fc.telemetry(ds, arms, reps, log)
    budget_sweep(log)
    log.line(f"\nfinished in {(time.time() - t0) / 3600:.2f} h; records in "
             f"{RESULTS.relative_to(REPO)} and {BUDGETS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
