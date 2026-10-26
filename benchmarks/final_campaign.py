"""The macOS campaign of record, at one commit, in one command.

    .venv/bin/python benchmarks/final_campaign.py            # everything, resumable
    .venv/bin/python benchmarks/final_campaign.py --list     # the plan, run nothing

The Polariseq laptop records come from here, after the 2026-09-20
fix to the neighbour-graph weights (clusters and UMAP changed; so, a little,
did the timings of the steps that read the graph). Stages run in order of
importance, so an interruption costs the least important
work; run the command again and completed runs are skipped.

1. public   — the four public 10x sets, every Polariseq and scanpy arm, 5
              repeats: output agreement.
2. ladder   — the 10x brain ladder 50K–1.3M, both Polariseq modes, 3 repeats.
3. fetal    — the human fetal atlas: out-of-core at 1M, 2M and 4.06M cells,
              in-memory at 1M and 2M, 3 repeats.
4. scanpy   — scanpy-tuned where the macOS re-run of 2026-09-19 did not
              reach (300K, 600K; scanpy is unchanged since, so its 50K–800K
              records stand), then its limit: 1M and 1.3M brain, and the
              fetal sizes, each attempted once and stopped at the first
              failure.
5. biology  — the atlas analyses: fetal 4.06M against the published
              annotation, brain 1.3M and 800K against scanpy, PBMC 3k.

Same protocol as every laptop campaign: 4 threads, 6 GB budget, no
UMAP in the timed runs, warm page cache, rotated arm order, calibrated load
gate. Run it after a restart, on the charger, with nothing else open.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import crossplat_run as xp  # noqa: E402
from telemetry.record import load_records  # noqa: E402

RESULTS = REPO / "benchmarks/results/telemetry_final"
PY = sys.executable

PUBLIC = "pbmc3k,pbmc10k,neuron10k,pbmc20k"
LADDER = "brain50k,brain100k,brain200k,brain300k,brain400k,brain600k,brain800k,brain1000k,brain1306k"
FETAL = "fetal1000k,fetal2000k,fetal4062k"
POLARISEQ = "polariseq-ram,polariseq-spill"

#: (stage, datasets, arms, repeats). A (label, arm) that fails is not
#: retried at that label (run_telemetry's --resume rule).
STAGES: list[tuple[str, str, str, int]] = [
    ("public", PUBLIC,
     "polariseq-ram,polariseq-spill,scanpy-default,scanpy-tuned,scanpy-tuned-seed1", 5),
    ("ladder", LADDER, POLARISEQ, 3),
    # In-memory at 4.06M cells would ask for ~19 GB on an 8.59 GB machine:
    # macOS would absorb it by growing swap into the free disk this campaign
    # needs, so that point is left to the planner's refusal rather than
    # measured. Out-of-core runs every size.
    ("fetal", FETAL, "polariseq-spill", 3),
    ("fetal", "fetal1000k,fetal2000k", "polariseq-ram", 3),
    ("scanpy", "brain300k,brain600k", "scanpy-tuned", 3),
]
#: scanpy's limit: each chain is walked upward and stops at the first size
#: scanpy does not complete. One repeat per size; three where it completes.
SCANPY_LIMIT = [["brain1000k", "brain1306k"], ["fetal1000k", "fetal2000k", "fetal4062k"]]

BIOLOGY = [
    [PY, "benchmarks/fetal_biology.py", "polariseq", "--size", "4062k"],
    [PY, "benchmarks/fetal_biology.py", "compare", "--size", "4062k"],
    [PY, "benchmarks/atlas_analysis.py", "polariseq", "--size", "1306k"],
    [PY, "benchmarks/atlas_analysis.py", "polariseq", "--size", "800k"],
    [PY, "benchmarks/atlas_analysis.py", "compare", "--size", "800k"],
    [PY, "benchmarks/pbmc3k_tutorial.py"],
]
BIOLOGY_DONE = {  # a biology step is skipped when its output exists and is newer than HEAD
    0: "benchmarks/results/fetal/4062k/summary.json",
    1: "benchmarks/results/fetal/4062k/comparison.json",
    2: "benchmarks/results/atlas/polariseq_1306k/summary.json",
    3: "benchmarks/results/atlas/polariseq_800k/summary.json",
    4: "benchmarks/results/atlas/comparison_800k.json",
    5: "benchmarks/results/pbmc3k/comparison.json",
}


def telemetry(datasets: str, arms: str, reps: int, log: xp.Log) -> None:
    rc = xp.stream([PY, str(HERE / "run_telemetry.py"), "--datasets", datasets,
                    "--arms", arms, *xp.CAMPAIGN, "--timeout", str(xp.TIMEOUT_S),
                    "--reps", str(reps), "--results", str(RESULTS), "--resume"], log)
    if rc != 0:
        xp.die(f"run_telemetry.py exited {rc} on {datasets}; see {log.path}")


def status(label: str, arm: str) -> str | None:
    """'ok', a failure status, or None if never run at this label."""
    recs = [r for r in load_records(RESULTS)
            if r["dataset"]["label"] == label and r["arm"] == arm]
    if not recs:
        return None
    return "ok" if any(r["status"] == "ok" for r in recs) else recs[-1]["status"]


def scanpy_limit(log: xp.Log) -> None:
    for chain in SCANPY_LIMIT:
        for label in chain:
            st = status(label, "scanpy-tuned")
            if st is None:
                log.line(f"\n[scanpy limit] attempting {label}")
                telemetry(label, "scanpy-tuned", 1, log)
                st = status(label, "scanpy-tuned")
            if st != "ok":
                log.line(f"[scanpy limit] scanpy stops at {label} ({st}); "
                         f"skipping the larger sizes in this chain")
                break
            telemetry(label, "scanpy-tuned", 3, log)


def head_time() -> float:
    out = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=REPO,
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def biology(log: xp.Log) -> None:
    since = head_time()
    for i, cmd in enumerate(BIOLOGY):
        done = REPO / BIOLOGY_DONE[i]
        if done.exists() and done.stat().st_mtime > since:
            log.line(f"[biology] {' '.join(cmd[1:])}: done, skipping")
            continue
        log.line(f"\n[biology] {' '.join(cmd[1:])}")
        rc = xp.stream(cmd, log)
        if rc != 0:
            xp.die(f"{' '.join(cmd[1:])} exited {rc}; see {log.path}")


def preflight(log: xp.Log, allow_dirty: bool, allow_battery: bool) -> None:
    import psutil

    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    log.line(f"commit {sha}{' (DIRTY)' if dirty else ''}; results -> {RESULTS.relative_to(REPO)}")
    if dirty and not allow_dirty:
        xp.die("the working tree has uncommitted changes, so records would carry a "
               f"'-dirty' commit:\n{dirty}\nCommit or stash them (or pass --allow-dirty).")
    batt = psutil.sensors_battery()
    if batt is not None and not batt.power_plugged and not allow_battery:
        xp.die("on battery: plug in the charger (battery power lowers the clocks).")
    swap = psutil.swap_memory().used
    log.line(f"swap in use: {swap / 1e9:.1f} GB"
             + ("  (restart first for a clean start: swap only clears on reboot)" if swap > 1e9 else ""))
    free = shutil.disk_usage(REPO).free
    log.line(f"disk free: {free / 1e9:.0f} GB")
    # Since the compact store (2026-10-03) the out-of-core runs' store peaks at 4.3 GB for the
    # 1.3M-cell brain and 3.8 GB for the 4.06M-cell fetal atlas (cluster records, block_store_gb);
    # the plain layout of earlier campaigns spilled ~22 GB, hence the old 30 GB floor.
    if free < 12e9:
        xp.die("less than 12 GB free: the largest out-of-core store needs about 4-5 GB, plus "
               "PCA's copy of the selected genes and room to spare.")
    missing = [n for n in (PUBLIC + "," + LADDER + "," + FETAL).split(",")
               if not (REPO / __import__("run_telemetry").DATASETS[n]).exists()]
    if missing:
        xp.die(f"missing inputs: {missing} (fetal: benchmarks/fetal_prepare.py all)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--only", choices=["public", "ladder", "fetal", "scanpy", "biology"],
                    action="append", help="run just these stages")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for name, ds, arms, reps in STAGES:
            print(f"{name:<8} {reps} x [{arms}] on {ds}")
        print(f"{'scanpy':<8} limit chains (1 attempt, then 3 reps if ok): {SCANPY_LIMIT}")
        for cmd in BIOLOGY:
            print(f"{'biology':<8} {' '.join(cmd[1:])}")
        return 0

    xp.LOGS.mkdir(parents=True, exist_ok=True)
    log = xp.Log(xp.LOGS / f"final_campaign_{time.strftime('%Y%m%dT%H%M%S')}.log")
    preflight(log, a.allow_dirty, a.allow_battery)
    xp.use_repo_scratch()
    xp.keep_awake()
    t0 = time.time()
    want = set(a.only or ["public", "ladder", "fetal", "scanpy", "biology"])
    for name, ds, arms, reps in STAGES:
        if name in want:
            log.line(f"\n===== {name}: {reps} x [{arms}] on {ds}")
            telemetry(ds, arms, reps, log)
    if "scanpy" in want:
        log.line("\n===== scanpy limit")
        scanpy_limit(log)
    if "biology" in want:
        log.line("\n===== biology")
        biology(log)
    log.line(f"\nfinished in {(time.time() - t0) / 3600:.1f} h; records in "
             f"{RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
