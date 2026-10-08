"""The macOS laptop campaign: every tool, one harness, one time limit.

    .venv/bin/python benchmarks/r3_campaign.py            # everything, resumable
    .venv/bin/python benchmarks/r3_campaign.py --list     # the plan, run nothing
    .venv/bin/python benchmarks/r3_campaign.py --umap     # the other tools with UMAP

What changed from ``final_campaign.py`` (whose records it supersedes for
speed and memory):

* Memory is the per-process peak footprint (``telemetry/footprint.py``),
  so every tool is re-run, not just Polariseq.
* Every tool runs under the same 60-minute limit and the same harness. The
  earlier Seurat records predate the harness's memory recording and used a
  30-minute limit.
* Two more tools: BPCells' own pipeline (``bpcells-native``) and Scarf.
* The 4.06M-cell file is attempted with scanpy after widening its indices
  (``scanpy-tuned-int64``), replacing the "file format" failure with an
  outcome of the analysis.

Stages, in order of importance:

1. public   — four public 10x sets, every tool (5 repeats for Polariseq and
              scanpy, 3 for the rest).
2. ladder   — Polariseq, both modes, brain 50K-1.3M, 3 repeats.
3. fetal    — Polariseq out-of-core at 1M, 2M, 4.06M; in-memory at 1M, 2M.
4. limits   — each competitor walked up the brain ladder (and, for the
              disk-backed ones and scanpy, the fetal sizes): one attempt per
              size, stopping at the first that does not complete; 3 repeats
              up to 400K cells, 1 above.
5. int64    — scanpy on the full fetal atlas with widened indices, once, last
              (it may push macOS into swap).

With ``--umap``: the same stages with UMAP as each tool runs it (Scanpy's sc.tl.umap,
Seurat's RunUMAP, Scarf's run_umap, uwot for BPCells as its tutorial does), into
benchmarks/results/telemetry_r4_umap. Polariseq's arms are left out here: they run under its new
default from ``r4_campaign.py --umap`` into the same folder, so every tool, UMAP included, is
read from one place. The records without UMAP are not touched.
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
fc.RESULTS = REPO / "benchmarks/results/telemetry_r3"
#: A run that needs more than 1.5x the laptop's memory is stopped and recorded
#: as exceeding it, rather than left to finish by swapping (decided
#: 2026-09-27, after Seurat ran 300K cells at a 46 GB footprint on 8.6 GB).
MEMORY_LIMIT_GB = 1.5 * 8.59
xp.CAMPAIGN = [*xp.CAMPAIGN, "--memory-limit-gb", f"{MEMORY_LIMIT_GB:.2f}"]

PUBLIC = fc.PUBLIC
LADDER = fc.LADDER.split(",")
FETAL = fc.FETAL.split(",")
POLARISEQ = fc.POLARISEQ

STAGES: list[tuple[str, str, str, int]] = [
    ("public", PUBLIC, "polariseq-ram,polariseq-spill,scanpy-tuned,scanpy-tuned-seed1", 5),
    ("public", PUBLIC, "seurat-tuned,seurat-bpcells,bpcells-native,scarf", 3),
    ("ladder", ",".join(LADDER), POLARISEQ, 3),
    ("fetal", ",".join(FETAL), "polariseq-spill", 3),
    ("fetal", "fetal1000k,fetal2000k", "polariseq-ram", 3),
]

#: (arm, chain). Walked upward; stops at the first size that does not complete.
LIMITS: list[tuple[str, list[str]]] = [
    ("scanpy-tuned", LADDER[:-1]),          # 1.3M exceeds int32 indices: see int64
    ("scanpy-tuned", FETAL[:-1]),
    ("bpcells-native", LADDER),
    ("bpcells-native", FETAL),
    ("scarf", LADDER),
    ("scarf", FETAL),
    ("seurat-tuned", LADDER),
    ("seurat-bpcells", LADDER),
]
INT64 = ("scanpy-tuned-int64", "fetal4062k")


def n_cells(label: str) -> int:
    return int(label.rstrip("k").lstrip("abcdefghijklmnopqrstuvwxyz")) * 1000


def walk(arm: str, chain: list[str], log: xp.Log) -> None:
    for label in chain:
        st = fc.status(label, arm)
        if st is None:
            log.line(f"\n[limit] {arm}: attempting {label}")
            fc.telemetry(label, arm, 1, log)
            st = fc.status(label, arm)
        if st != "ok":
            log.line(f"[limit] {arm} stops at {label} ({st}); skipping larger sizes")
            return
        if n_cells(label) <= 400_000:
            fc.telemetry(label, arm, 3, log)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", choices=["public", "ladder", "fetal", "limits", "int64"],
                    action="append")
    ap.add_argument("--arm", action="append",
                    help="with --only limits/int64: walk only these arms")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--umap", action="store_true",
                    help="revision 4: with UMAP, other tools only, into telemetry_r4_umap")
    ap.add_argument("--gate-pressured-fraction", type=float, default=None,
                    help="passed to run_telemetry (see its help)")
    a = ap.parse_args(argv)
    stages = STAGES
    if a.umap:
        xp.CAMPAIGN = [x for x in xp.CAMPAIGN if x != "--no-umap"]
        fc.RESULTS = REPO / "benchmarks/results/telemetry_r4_umap"
        stages = []
        for name, ds, arms, reps in STAGES:
            others = ",".join(x for x in arms.split(",") if not x.startswith("polariseq"))
            if others:
                stages.append((name, ds, others, reps))
    if a.list:
        for name, ds, arms, reps in stages:
            print(f"{name:<7} {reps} x [{arms}] on {ds}")
        for arm, chain in LIMITS:
            print(f"{'limits':<7} {arm}: {' -> '.join(chain)}")
        print(f"{'int64':<7} {INT64[0]} on {INT64[1]}, once")
        return 0

    if a.gate_pressured_fraction is not None:
        xp.CAMPAIGN = [*xp.CAMPAIGN, "--gate-pressured-fraction", str(a.gate_pressured_fraction)]
    xp.LOGS.mkdir(parents=True, exist_ok=True)
    log = xp.Log(xp.LOGS / f"r3_campaign_{time.strftime('%Y%m%dT%H%M%S')}.log")
    fc.preflight(log, a.allow_dirty, a.allow_battery)
    xp.use_repo_scratch()
    xp.keep_awake()
    t0 = time.time()
    want = set(a.only or ["public", "ladder", "fetal", "limits", "int64"])
    log.line(f"UMAP: {'yes' if a.umap else 'no'}; results -> {fc.RESULTS.relative_to(REPO)}")
    for name, ds, arms, reps in stages:
        if name in want:
            log.line(f"\n===== {name}: {reps} x [{arms}] on {ds}")
            fc.telemetry(ds, arms, reps, log)
    if "limits" in want:
        for arm, chain in LIMITS:
            if a.arm and arm not in a.arm:
                continue
            log.line(f"\n===== limits: {arm}")
            walk(arm, chain, log)
    if "int64" in want and fc.status(INT64[1], INT64[0]) is None:
        log.line(f"\n===== {INT64[0]} on {INT64[1]}")
        fc.telemetry(INT64[1], INT64[0], 1, log)
    log.line(f"\nfinished in {(time.time() - t0) / 3600:.1f} h; records in "
             f"{fc.RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
