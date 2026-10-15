#!/usr/bin/env python
"""The laptop series on Windows with Polariseq 0.2.0: mouse brain 50,000 to 1.3 million cells and the fetal atlas
1, 2 and 4.06 million cells, in memory and out of core (the Windows counterpart of the macOS laptop series).

    .venv\\Scripts\\python.exe benchmarks\\relicence\\windows_series_v020.py           # resumable
    .venv\\Scripts\\python.exe benchmarks\\relicence\\windows_series_v020.py --publish-only

Run through benchmarks\\windows\\run_series_v020.ps1, which rebuilds the library first.

1. The data: the brain subsets are drawn from the 10x Genomics parent file with make_subsets.py (seed 0), the
   fetal subsets from fetal4062k.h5ad with fetal_prepare.py (nested, seed 0), and each is checked against the
   macOS build (cells, genes and stored values), so both machines analyse the same cells.
2. The runs: the stages of the macOS series (benchmarks/r4_campaign.py, without the four public datasets) with
   the same harness and settings (run_telemetry.py: 4 threads, 6 GB budget, quality control through
   clustering without UMAP, warm cache, rotating order, calibrated quiet gate, three repeats). A run is stopped
   once it holds more than the installed memory (the Mac's series allows 1.5 times its 8 GB). Windows records
   the memory the run makes the machine hold (resident set, the growth of the compression store and the
   pagefile in use) beside the resident set. Records: benchmarks/results/telemetry_win_v020_series.
3. Pushed to the branch results/windows-v020-series after each stage, as
   benchmarks/platform_runs/windows-v020-series/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

import crossplat_run as xp  # noqa: E402

BRAIN = [50_000, 100_000, 200_000, 300_000, 400_000, 600_000, 800_000, 1_000_000]
#: cells -> stored values of the seed-0 subsets on macOS (benchmarks/results/validation_data/subsets_manifest.json)
BRAIN_NNZ = {50_000: 100_488_310, 100_000: 201_050_280, 200_000: 401_853_182, 300_000: 602_912_005,
             400_000: 803_920_843, 600_000: 1_205_611_786, 800_000: 1_607_496_131, 1_000_000: 2_009_685_138}
BRAIN_BYTES = {50_000: 808_515_462, 100_000: 1_616_390_326, 200_000: 3_228_910_678, 300_000: 4_845_228_110,
               400_000: 6_460_139_118, 600_000: 9_687_309_470, 800_000: 12_916_093_510,
               1_000_000: 16_147_289_142}
FULL_FILES = {   # fetched by run_v020.ps1 for the earlier Windows run; label: (path, bytes, cells, genes, stored values)
    "brain1306k": (REPO / "benchmarks/results/validation_data/brain1306k.h5ad", 21_094_963_453,
                   1_306_127, 27_998, 2_624_828_308),
    "fetal4062k": (REPO / "benchmarks/results/fetal_data/fetal4062k.h5ad", 6_724_796_853,
                   4_062_980, 63_561, 2_631_157_163),
}
#: the nested fetal subsets on macOS (benchmarks/results/fetal_data/fetal_manifest.json)
FETAL = {"fetal1000k": (1_000_000, 647_386_786), "fetal2000k": (2_000_000, 1_295_253_205)}
LADDER = ",".join([*(xp.label_of(n) for n in BRAIN), "brain1306k"])
STAGES = [   # benchmarks/r4_campaign.py's stages without the public datasets
    ("ladder", LADDER, "polariseq-ram,polariseq-spill", 3),
    ("fetal", "fetal1000k,fetal2000k,fetal4062k", "polariseq-spill", 3),
    ("fetal", "fetal1000k,fetal2000k", "polariseq-ram", 3),
]
RESULTS = REPO / "benchmarks/results/telemetry_win_v020_series"
LABEL = "windows-v020-series"


def check(log: xp.Log) -> None:
    import polariseq
    from polariseq import planning
    log.line(f"polariseq {polariseq.__version__} from {polariseq.__file__}")
    if not polariseq.__version__.startswith("0.2.0") or not hasattr(planning, "IN_MEMORY_VIEW_CHUNKS_MAX"):
        xp.die(f"this is polariseq {polariseq.__version__} without the planner of 2026-10-15: run git pull and "
               "run_series_v020.ps1 again")
    for label, (path, size, *_ ) in FULL_FILES.items():
        if not path.exists() or path.stat().st_size != size:
            xp.die(f"{path} is missing or not {size:,} bytes; benchmarks\\windows\\run_v020.ps1 fetches it")


def data(log: xp.Log) -> None:
    xp.EXPECTED_NNZ.update(BRAIN_NNZ)
    xp.SUBSET_BYTES.update(BRAIN_BYTES)
    missing = [n for n in BRAIN if xp.subset_nnz(xp.DATA / f"{xp.label_of(n)}.h5ad") != (n, xp.N_GENES, BRAIN_NNZ[n])]
    fetal_dir = FULL_FILES["fetal4062k"][0].parent
    need = sum(BRAIN_BYTES[n] for n in missing) + 4.3e9 * bool(missing) + sum(
        int(1.7e9 * c / 1e6) for k, (c, _) in FETAL.items() if not (fetal_dir / f"{k}.h5ad").exists())
    free = xp._free(xp.DATA)
    log.line(f"disk:    {free / 1e9:.0f} GB free; the subsets still to build take {need / 1e9:.0f} GB, "
             "and a run's block store up to 5 GB")
    if free < need + 20e9:
        xp.die(f"not enough disk: {free / 1e9:.0f} GB free, {(need + 20e9) / 1e9:.0f} GB needed")
    parent = xp.fetch_parent(log) if missing else None
    if missing:
        xp.ensure_subsets(parent, BRAIN, log)
    for n in BRAIN:
        log.line(f"subset:  {xp.label_of(n)} verified ({BRAIN_NNZ[n]:,} stored values, as on macOS)")
    for label, (_, _, cells, genes, nnz) in FULL_FILES.items():
        path = FULL_FILES[label][0]
        if xp.subset_nnz(path) != (cells, genes, nnz):
            xp.die(f"{label}: {xp.subset_nnz(path)}, macOS has {(cells, genes, nnz)}")
        log.line(f"file:    {label} verified ({nnz:,} stored values, as on macOS)")
    if any(xp.subset_nnz(fetal_dir / f"{k}.h5ad") != (c, 63_561, z) for k, (c, z) in FETAL.items()):
        log.line("subset:  building fetal1000k and fetal2000k (fetal_prepare.py subsets)")
        if xp.stream([sys.executable, str(BENCH / "fetal_prepare.py"), "subsets"], log) != 0:
            xp.die("fetal_prepare.py subsets failed")
    for k, (c, z) in FETAL.items():
        got = xp.subset_nnz(fetal_dir / f"{k}.h5ad")
        if got != (c, 63_561, z):
            xp.die(f"{k}: built {got}, macOS has {(c, 63_561, z)}; the subsets would not be the same cells")
        log.line(f"subset:  {k} verified ({z:,} stored values, as on macOS)")


def runs(log: xp.Log, push: bool) -> None:
    import psutil
    vm = psutil.virtual_memory()
    limit = vm.total / 1e9
    log.line(f"memory: {vm.total / 1e9:.1f} GB installed, {vm.available / 1e9:.1f} GB available now; "
             f"budget 6 GB; runs stopped above {limit:.1f} GB")
    if vm.available < 6.5e9:
        log.line("WARNING: less than 6.5 GB available; close programs for a fair measurement")
    campaign = [*xp.CAMPAIGN, "--memory-limit-gb", f"{limit:.2f}"]   # without UMAP, as the macOS series
    log.line(f"harness arguments: {' '.join(campaign)}")
    for name, datasets, arms, reps in STAGES:
        log.line(f"\n===== {name}: {reps} x [{arms}] on {datasets}")
        rc = xp.stream([sys.executable, str(BENCH / "run_telemetry.py"), "--datasets", datasets, "--arms", arms,
                        *campaign, "--timeout", str(xp.TIMEOUT_S), "--reps", str(reps),
                        "--results", str(RESULTS), "--resume"], log)
        if rc != 0:
            log.line(f"run_telemetry.py exited {rc} on {name}; continuing")
        publish(log, push)


def publish(log: xp.Log, push: bool) -> None:
    xp.RESULTS, xp.LABEL, xp.ARMS = RESULTS, LABEL, "polariseq-ram,polariseq-spill"
    stage = xp.report(log)
    (stage / "build.json").write_text(json.dumps({"commit": xp.git("rev-parse", "HEAD").strip()}), encoding="utf-8")
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
        data(log)
        runs(log, not a.no_publish)
    publish(log, not a.no_publish)
    log.line(f"\ndone. Log: {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
