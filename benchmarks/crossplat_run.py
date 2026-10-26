#!/usr/bin/env python3
"""The cross-platform campaign, start to finish, in one command.

    python benchmarks/crossplat_run.py                 # everything below
    python benchmarks/crossplat_run.py --smoke         # 1-minute wiring check
    python benchmarks/crossplat_run.py --publish-only  # push what is on record

Every step skips work whose result already
exists, so running the command again after an interruption resumes it.

1. Fetch the 10x 1.3M mouse-brain file (4.2 GB), checked by size and by the
   checksum the macOS campaign recorded.
2. Build the seed-0 subsets and check each one's nnz against the macOS
   build. Same parent, same seed and same nnz means the same cells, so every
   platform analyses identical data.
3. Run the ladder with the macOS campaign's exact parameters, one dataset
   size at a time, smallest first.
4. After each size: summary tables, figures, and a commit of records, logs
   and figures to the data-only branch ``results/<platform>`` on GitHub. A
   long run therefore delivers its small sizes before the large ones finish.

The branch holds only ``benchmarks/platform_runs/<platform>/``; the macOS
repository takes the directory from it. Nothing here touches the checked-out
branch or the working tree outside ``benchmarks/results/`` (ignored), so the
git SHA each record carries stays the SHA of the code that ran.
"""

from __future__ import annotations

import argparse
import codecs
import functools
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from telemetry.record import capture_env, load_records  # noqa: E402

DATA = REPO / "benchmarks" / "results" / "validation_data"
RESULTS = REPO / "benchmarks" / "results" / "telemetry"
LOGS = REPO / "benchmarks" / "results" / "logs"
STAGE = REPO / "benchmarks" / "results" / "publish"
SCRATCH = REPO / "benchmarks" / "results" / "tmp"

PARENT_NAME = "1M_neurons_filtered_gene_bc_matrices_h5.h5"
PARENT_URL = f"https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/{PARENT_NAME}"
PARENT_BYTES = 4_216_018_749
#: The download names itself. 10x serves the file through Cloudflare, which
#: refuses Python's default "Python-urllib/3.x" with 403 Forbidden (measured
#: 2026-09-19 on the Ryzen laptop, and reproduced from macOS); an explicit
#: agent is served normally, range requests included.
USER_AGENT = "polariseq-benchmark/0.1 (crossplat_run.py)"
#: SHA-256 of the parent's first 64 MiB, as make_subsets.py records it.
PARENT_SHA256_64MB = "156d18a98b188d501d9cb95c84d522f9a25f5e067d70964ea669f27563f8af15"
N_GENES = 27_998

#: nnz of each seed-0 subset as built on the macOS machine (its
#: subsets_manifest.json). make_subsets.py draws cells with a seeded
#: Generator, so the same parent and seed give the same cells; equal nnz is
#: the check that they did.
EXPECTED_NNZ = {
    50_000: 100_488_310,
    100_000: 201_050_280,
    200_000: 401_853_182,
    400_000: 803_920_843,
    800_000: 1_607_496_131,
}
#: Bytes on disk per subset, for the free-space check. The out-of-core arm's
#: scratch peaks at ~1.07x the file (13.8 GB at 800K on macOS).
SUBSET_BYTES = {
    50_000: 808_515_462,
    100_000: 1_616_390_326,
    200_000: 3_228_910_678,
    400_000: 6_460_139_118,
    800_000: 12_916_093_510,
}

#: The macOS campaign's parameters, verbatim. Changing any of them makes the
#: platforms incomparable, which is why they are not options here.
CAMPAIGN_ARMS = "polariseq-ram,polariseq-spill,scanpy-tuned"
CAMPAIGN = [
    "--threads", "4", "--ram-gb", "6", "--no-umap",
    "--cache", "warm", "--order", "rotate", "--gate", "calibrate",
]

#: Set from the command line in main(). A supplementary run (the Windows
#: pagefile one) runs a subset of the arms, under its own label, with a
#: longer per-run limit; the parameters above never change.
ARMS = CAMPAIGN_ARMS
LABEL: str | None = None
TIMEOUT_S = 3600.0


def label_of(n: int) -> str:
    return f"brain{n // 1000}k"


def platform_key() -> str:
    return {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(
        platform.system(), platform.system().lower())


def run_label() -> str:
    """What these results travel under: the platform, or a supplement's name."""
    return LABEL or platform_key()


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------- plumbing


class Log:
    """Console and log file at once, so an unattended run leaves a record."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: TextIO = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self._fh.write(text)
        self._fh.flush()

    def line(self, text: str = "") -> None:
        self.write(text + "\n")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    # UTF-8 everywhere: the harness prints em dashes and the records hold
    # arbitrary stderr, and Windows would otherwise encode a pipe with the
    # ANSI code page and fail on the first character outside it.
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stream(cmd: list[str], log: Log) -> int:
    """Run `cmd`, echoing its output live to the console and the log.

    Read in chunks rather than lines: the harness prints "[3/45] ... " and
    only ends the line when the run finishes, which can be half an hour.
    """
    log.line(f"\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=REPO, env=child_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
    assert proc.stdout is not None
    while chunk := proc.stdout.read1(65536):
        log.write(decode(chunk))
    return proc.wait()


def git(*args: str, env: dict[str, str] | None = None, stdin: str | None = None,
        check: bool = True) -> str:
    # Bytes in and out, never text mode: on Windows a text-mode pipe writes
    # "\n" as "\r\n", and `update-index --index-info` keeps the "\r" as part
    # of the path, so every published file would be named "…\r" (caught by
    # the test suite on the Ryzen laptop).
    r = subprocess.run(["git", *args], cwd=REPO, env=env, capture_output=True,
                       input=stdin.encode("utf-8") if stdin is not None else None)
    out, err = r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {err.strip() or out.strip()}")
    return out.strip()


def use_repo_scratch() -> Path:
    """Send every temporary file of the campaign into the repository folder.

    The out-of-core arm's scratch is the largest thing a run writes (13.8 GB
    at 800K) and goes wherever the temp directory is, which on Windows is the
    system drive whatever drive the repository is on. Kept here, the data,
    the scratch and the results share one disk and one free-space check.
    Children inherit the variables; `tempfile` is told to look again.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        os.environ[var] = str(SCRATCH)
    tempfile.tempdir = None
    return SCRATCH


@functools.cache
def disk_description(path: Path) -> str:
    """The physical disk under `path`, e.g. "SSD NVMe <model>" (Windows only).

    A hard disk would make the out-of-core timings a measurement of the disk,
    not comparable with the macOS machine's SSD, so the record says which.
    """
    if sys.platform != "win32":
        return "unknown"
    letter = _existing(path).resolve().drive.rstrip(":")
    script = (f"$n = (Get-Partition -DriveLetter {letter}).DiskNumber; "
              "Get-PhysicalDisk | Where-Object { $_.DeviceId -eq [string]$n } | "
              "ForEach-Object { '{0} {1} {2}' -f $_.MediaType, $_.BusType, $_.FriendlyName }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{letter}: {out.stdout.strip() or 'unknown'}"


def keep_awake() -> None:
    """Stop the machine sleeping while this process runs.

    A laptop that sleeps mid-run adds the nap to the wall time it records,
    and nothing downstream could tell. Only idle sleep is prevented; closing
    the lid still sleeps, so the lid stays open.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    elif sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


# --------------------------------------------------------------- preflight


def default_sizes(total_ram: int) -> list[int]:
    sizes = [50_000, 100_000, 200_000, 400_000]
    # 800K needs ~13 GB of input read plus the arm's working set; below
    # 12 GB of RAM it is a paging benchmark, which the macOS 8 GB machine
    # already reports.
    if total_ram >= 12e9:
        sizes.append(800_000)
    return sizes


#: Left free on the folder's drive at every point of the campaign.
DISK_MARGIN = 3e9


def _subset(n: int) -> Path:
    return DATA / f"{label_of(n)}.h5ad"


def _built_marker(n: int) -> Path:
    """Sidecar marking a subset this script built. Only marked subsets are
    ever deleted, so a machine that already holds the benchmark datasets (the
    macOS one) cannot lose them to this script."""
    return DATA / f".{label_of(n)}.built-by-crossplat"


def size_complete(n: int, reps: int) -> bool:
    """Is every (arm, repeat) of this size on record, or given up on?"""
    from run_telemetry import on_record
    from telemetry.record import Params

    done, failed = on_record(RESULTS, Params(with_umap=False, threads=4, ram_gb=6.0))
    label = label_of(n)
    return all((label, arm) in failed
               or all((label, arm, r) in done for r in range(1, reps + 1))
               for arm in ARMS.split(","))


def disk_plan(pending: list[int], free: float) -> tuple[str | None, float, float]:
    """(mode, bytes needed to keep every subset, bytes needed to roll).

    "keep": every subset stays on disk. "rolling": subsets are built one
    size at a time and each is deleted once its runs are on record, so the
    peak is the largest size alone: while it is built, its part files and
    the output coexist (2x the subset); while it runs, the subset and the
    out-of-core scratch do (~2.1x). None: the ladder does not fit either way.
    Subsets this script built for other sizes count as reclaimable.
    """
    if not pending:
        return "keep", 0.0, 0.0
    parent = DATA / PARENT_NAME
    parent_need = 0.0 if parent.exists() and parent.stat().st_size == PARENT_BYTES \
        else float(PARENT_BYTES)
    top = max(pending)
    s_top = SUBSET_BYTES[top]
    need_keep = (parent_need + sum(SUBSET_BYTES[n] for n in pending if not _subset(n).exists())
                 + 1.1 * s_top + DISK_MARGIN)
    reclaimable = sum(SUBSET_BYTES[n] for n in EXPECTED_NNZ
                      if n != top and _subset(n).exists() and _built_marker(n).exists())
    need_roll = parent_need + (1.1 if _subset(top).exists() else 2.1) * s_top + DISK_MARGIN
    if free >= need_keep:
        return "keep", need_keep, need_roll
    if free + reclaimable >= need_roll:
        return "rolling", need_keep, need_roll
    return None, need_keep, need_roll


def release_subset(n: int, log: Log) -> None:
    """Delete a subset this script built, once its runs are on record."""
    marker, path = _built_marker(n), _subset(n)
    if marker.exists() and path.exists():
        path.unlink()
        log.line(f"subset:  removed {path.name} ({SUBSET_BYTES[n] / 1e9:.1f} GB) to make room; "
                 f"it rebuilds in minutes if ever needed")
    marker.unlink(missing_ok=True)


def _free(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def preflight(args: argparse.Namespace, log: Log) -> tuple[list[int], bool]:
    import psutil

    vm = psutil.virtual_memory()
    env = capture_env()
    log.line(f"machine: {env['platform']} {env['release']} ({env['machine']}), "
             f"{env['cpu']}, {env['logical_cores']} logical cores, "
             f"{vm.total / 1e9:.1f} GB RAM ({vm.available / 1e9:.1f} GB available)")
    log.line(f"python:  {sys.executable} ({platform.python_version()})")
    log.line(f"code:    git {env['git_sha']}")
    if env["git_sha"].endswith("-dirty"):
        # Every record carries this SHA, so say what the "-dirty" is.
        changed = git("status", "--short", check=False).splitlines()
        log.line("         differs from the commit in:")
        for line in changed[:15]:
            log.line(f"           {line}")
    log.line(f"folder:  {REPO} (data, scratch and results all stay in it)")
    disk = disk_description(DATA)
    if disk != "unknown":
        log.line(f"disk:    {disk}")
    if "HDD" in disk.split():
        log.line("WARNING: this folder is on a hard disk. The out-of-core arm would then "
                 "measure the disk, not comparable with the macOS SSD; an SSD drive is better.")

    swap = env.get("swap_total_bytes", 0)
    log.line(f"memory:  RAM {vm.total / 1e9:.1f} GB + swap/pagefile {swap / 1e9:.1f} GB")
    if args.require_commit_gb and vm.total + swap < args.require_commit_gb * 1e9:
        die(f"RAM + pagefile is {(vm.total + swap) / 1e9:.1f} GB; this run needs at least "
            f"{args.require_commit_gb:.0f} GB. Enlarge the pagefile first (benchmarks/"
            f"crossplat.md, \"Supplement: a larger pagefile\") and restart Windows.")

    if env["power_plugged"] is False and not args.allow_battery:
        die("running on battery. Plug the charger in (battery power lowers the "
            "clocks and every wall time with it), or pass --allow-battery.")

    probe = subprocess.run(
        [sys.executable, "-c", "import polariseq, scanpy, h5py; print(polariseq.__version__)"],
        cwd=REPO, env=child_env(), capture_output=True, text=True)
    if probe.returncode != 0:
        die("polariseq/scanpy do not import in this environment. Run setup first "
            f"(benchmarks/windows/setup.ps1).\n{probe.stderr.strip()[-800:]}")
    log.line(f"polariseq {probe.stdout.strip()} imports")

    auto = not args.sizes
    sizes = sorted(args.sizes or default_sizes(vm.total))
    unknown = [n for n in sizes if n not in EXPECTED_NNZ]
    if unknown:
        die(f"no macOS reference for sizes {unknown}; known: {sorted(EXPECTED_NNZ)}")
    # Data and scratch share the folder's drive (use_repo_scratch), so one
    # free-space figure decides.
    free = _free(DATA)
    while True:
        pending = [n for n in sizes if not size_complete(n, args.reps)]
        mode, need_keep, need_roll = disk_plan(pending, free)
        if mode:
            break
        if auto and len(sizes) > 1:
            log.line(f"[disk] {free / 1e9:.0f} GB free is not enough for {label_of(sizes[-1])} "
                     f"(needs {need_roll / 1e9:.0f} GB even one size at a time); dropping it")
            sizes = sizes[:-1]
            continue
        die(f"not enough disk: {label_of(max(pending))} needs {need_roll / 1e9:.0f} GB free "
            f"even one size at a time, and {free / 1e9:.0f} GB is free. Free some space or "
            f"pass fewer --sizes.")
    log.line(f"ladder:  {', '.join(label_of(n) for n in sizes)} "
             f"({'auto' if auto else 'as given'}); {args.reps} reps; arms {ARMS}")
    if mode == "rolling":
        log.line(f"disk:    {free / 1e9:.0f} GB free; keeping every subset would need "
                 f"{need_keep / 1e9:.0f} GB, so each subset is built just before its size "
                 f"runs and deleted after (peak {need_roll / 1e9:.0f} GB)")
    else:
        log.line(f"disk:    {free / 1e9:.0f} GB free, {need_keep / 1e9:.0f} GB needed")
    return sizes, mode == "rolling"


def _existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


# -------------------------------------------------------------------- data


def _sha256_prefix(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while limit > 0 and (chunk := fh.read(min(1 << 20, limit))):
            h.update(chunk)
            limit -= len(chunk)
    return h.hexdigest()


def fetch_parent(log: Log) -> Path:
    """Download the 10x parent, resuming a partial download, then verify it."""
    dest = DATA / PARENT_NAME
    if dest.exists() and dest.stat().st_size == PARENT_BYTES:
        if _sha256_prefix(dest) != PARENT_SHA256_64MB:
            die(f"{dest} has the right size but the wrong checksum; delete it and re-run")
        log.line(f"parent:  {dest.name} present and verified")
        return dest

    DATA.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.line(f"parent:  downloading {PARENT_URL} (4.2 GB)")
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if have >= PARENT_BYTES:
            break
        req = urllib.request.Request(
            PARENT_URL, headers={"Range": f"bytes={have}-", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0  # the server ignored the range: start over
                with open(part, "ab" if have else "wb") as fh:
                    got, t0, shown = have, time.monotonic(), -1
                    while chunk := resp.read(8 << 20):
                        fh.write(chunk)
                        got += len(chunk)
                        pct = int(100 * got / PARENT_BYTES)
                        if pct != shown and pct % 5 == 0:
                            rate = (got - have) / max(time.monotonic() - t0, 1e-6) / 1e6
                            log.line(f"         {pct:3d}%  ({got / 1e9:.2f} GB, {rate:.0f} MB/s)")
                            shown = pct
        except urllib.error.HTTPError as exc:
            # A refusal does not go away by asking again; a timeout or a
            # rate limit might.
            if exc.code not in (408, 429) and exc.code < 500:
                die(f"the server refused the download: HTTP {exc.code} {exc.reason}. "
                    f"Download {PARENT_URL} another way and put it in {DATA}, then re-run.")
            log.line(f"         HTTP {exc.code}; retry {attempt}/8 in {5 * attempt} s")
            time.sleep(5 * attempt)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            log.line(f"         interrupted ({exc}); retry {attempt}/8 in {5 * attempt} s")
            time.sleep(5 * attempt)
    if not part.exists() or part.stat().st_size != PARENT_BYTES:
        die("download did not complete; re-run the command to resume it")
    part.replace(dest)
    if _sha256_prefix(dest) != PARENT_SHA256_64MB:
        die(f"downloaded {dest.name} does not match the macOS checksum")
    log.line("parent:  downloaded and verified")
    return dest


def subset_nnz(path: Path) -> tuple[int, int, int] | None:
    """(n_cells, n_genes, nnz) from the file's metadata, or None if unreadable."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            x = f["X"]
            shape = tuple(int(s) for s in x.attrs["shape"])
            return shape[0], shape[1], int(x["data"].shape[0])
    except (OSError, KeyError, ValueError):
        return None


def ensure_subsets(parent: Path, sizes: list[int], log: Log) -> None:
    """Build the missing subsets and verify all of them against macOS."""
    missing = []
    for n in sizes:
        path = DATA / f"{label_of(n)}.h5ad"
        if path.exists() and subset_nnz(path) == (n, N_GENES, EXPECTED_NNZ[n]):
            continue
        if path.exists():
            log.line(f"subset:  {path.name} does not match the macOS build; rebuilding")
            path.unlink()
        missing.append(n)
    if missing:
        log.line(f"subset:  building {', '.join(label_of(n) for n in missing)} (seed 0)")
        rc = stream([sys.executable, str(HERE / "make_subsets.py"), "--source", str(parent),
                     "--out-dir", str(DATA), "--seed", "0",
                     "--sizes", *map(str, missing)], log)
        if rc != 0:
            die(f"make_subsets.py failed (exit {rc}); see {log.path}")

    for n in sizes:
        got = subset_nnz(_subset(n))
        want = (n, N_GENES, EXPECTED_NNZ[n])
        if got != want:
            die(f"{label_of(n)}: built {got}, macOS has {want}. The subset draw differs "
                f"from the macOS one, so the platforms would not analyse the same cells.")
        if n in missing:
            _built_marker(n).touch()
        log.line(f"subset:  {label_of(n)} verified ({got[2]:,} nnz, same as macOS)")


# ---------------------------------------------------------------- campaign


def run_size(n: int, reps: int, log: Log) -> None:
    rc = stream([sys.executable, str(HERE / "run_telemetry.py"),
                 "--datasets", label_of(n), "--arms", ARMS, *CAMPAIGN,
                 "--timeout", str(TIMEOUT_S), "--reps", str(reps),
                 "--results", str(RESULTS), "--resume"], log)
    if rc != 0:
        die(f"run_telemetry.py exited {rc} on {label_of(n)}; see {log.path}")


def smoke(log: Log) -> None:
    """The whole harness on the committed 10K fixture, in about a minute.

    Proves this machine can build, launch and measure every arm before the
    real campaign commits hours to it, and shows the memory counters this
    platform reports.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rc = stream([sys.executable, str(HERE / "run_telemetry.py"),
                     "--datasets", "fixture", "--quick", "--reps", "1",
                     "--arms", CAMPAIGN_ARMS, "--threads", "4", "--ram-gb", "6",
                     "--gate", "off", "--results", tmp], log)
        records = load_records(tmp)
    bad = [f"{r['arm']}: {r['status']} {r.get('note', '')[:200]}"
           for r in records if r["status"] != "ok"]
    if rc != 0 or len(records) != len(CAMPAIGN_ARMS.split(",")) or bad:
        die(f"smoke test failed (exit {rc}, {len(records)} records): {bad}")
    log.line("\nsmoke test passed:")
    for r in records:
        pa = r["parent"]
        log.line(f"  {r['arm']:<16} {r['wall_s']:6.1f} s   peak RSS {pa['peak_rss'] / 1e9:.2f} GB"
                 f"   demand {pa['peak_demand_bytes'] / 1e9:.2f} GB   [{pa['pressure_kind']}]")
    out = subprocess.run([sys.executable, "-c", "import polariseq as ps; print(ps.resources())"],
                         cwd=REPO, env=child_env(), capture_output=True, text=True)
    log.line(f"  {out.stdout.strip()}")


# ------------------------------------------------------------------ report


def _admissible_rule():
    spec = importlib.util.spec_from_file_location(
        "_rpn", HERE / "render_paper_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._admissible


def _fmt(values: list[float], nd: int = 1) -> str:
    if not values:
        return "–"
    med = st.median(values)
    if len(values) == 1:
        return f"{med:.{nd}f} (n=1)"
    return f"{med:.{nd}f} [{min(values):.{nd}f}–{max(values):.{nd}f}] (n={len(values)})"


def datasets_on_record(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each dataset as the runs measured it, checked against the macOS build.

    From the records rather than the files: with rolling subsets the file of
    a finished size is gone, and the record is what the numbers came from.
    """
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        d = r["dataset"]
        n, nnz = int(d.get("n_cells") or 0), int(d.get("nnz") or 0)
        out.setdefault(str(d["label"]), {"cells": n, "genes": int(d.get("n_genes") or 0),
                                         "nnz": nnz, "matches_macos": EXPECTED_NNZ.get(n) == nnz})
    return dict(sorted(out.items(), key=lambda kv: kv[1]["cells"]))


def summary_markdown(records: list[dict[str, Any]], harness_report: str) -> str:
    admissible = _admissible_rule()
    verified = datasets_on_record(records)
    env = records[-1]["env"] if records else capture_env()
    ok: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    failed: dict[tuple[int, str], list[str]] = defaultdict(list)
    excluded: dict[tuple[int, str], int] = defaultdict(int)
    for r in records:
        key = (int(r["dataset"]["n_cells"]), r["arm"])
        if r["status"] != "ok":
            failed[key].append(f"{r['status']}: {r.get('note', '')[:160]}")
        elif admissible(r):
            ok[key].append(r)
        else:
            excluded[key] += 1

    shas = sorted({r["env"].get("git_sha", "?") for r in records})
    versions = env.get("versions", {})
    lines = [
        f"# Cross-platform campaign — {env['platform']}"
        + (f" ({run_label()})" if run_label() != platform_key() else ""),
        "",
        f"- **Machine:** {env['cpu']}, {env['logical_cores']} logical cores, "
        f"{env['total_ram_bytes'] / 1e9:.1f} GB RAM; {env['platform']} {env['release']} "
        f"({env['machine']}); host `{env['hostname']}`; on mains power: "
        f"{env.get('power_plugged')}",
        f"- **Code:** git {', '.join(shas)}; Python {versions.get('python', '?')}; "
        + ", ".join(f"{k} {versions[k]}" for k in
                    ("polariseq", "scanpy", "anndata", "numpy", "scipy", "numba") if k in versions),
        "- **Parameters (identical to the macOS campaign):** 4 threads, 6 GB budget, "
        "no UMAP, warm page cache, arm order rotated per repeat, calibrated load gate.",
        "- **Datasets:** " + ", ".join(
            f"{k} ({v['nnz'] or 0:,} nnz, "
            f"{'same as macOS' if v['matches_macos'] else 'DIFFERS from macOS'})"
            for k, v in verified.items()),
        "",
        "## Results",
        "",
        "Median [min–max] over admissible repeats (gate met). Demand is peak "
        "RSS plus the memory the OS moved aside (`pressure_kind` "
        f"`{records[-1]['parent'].get('pressure_kind', '?') if records else '?'}`).",
        "",
        "| cells | arm | wall time (s) | peak demand (GB) | peak RSS (GB) | spill peak (GB) |",
        "|---:|---|---|---|---|---|",
    ]
    for key in sorted(set(ok) | set(failed) | set(excluded)):
        n, arm = key
        rs = ok.get(key, [])
        if rs:
            spill = [(r.get("disk") or {}).get("spill_peak_bytes", 0) / 1e9 for r in rs]
            lines.append(
                f"| {n:,} | {arm} | {_fmt([r['wall_s'] for r in rs])} "
                f"| {_fmt([r['parent']['peak_demand_bytes'] / 1e9 for r in rs], 2)} "
                f"| {_fmt([r['parent']['peak_rss'] / 1e9 for r in rs], 2)} "
                f"| {_fmt(spill, 2)} |")
        for why in failed.get(key, []):
            lines.append(f"| {n:,} | {arm} | **did not complete** — {why} | | | |")
        if excluded.get(key):
            lines.append(f"| {n:,} | {arm} | {excluded[key]} run(s) excluded: gate not met | | | |")
    lines += ["", "## Harness report", "", harness_report.strip(), ""]
    return "\n".join(lines)


def report(log: Log) -> Path:
    """Tables, figures and the staged directory that `publish` commits."""
    stage = STAGE / run_label()
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "telemetry").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "logs").mkdir()

    records = load_records(RESULTS)
    out = subprocess.run([sys.executable, str(HERE / "run_telemetry.py"), "--report-only",
                          "--figures", "--results", str(RESULTS)],
                         cwd=REPO, env=child_env(), capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    harness_report = out.stdout if out.returncode == 0 else (
        f"(report failed, exit {out.returncode})\n{out.stderr[-2000:]}")
    fig = subprocess.run([sys.executable, str(HERE / "figures" / "fig_crossplat.py"),
                          "--records", str(RESULTS), "--out", str(stage / "figures")],
                         cwd=REPO, env=child_env(), capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if fig.returncode != 0:
        log.line(f"[fig_crossplat] failed (exit {fig.returncode}):\n{fig.stderr[-800:]}")
    else:
        log.line(fig.stdout.strip())

    (stage / "summary.md").write_text(
        summary_markdown(records, harness_report), encoding="utf-8")
    env = capture_env()
    if sys.platform == "win32":
        scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        env["power_scheme"] = scheme.stdout.strip()
    env["datasets"] = datasets_on_record(records)
    env["folder"] = str(REPO)
    env["git_status"] = git("status", "--short", check=False).splitlines()
    env["scratch_dir"] = tempfile.gettempdir()
    env["data_disk"] = disk_description(DATA)
    (stage / "machine.json").write_text(json.dumps(env, indent=1), encoding="utf-8")

    for rec in records:
        src = Path(rec["_path"])
        shutil.copy2(src, stage / "telemetry" / src.name)
        # The first repeat's cluster labels and HVG list, so the macOS side
        # can check whether this platform reached the same answer. The PCA
        # embedding and barcodes stay behind: tens of MB each.
        art = RESULTS / "artifacts" / rec["run_id"]
        if (art / "labels.npy").exists():
            import numpy as np

            dst = stage / "artifacts" / rec["run_id"]
            dst.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst / "labels.npz", labels=np.load(art / "labels.npy"))
            if (art / "hvg.txt").exists():
                shutil.copy2(art / "hvg.txt", dst / "hvg.txt")
    # run_telemetry.py --figures writes next to the records directory.
    for png in (RESULTS.parent / "figures").glob("telemetry_*.png"):
        shutil.copy2(png, stage / "figures" / png.name)
    for lg in LOGS.glob("*.log"):
        if lg.name.startswith((f"crossplat_{run_label()}_", "setup_", "pytest_")):
            shutil.copy2(lg, stage / "logs" / lg.name)
    log.line(f"report:  {len(records)} records staged in {stage}")
    return stage


# ----------------------------------------------------------------- publish


def publish(stage: Path, branch: str, remote: str, log: Log) -> bool:
    """Commit `stage` to `branch` on `remote` without touching the checkout.

    The branch is data-only: its tree is `benchmarks/platform_runs/<platform>/`
    and nothing else. The commit is built in a scratch index with the previous
    tip as parent, so every push is a fast-forward and the working tree, the
    index and the checked-out branch never change.
    """
    prefix = f"benchmarks/platform_runs/{run_label()}"
    remote_ref = f"refs/remotes/{remote}/{branch}"
    # A failed fetch (first publish: no such branch yet; or no network) is
    # not fatal. The last tip we know of is still the right parent, and if it
    # is stale the push is refused as a non-fast-forward rather than
    # overwriting anything.
    subprocess.run(["git", "fetch", "-q", remote, f"+refs/heads/{branch}:{remote_ref}"],
                   cwd=REPO, capture_output=True, text=True)
    base = git("rev-parse", "-q", "--verify", remote_ref, check=False)
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("read-tree", *([base] if base else ["--empty"]), env=env)
        # --no-filters: store the bytes as written. Line-ending conversion
        # would make a record's blob differ from the file it came from.
        blobs = git("hash-object", "-w", "--no-filters", "--stdin-paths", env=env,
                    stdin="\n".join(str(p) for p in files)).splitlines()
        entries = [f"100644 {sha}\t{prefix}/{p.relative_to(stage).as_posix()}"
                   for sha, p in zip(blobs, files, strict=True)]
        git("update-index", "-z", "--index-info", env=env, stdin="\0".join(entries) + "\0")
        tree = git("write-tree", env=env)
    if base and tree == git("rev-parse", f"{base}^{{tree}}"):
        log.line(f"publish: nothing new for {branch}")
        return True

    env = dict(os.environ)
    if not git("config", "user.email", check=False):
        env.update(GIT_AUTHOR_NAME="Polariseq bench", GIT_AUTHOR_EMAIL="dev@polariseq.local",
                   GIT_COMMITTER_NAME="Polariseq bench", GIT_COMMITTER_EMAIL="dev@polariseq.local")
    records = len(list((stage / "telemetry").glob("*.json")))
    msg = (f"{platform.system()} campaign on {platform.node()}: {records} records\n\n"
           f"Code: git {git('rev-parse', '--short', 'HEAD', check=False)}. "
           f"Generated by benchmarks/crossplat_run.py.")
    commit = git("commit-tree", tree, *(["-p", base] if base else []), "-m", msg, env=env)
    pushed = subprocess.run(["git", "push", "-q", remote, f"{commit}:refs/heads/{branch}"],
                            cwd=REPO, capture_output=True, text=True)
    if pushed.returncode != 0:
        log.line(f"publish: push to {remote}/{branch} FAILED — results are safe locally; "
                 f"re-run with --publish-only to retry.\n{pushed.stderr.strip()[-600:]}")
        return False
    git("update-ref", remote_ref, commit)
    log.line(f"publish: {records} records -> {remote}/{branch} ({commit[:10]})")
    return True


# -------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                   help="comma-separated cell counts (default: 50K-400K, plus 800K "
                        "with at least 12 GB of RAM and enough disk)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--smoke", action="store_true",
                   help="1-minute harness check on the 10K fixture, then exit")
    p.add_argument("--publish-only", action="store_true",
                   help="skip data and campaign; report and push what is on record")
    p.add_argument("--no-publish", action="store_true", help="never push")
    p.add_argument("--branch", help="default: results/<label>")
    p.add_argument("--arms", default=CAMPAIGN_ARMS,
                   help="a subset of the campaign's arms, for a supplementary run")
    p.add_argument("--label", help="name for the results directory and branch "
                   "(default: the platform, e.g. windows)")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    p.add_argument("--require-commit-gb", type=float, default=0.0,
                   help="refuse to start unless RAM + swap/pagefile is at least this")
    p.add_argument("--remote", default="origin")
    p.add_argument("--allow-battery", action="store_true")
    p.add_argument("--results", type=Path, default=RESULTS,
                   help="records directory (default: benchmarks/results/telemetry)")
    return p


def main(argv: list[str] | None = None) -> int:
    global RESULTS, ARMS, LABEL, TIMEOUT_S
    args = build_parser().parse_args(argv)
    RESULTS = args.results.resolve()
    ARMS, LABEL, TIMEOUT_S = args.arms, args.label, args.timeout
    unknown = set(ARMS.split(",")) - set(CAMPAIGN_ARMS.split(","))
    if unknown:
        die(f"not campaign arms: {sorted(unknown)}; choose from {CAMPAIGN_ARMS}")
    args.branch = args.branch or f"results/{run_label()}"
    use_repo_scratch()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = Log(LOGS / f"crossplat_{run_label()}_{stamp}.log")
    keep_awake()

    if args.smoke:
        smoke(log)
        return 0
    if args.publish_only:
        stage = report(log)
        return 0 if args.no_publish or publish(stage, args.branch, args.remote, log) else 1

    sizes, rolling = preflight(args, log)
    try:
        parent = None
        for n in sizes:
            log.line(f"\n=== {label_of(n)}: {args.reps} reps x {ARMS} ===")
            if size_complete(n, args.reps):
                log.line(f"{label_of(n)}: every run is on record")
            else:
                # The parent is fetched only when a subset has to be built.
                if parent is None and subset_nnz(_subset(n)) != (n, N_GENES, EXPECTED_NNZ[n]):
                    parent = fetch_parent(log)
                ensure_subsets(parent or DATA / PARENT_NAME, [n], log)
                run_size(n, args.reps, log)
                stage = report(log)
                if not args.no_publish:
                    publish(stage, args.branch, args.remote, log)
            if rolling:
                release_subset(n, log)
    except KeyboardInterrupt:
        log.line("\ninterrupted. Run the same command again to resume; "
                 "completed runs are kept and skipped.")
        return 130
    log.line(f"\ndone. Log: {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
