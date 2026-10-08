"""The parent: launches an arm, watches it from outside, records the result.

The parent sampler exists because the child sampler cannot be trusted alone.
Two independent reasons:

1. **The GIL.** A Python thread inside the child cannot run while a native
   extension holds the GIL. Polariseq's Rust kernels currently do, for their
   whole duration — measured: 1 sample where 36 were due, across a 0.72 s
   PCA. A blind sampler draws a flat line, and a flat line is exactly the
   result we would like to see, which is why it must not be the only
   evidence.
2. **Death.** A process killed by the OOM killer writes nothing. Only an
   observer outside it can report how far it got and how big it was when it
   died.

The parent reads RSS (readable across processes everywhere) plus system-wide
memory and swap counters. It cannot read USS: macOS denies ``task_for_pid``
to non-root, measured. So the division of labour is fixed and honest —
**USS from inside, RSS from outside, and the two RSS series cross-check each
other.** Where they disagree, the child was blind, and the analysis says so.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

from .arms import ARMS, dataset_facts
from .footprint import rusage
from .loadgate import Band, MachineState, wait_for_quiet
from .record import THREAD_ENV_VARS, Dataset, Params, RunRecord, capture_env

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
REPO_ROOT = BENCH_ROOT.parent

#: The parent samples more slowly than the child: it is measuring a coarser
#: thing (RSS from outside, plus system counters) and every sample costs a
#: cross-process syscall.
PARENT_INTERVAL_S = 0.05

#: Reusable scratch for the page-cache warm read. Module-level so warming a
#: hundred files does not allocate a hundred buffers.
_WARM_BUF = bytearray(8 * 1024 * 1024)


def _pressure_kind() -> str:
    """Which out-of-the-way memory counter this operating system has.

    The macOS compressor, the Linux swap-out counter and the Windows
    compression store plus pagefile answer the same question — how much
    memory did the machine have to move aside to keep this run alive — so the
    sampler reads whichever one exists and records its name next to the
    numbers.
    """
    if sys.platform == "darwin":
        return "compressor"
    if sys.platform.startswith("linux"):
        return "swap-out"
    if sys.platform == "win32":
        return "compression+pagefile"
    return "none"


#: PID of Windows' compression store, found once per process (0: none).
_STORE_PID: int | None = None


def _compression_store_pid() -> int:
    """The "Memory Compression" process: Windows keeps compressed pages in
    its working set, as macOS keeps them in the compressor. psutil reports
    it as "MemCompression" and reads its memory without administrator
    rights (it falls back to NtQuerySystemInformation)."""
    global _STORE_PID
    if _STORE_PID is None:
        _STORE_PID = 0
        for p in psutil.process_iter(["name"]):
            name = (p.info.get("name") or "").lower().replace(" ", "")
            if name in ("memcompression", "memorycompression"):
                _STORE_PID = p.pid
                break
    return _STORE_PID


class ParentSampler:
    """Watches a child process *and its descendants* from outside the GIL.

    The process tree matters, not just the process. An arm that shells out —
    the Seurat arm drives `Rscript` as a grandchild — does its real work in a
    process the wrapper does not own. Sampling only the direct child would
    report the near-idle Python wrapper's footprint and silently record a
    Seurat run as costing almost nothing.

    RSS is summed across the tree, which double-counts pages shared between
    processes. That is the right error to make here: every arm in this
    campaign does its work in one process, so the sum equals that process,
    and an arm that ever did fan out would be over-reported rather than
    flatteringly under-reported. `n_procs` records the tree size so a run
    where it exceeded one can be identified.
    """

    def __init__(self, pid: int, interval_s: float = PARENT_INTERVAL_S) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.origin = 0.0
        self.t: list[float] = []
        self.rss: list[int] = []
        self.n_procs: list[int] = []
        self.sys_available: list[int] = []
        self.swap_in: list[int] = []
        self.swap_out: list[int] = []
        self.compressed: list[int] = []
        #: The out-of-the-way memory term, named by platform: the macOS
        #: compressor, Linux swap-out, Windows compression store + pagefile.
        #: One concept in three dialects — memory the machine moved aside to
        #: keep the run alive, which a resident-set peak alone cannot see.
        self.pressure_kind = _pressure_kind()
        #: macOS only: the kernel's lifetime peak footprint per process in
        #: the tree (see footprint.py). Summed over processes at the end.
        self._fp_peak: dict[int, int] = {}
        #: Stop the run once the tree's footprint passes this (bytes; None
        #: for no limit), and say so: an analysis that only finishes because
        #: the system swaps many times the machine's memory measures the
        #: swap, not the tool.
        self.memory_limit: int | None = None
        self.exceeded = False
        self.pressure: list[int] = []
        self._last_pressure_read = -1e9
        self._pressure_last = 0
        #: Windows only, diagnostics: the tree's private commit per sample,
        #: and the kernel's own high-water marks (working set, private
        #: commit) as of the last sample. The kernel tracks those
        #: continuously, so they catch a spike between two 50 ms samples.
        self.private: list[int] = []
        self._os_peak_rss = 0
        self._os_peak_private = 0
        self._store_found: bool | None = None

    @staticmethod
    def _compressed_bytes() -> int:
        """Bytes held by the macOS memory compressor, system-wide.

        Peak RSS is not a measure of memory demand on macOS. When the machine
        runs short, pages are compressed in place rather than swapped, and a
        compressed page is no longer resident, so it leaves RSS. Measured
        2026-09-10: scanpy read an 800K matrix whose in-memory CSR form is
        12.86 GB, on a machine with 8.59 GB installed, and this sampler
        recorded a 5.54 GB peak with 37 MB of swap. The missing memory was
        compressed, and a table built from RSS alone would have reported that
        run as fitting comfortably in half the machine.

        Read from `vm_stat`, because psutil exposes no compressor counter.
        Returns 0 off macOS or on any failure, leaving the delta at zero
        rather than inventing one.
        """
        if sys.platform != "darwin":
            return 0
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0
        page, occupied = 4096, 0
        for line in out.splitlines():
            if "page size of" in line:
                for tok in line.split():
                    if tok.isdigit():
                        page = int(tok)
                        break
            elif line.startswith("Pages occupied by compressor"):
                try:
                    occupied = int(line.rsplit(":", 1)[1].strip().rstrip("."))
                except ValueError:
                    return 0
        return occupied * page

    @staticmethod
    def _swap_out_bytes() -> int:
        """Bytes the Linux kernel has swapped out, system-wide (cumulative).

        The Linux analogue of the macOS compressor: pages the kernel moved
        to swap leave the resident set, so a run that oversubscribed memory
        shows a peak RSS that fits and a swap-out counter that does not.
        psutil reads this from /proc/vmstat cheaply, so unlike `vm_stat` it
        needs no subprocess.
        """
        if sys.platform != "linux":
            return 0
        try:
            return int(psutil.swap_memory().sout)
        except (OSError, RuntimeError, ValueError):
            return 0

    def _windows_moved_aside_bytes(self) -> int:
        """Compression store plus pagefile in use, system-wide (Windows).

        Windows first compresses the pages it trims from a working set into
        the store, as macOS does into its compressor, and writes to the
        pagefile after that; the two together are what it moved aside.

        Not the process's private commit, which was the first version of
        this term: private commit also counts memory committed but never
        touched (DLL data sections, allocator reservations). On the Ryzen
        laptop's first smoke test that put 'demand' 0.1-0.3 GB above peak
        RSS on runs that fit comfortably, where the other platforms report
        demand equal to RSS. Commit charge is worse still: it grows with
        every allocation, resident or not.
        """
        total = 0
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            total += int(psutil.swap_memory().used)
        pid = _compression_store_pid()
        if pid:
            try:
                total += int(psutil.Process(pid).memory_info().rss)
                self._store_found = True
            except (psutil.Error, OSError):
                self._store_found = False
        else:
            self._store_found = False
        return total

    def _pressure_bytes(self) -> int:
        """Current system-wide pressure term for this platform, in bytes."""
        return {
            "compressor": self._compressed_bytes,
            "swap-out": self._swap_out_bytes,
            "compression+pagefile": self._windows_moved_aside_bytes,
        }.get(self.pressure_kind, lambda: 0)()

    def _loop(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        windows = self.pressure_kind == "compression+pagefile"
        while not self._stop.is_set():
            try:
                mi = proc.memory_info()
                rss = int(mi.rss)
                # Windows: `private` is the process's private commit, kept
                # as a diagnostic (see _windows_moved_aside_bytes for why it
                # is not the pressure term). Zero elsewhere.
                private = int(getattr(mi, "private", 0))
                os_rss = int(getattr(mi, "peak_wset", 0))
                os_private = int(getattr(mi, "peak_pagefile", 0))
                n_procs = 1
                self._note_footprint(self.pid)
                # Descendants: an arm that shells out does its work here.
                for ch in proc.children(recursive=True):
                    self._note_footprint(ch.pid)
                    try:
                        cmi = ch.memory_info()
                        rss += int(cmi.rss)
                        private += int(getattr(cmi, "private", 0))
                        os_rss += int(getattr(cmi, "peak_wset", 0))
                        os_private += int(getattr(cmi, "peak_pagefile", 0))
                        n_procs += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except OSError:
                break
            try:
                vm = psutil.virtual_memory()
                avail = int(vm.available)
                if windows:
                    # Windows leaves the swap-in/out counters at zero, and
                    # reading them costs a performance-counter query.
                    sin = sout = 0
                else:
                    sw = psutil.swap_memory()
                    sin, sout = int(sw.sin), int(sw.sout)
            except (OSError, RuntimeError):
                avail = sin = sout = 0
            self.t.append(time.perf_counter() - self.origin)
            self.rss.append(rss)
            self.n_procs.append(n_procs)
            self.sys_available.append(avail)
            self.swap_in.append(sin)
            self.swap_out.append(sout)
            if windows:
                self.private.append(private)
                self._os_peak_rss = max(self._os_peak_rss, os_rss)
                self._os_peak_private = max(self._os_peak_private, os_private)
            # Throttled to ~1 Hz. The macOS reader shells out to `vm_stat` at
            # about 2 ms a call and the Windows one queries a performance
            # counter and the store process, and the sampler ticks 20 times a
            # second, so reading every tick would perturb the thing being
            # measured. Pressure moves over seconds, so 1 Hz resolves it
            # fully, and the same clock keeps the series comparable.
            now = time.perf_counter()
            if now - self._last_pressure_read >= 1.0:
                self._last_pressure_read = now
                self._pressure_last = self._pressure_bytes()
            if (self.memory_limit and not self.exceeded
                    and sum(self._fp_peak.values()) > self.memory_limit):
                self.exceeded = True
                for p_ in [*proc.children(recursive=True), proc]:
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        p_.kill()
            self.pressure.append(self._pressure_last)
            self.compressed.append(
                self._pressure_last if self.pressure_kind == "compressor" else 0
            )
            self._stop.wait(self.interval_s)

    def _note_footprint(self, pid: int) -> None:
        r = rusage(pid)
        if r is not None:
            peak = r["lifetime_max_phys_footprint"]
            self._fp_peak[pid] = max(self._fp_peak.get(pid, 0), peak)

    def start(self) -> ParentSampler:
        self.origin = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> ParentSampler:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self

    def as_dict(self) -> dict[str, Any]:
        # Swap counters are cumulative and system-wide; the delta over the run
        # is what matters. Non-zero swap means the wall time measured the
        # machine's paging, not the library — a run like that is reported, not
        # averaged in silently.
        swapped = 0
        if len(self.swap_out) >= 2:
            swapped = max(0, self.swap_out[-1] - self.swap_out[0])
        # Pressure this run caused, over its own starting point. The gate
        # keeps the machine quiet, so the rise is attributable to the run.
        # Peak RSS plus this is the honest figure for memory demanded; RSS
        # alone is only what the machine chose to keep resident. Which
        # counter this is depends on the platform (`pressure_kind`); on this
        # macOS it is the compressor, and `compressed_delta_bytes` keeps
        # carrying that value for the existing records and figures.
        pressure_delta = 0
        if len(self.pressure) >= 2:
            pressure_delta = max(0, max(self.pressure) - self.pressure[0])
        compressed_delta = (
            pressure_delta if self.pressure_kind == "compressor" else 0
        )
        samples: dict[str, Any] = {
            "t": [round(x, 4) for x in self.t],
            "rss": self.rss,
            "n_procs": self.n_procs,
            "sys_available": self.sys_available,
        }
        if self.private:
            samples["private"] = self.private
        return {
            "interval_s": self.interval_s,
            "samples": samples,
            "peak_rss": max(self.rss) if self.rss else 0,
            # macOS: sum over the tree of each process's lifetime peak
            # footprint (0 elsewhere). The child reports its own exactly.
            "peak_footprint_tree": sum(self._fp_peak.values()),
            "max_procs": max(self.n_procs) if self.n_procs else 0,
            "min_sys_available": min(self.sys_available) if self.sys_available else 0,
            "swap_out_delta_bytes": swapped,
            "compressed_delta_bytes": compressed_delta,
            "pressure_kind": self.pressure_kind,
            "pressure_delta_bytes": pressure_delta,
            # What the run actually demanded of memory: what stayed resident
            # plus what the machine moved aside to keep it running. On a
            # machine under pressure these differ by several GB.
            "peak_demand_bytes": (max(self.rss) if self.rss else 0) + pressure_delta,
            # Windows only (0 elsewhere): the kernel's own high-water marks
            # for the tree, summed per process. They see between samples, so
            # a sampled peak far below them means the sampler missed a spike.
            "os_peak_rss": self._os_peak_rss,
            "os_peak_private": self._os_peak_private,
            # Windows only (None elsewhere): whether the compression store
            # was readable. False means the pressure term is pagefile only.
            "compression_store_found": self._store_found,
            "n": len(self.t),
        }


def _dir_bytes(path: str) -> int:
    total = 0
    p = Path(path)
    if not p.is_dir():
        return 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


class SpillWatcher:
    """Tracks the peak size of the spill directory during a run.

    Without this the out-of-core arm looks free: it trades RAM for SSD, and a
    comparison that reports only the RAM half of that trade is advertising,
    not measurement.
    """

    def __init__(self, path: str, interval_s: float = 0.25) -> None:
        self.path = path
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _dir_bytes(self.path))
            self._stop.wait(self.interval_s)

    def start(self) -> SpillWatcher:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> SpillWatcher:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self


def warm_page_cache(path: str) -> dict[str, Any]:
    """Read the file once so every arm starts from the same cache state.

    Dropping caches requires root on macOS, so the reproducible option is to
    warm rather than to clear. It is also the conservative choice for us: a
    cold cache would penalise whichever arm reads the file most, and that is
    the out-of-core arm.

    The read is *timed*, and that timing is the evidence for the claim. A
    file already resident in the page cache reads at memory bandwidth
    (several GB/s); one being fetched from SSD reads at device speed. So the
    returned ``throughput_gbps`` says whether this warm-up actually did any
    work, which is how "was an arm unfairly pre-cached?" stops being a
    worry and becomes a number on the record.
    """
    t0 = time.perf_counter()
    ok = False
    # Read the file in Python rather than shelling out to `cat`: this has to
    # run on Windows too, and a subprocess spawn is measurable overhead on a
    # small file anyway. 8 MB chunks into a scratch buffer, discarded.
    try:
        with open(path, "rb") as fh:
            while fh.readinto(_WARM_BUF):
                pass
        ok = True
    except (OSError, ValueError):
        pass
    elapsed = time.perf_counter() - t0
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    time.sleep(0.5)
    return {
        "warmed": ok,
        "read_s": round(elapsed, 4),
        "bytes": size,
        "throughput_gbps": round(size / elapsed / 1e9, 3) if elapsed > 0 else None,
        # A rough indicator, not a threshold anything depends on: SSDs on
        # this class of machine top out well under 5 GB/s, so faster than
        # that means the data was already in RAM before we asked.
        #
        # Only meaningful above ~100 MB. Below it the `cat` process spawn
        # (a few ms) dominates the read and the derived throughput
        # understates residency badly — a 14 MB file that *was* resident
        # measures 2.5 GB/s. Reported as None rather than as a wrong answer.
        "already_resident": (
            bool(size / elapsed > 5e9)
            if elapsed > 0 and size >= 100_000_000
            else None
        ),
    }


def run_one(
    *,
    arm: str,
    path: str,
    label: str,
    params: Params,
    results_dir: Path,
    rep: int = 1,
    timeout_s: float = 3600.0,
    memory_limit_bytes: int | None = None,
    interval_s: float = 0.02,
    cache: str = "warm",
    keep_artifacts: bool = True,
    facts: dict[str, int] | None = None,
    python: str | None = None,
    gate: Band | None = None,
    gate_timeout_s: float = 1800.0,
    needs_available_bytes: int = 0,
    pressured_fraction: float = 1.0,
) -> RunRecord:
    """Execute one arm on one dataset, in its own process, and record it.

    When `gate` is given, the run waits for the machine to fall inside the
    acceptance band before starting, and the state it started from is recorded.
    Measured 2026-09-08: identical code and thread count took 22.3 s on a
    loaded machine and 10.8 s on a quiet one, so a campaign that runs
    unattended must gate rather than hope.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")

    facts = facts if facts is not None else dataset_facts(path)
    record = RunRecord(
        arm=arm,
        arm_label=ARMS[arm].label,
        dataset=Dataset(label=label, path=str(path), cache_state=cache, **facts),
        params=params,
        rep=rep,
        env=capture_env(),
    ).finalize_id()

    artifacts_dir = results_dir / "artifacts" / record.run_id
    spill_dir = Path(tempfile.gettempdir()) / f"polariseq_spill_{record.run_id}"
    workdir = Path(tempfile.mkdtemp(prefix="telemetry_"))
    cfg_path, child_out = workdir / "config.json", workdir / "child.json"
    cfg_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "path": str(path),
                "params": params.__dict__,
                "artifacts_dir": str(artifacts_dir),
                "child_out": str(child_out),
                "interval_s": interval_s,
            }
        )
    )

    env = dict(os.environ)
    # Give every arm the same thread count, explicitly. Leaving these unset
    # means each library picks its own, and a comparison of two libraries at
    # different thread counts measures the launcher.
    if params.threads:
        for var in THREAD_ENV_VARS:
            env[var] = str(params.threads)
    env["POLARISEQ_SPILL_DIR"] = str(spill_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BENCH_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    # Gate before warming the cache: warming is itself I/O, and doing it while
    # the machine is busy would both distort the warm and waste the wait.
    gate_state: MachineState | None = None
    effective_gate = gate
    gate_need_capped = False
    if gate is not None:
        # The calibrated floor describes the machine, not the run. Calibration
        # takes 70% of whatever is free at the time, so on a loaded machine it
        # adapts *downward* and admits runs it cannot hold: measured
        # 2026-09-09, a floor of 2.76 GB admitted brain100k/seurat-tuned runs
        # that peaked at 5.35 GB, and every one of eight such runs peaked above
        # the memory free when it started. Raise the floor to what this run is
        # known to need so the gate refuses rather than pages.
        # ...but never above what the machine has free at rest. A run whose
        # known peak exceeds that will not be admitted by waiting, however
        # long: the gate would sit out its whole timeout and start the run
        # regardless. Measured on the 13.86 GiB Windows laptop, whose idle
        # usage is 30-45 per cent: the 800K in-memory arm needs ~11.9 GB and
        # the machine offers ~9, so every repeat cost 30 minutes of waiting
        # and gained nothing. Where the cap binds, the run is inherently
        # pressured on this machine, and the record says so.
        need = int(needs_available_bytes)
        ceiling = int(gate.idle_available_bytes or need)
        gate_need_capped = need > ceiling
        target = min(need, ceiling)
        if target > gate.min_available_bytes:
            effective_gate = replace(gate, min_available_bytes=target)
        if gate_need_capped and pressured_fraction < 1.0:
            # Relaxed rule (decided 2026-09-29): a run that needs more than the
            # machine ever has free at rest is memory-bound whatever it starts
            # with, and waiting for the at-rest median fails about half the
            # time on an 8 GB laptop (normal fluctuation around a median).
            # Such runs start once available memory reaches this fraction of
            # the at-rest level; CPU and load criteria are unchanged.
            relaxed = int(ceiling * pressured_fraction)
            effective_gate = replace(gate, min_available_bytes=min(gate.min_available_bytes, relaxed))
        if gate_need_capped:
            print(f"    [loadgate] this arm needs {need / 1e9:.2f} GB available but the "
                  f"machine offers {ceiling / 1e9:.2f} GB at rest; starting under "
                  f"pressure rather than waiting", flush=True)
        gate_state = wait_for_quiet(effective_gate, timeout_s=gate_timeout_s)

    cache_info: dict[str, Any] = {"mode": cache}
    if cache == "warm":
        cache_info.update(warm_page_cache(str(path)))

    cmd = [python or sys.executable, "-m", "telemetry.child", str(cfg_path)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    parent = ParentSampler(proc.pid, PARENT_INTERVAL_S)
    parent.memory_limit = memory_limit_bytes
    parent.start()
    spill = SpillWatcher(str(spill_dir)).start()
    status, note = "ok", ""
    try:
        _out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        _out, err = proc.communicate()
        status, note = "timeout", f"exceeded {timeout_s:.0f}s"
    finally:
        parent.stop()
        spill.stop()
    wall = time.perf_counter() - t0

    child_payload: dict[str, Any] = {}
    if child_out.exists():
        with contextlib.suppress(json.JSONDecodeError):
            child_payload = json.loads(child_out.read_text())

    if parent.exceeded:
        status = "exceeds_memory"
        note = (f"stopped when its footprint passed {memory_limit_bytes / 1e9:.1f} GB, "
                "the campaign's limit (1.5x installed memory)")
    if status == "ok":
        if child_payload:
            status = str(child_payload.get("status", "ok"))
            note = str(child_payload.get("note", ""))
        elif proc.returncode and proc.returncode < 0:
            # Negative return code == killed by signal. SIGKILL during a large
            # allocation on macOS/Linux is the OOM killer's signature.
            status = "oom"
            note = f"killed by signal {-proc.returncode}; likely OOM"
        elif "memory allocation of" in (err or ""):
            # Rust's allocation-failure abort. Windows has no OOM killer: an
            # allocation past the commit limit fails, the process aborts, and
            # the exit code is positive, so the signal test above misses it.
            status = "oom"
            note = f"rc={proc.returncode}; allocation failed: {err.strip()[-300:]}"
        else:
            status = "failed"
            note = f"rc={proc.returncode}; no child record. stderr tail: {err.strip()[-500:]}"

    if gate_state is not None and not gate_state.ok:
        # Visible in the console line and in the record, so a run measured
        # through interference is never quietly averaged in with the rest.
        why = "; ".join(gate_state.reasons) or "gate timeout"
        note = (note + " | " if note else "") + f"GATE NOT MET: {why}"

    record.status = status
    record.note = note
    record.wall_s = round(wall, 3)
    record.steps = child_payload.get("steps", [])
    record.child = child_payload.get("telemetry", {})
    record.parent = parent.as_dict()
    record.outputs = child_payload.get("outputs", {})
    record.outputs["tuning_notes"] = ARMS[arm].tuning_notes
    record.disk = {
        "page_cache": cache_info,
        "machine_at_start": gate_state.as_dict() if gate_state else None,
        "gate_band": effective_gate.as_dict() if effective_gate is not None else None,
        # False means the gate waited its full timeout and the run went ahead
        # anyway. wait_for_quiet's contract is that such a run is measured
        # through interference, so it is flagged here rather than silently
        # mixed into the medians.
        "gate_ok": bool(gate_state.ok) if gate_state is not None else None,
        "needs_available_bytes": int(needs_available_bytes),
        "gate_need_capped": gate_need_capped,
        "gate_pressured_fraction": pressured_fraction if gate_need_capped else None,
        "spill_dir": str(spill_dir),
        "spill_peak_bytes": spill.peak,
        "input_file_bytes": facts.get("file_bytes", 0),
    }
    record.peaks = {
        "child_rss": record.child.get("peaks", {}).get("rss", 0),
        "child_uss": record.child.get("peaks", {}).get("uss", 0),
        "parent_rss": record.parent.get("peak_rss", 0),
        # The headline memory figure (macOS): the measured process's own
        # lifetime peak footprint, read by the kernel at exit, plus that of
        # any subprocess it ran (Seurat's R), sampled from the parent.
        # An arm that runs its pipeline in a subprocess (R) reports that
        # process's peak itself, because the measured Python process also
        # held the untimed input conversion.
        "footprint": int(record.outputs.get("footprint_bytes") or max(
            int(child_payload.get("footprint", {}).get("lifetime_max_phys_footprint", 0)),
            int(record.parent.get("peak_footprint_tree", 0)),
        )),
    }
    if child_payload.get("traceback"):
        record.outputs["traceback"] = child_payload["traceback"]

    record.write(results_dir)
    shutil.rmtree(workdir, ignore_errors=True)
    shutil.rmtree(spill_dir, ignore_errors=True)
    if not keep_artifacts:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
    return record
