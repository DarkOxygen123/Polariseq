"""Time and memory tracking for the examples.

    from _resources import ResourceLog

    with ResourceLog(watch="/scratch/me") as log:
        with log.step("pca"):
            ...                       # your analysis
    log.save("out/")                  # resources.csv, resources.png, steps in the dict
    print(log.summary())

A background thread samples this process twice a second: the memory it holds
(resident set size), the memory the machine still has free, and the free
space on the disk Polariseq writes to. Each ``step`` records its wall time
and the peak it reached. Standard library only; if ``psutil`` is installed it
is used for the readings, otherwise ``/proc`` (Linux) or ``ps`` (macOS).

Two things the numbers do not say:

- The resident set size is what the process holds in RAM. On macOS the
  system compresses memory under pressure, so a run can hold more than its
  resident size shows (macOS reports the larger "physical footprint" in
  Activity Monitor). On Linux and Windows the resident size is the figure
  to watch.
- Polariseq runs inside this process, so its Rust threads are counted. Tools
  you start as separate processes are not.
"""
from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:  # not on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None
try:  # optional
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

GB = 1e9


def _rss_bytes() -> int:
    """Memory the process holds in RAM right now."""
    if psutil is not None:
        return psutil.Process().memory_info().rss
    try:  # Linux
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, AttributeError):
        pass
    try:  # macOS
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return 0   # Windows without psutil: install psutil
    return int(out) * 1024 if out else 0


def _free_ram_bytes() -> int | None:
    """Memory the machine can still hand out without swapping, if known."""
    if psutil is not None:
        return psutil.virtual_memory().available
    try:  # Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _peak_rss_bytes() -> int:
    """The operating system's own record of the highest resident size so far
    (bytes on macOS, kibibytes on Linux, the peak working set on Windows)."""
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if platform.system() == "Darwin" else peak * 1024
    if psutil is not None:
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss)   # Windows keeps the peak itself
    return 0


class ResourceLog:
    """Samples memory and disk in the background and times named steps."""

    def __init__(self, interval: float = 0.5, watch: str | Path | None = None,
                 verbose: bool = False) -> None:
        self.verbose = verbose
        self.interval = interval
        self.watch = str(watch) if watch else None   # a path on the disk Polariseq writes to
        self.samples: list[tuple[float, float, float | None, float | None, str]] = []
        self.steps: list[dict] = []
        self._current = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._disk0: float | None = None

    # -- sampling -----------------------------------------------------------
    def _disk_free(self) -> float | None:
        if not self.watch:
            return None
        try:
            return shutil.disk_usage(self.watch).free / GB
        except OSError:
            return None

    def _sample(self) -> None:
        free_ram = _free_ram_bytes()
        self.samples.append((
            time.perf_counter() - self._t0, _rss_bytes() / GB,
            None if free_ram is None else free_ram / GB, self._disk_free(), self._current))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> ResourceLog:
        self._t0 = time.perf_counter()
        self._disk0 = self._disk_free()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._sample()

    def __enter__(self) -> ResourceLog:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- steps --------------------------------------------------------------
    @contextmanager
    def step(self, name: str):
        """Time a block and record the most memory the process held in it."""
        self._current = name
        start = time.perf_counter() - self._t0
        self._sample()
        if self.verbose:   # a live line per step: a long run's log shows where it is
            print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        try:
            yield
        finally:
            self._sample()
            end = time.perf_counter() - self._t0
            peak = max((s[1] for s in self.samples if s[4] == name), default=0.0)
            if self.verbose:
                print(f"[{time.strftime('%H:%M:%S')}] {name} done in {_duration(end - start)}, "
                      f"peak memory {peak:.1f} GB", flush=True)
            self.steps.append({"step": name, "start_s": start, "seconds": end - start,
                               "peak_rss_gb": peak})
            self._current = ""

    # -- results ------------------------------------------------------------
    def summary(self) -> dict:
        rss = [s[1] for s in self.samples]
        free = [s[2] for s in self.samples if s[2] is not None]
        disk = [s[3] for s in self.samples if s[3] is not None]
        return {
            "seconds": self.samples[-1][0] if self.samples else 0.0,
            "peak_rss_gb_sampled": max(rss, default=0.0),
            "peak_rss_gb_os": _peak_rss_bytes() / GB,
            "min_free_ram_gb": min(free, default=None),
            "disk_written_gb": (self._disk0 - min(disk)
                                if disk and self._disk0 is not None else None),
            "steps": self.steps,
        }

    def save(self, out: str | Path) -> dict[str, Path]:
        """Write ``resources.csv`` (one row per sample) and ``resources.png``."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "resources.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seconds", "rss_gb", "free_ram_gb", "free_disk_gb", "step"])
            w.writerows(self.samples)
        return {"csv": csv_path, "png": self.plot(out / "resources.png")}

    def plot(self, path: str | Path) -> Path:
        """Memory held, memory left free and disk written against time, with
        each step shaded."""
        import matplotlib.pyplot as plt

        long = self.samples[-1][0] > 180          # minutes for a long run, seconds for a short one
        unit, k = ("minutes", 60.0) if long else ("seconds", 1.0)
        t = [s[0] / k for s in self.samples]
        rows = 3 if any(s[3] is not None for s in self.samples) else 2
        fig, axes = plt.subplots(rows, 1, figsize=(7.2, 1.9 * rows + 0.6), sharex=True)
        shade = ["#eef2f8", "#ffffff"]
        for ax in axes:
            for i, s in enumerate(self.steps):
                ax.axvspan(s["start_s"] / k, (s["start_s"] + s["seconds"]) / k,
                           color=shade[i % 2], lw=0, zorder=0)
            ax.grid(axis="y", color="#e6e5e0", lw=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        a = axes[0]
        a.plot(t, [s[1] for s in self.samples], color="#2a78d6", lw=1.2)
        a.set_ylabel("memory held (GB)")
        a.set_ylim(bottom=0)
        a.set_title("Memory held by the run, with its steps shaded", loc="left", fontsize=9,
                    fontweight="bold")
        top = a.get_ylim()[1]
        for x_s, name, turned in step_labels(self.steps, self.samples[-1][0]):
            a.text(x_s / k, top * 0.97, name, ha="center", va="top",
                   rotation=90 if turned else 0, fontsize=6.0 if turned else 6.5,
                   color="#52514e")
        b = axes[1]
        free = [s[2] for s in self.samples]
        if any(v is not None for v in free):
            b.plot(t, [float("nan") if v is None else v for v in free], color="#1baf7a", lw=1.2)
            b.set_ylim(bottom=0)
        else:
            b.text(0.5, 0.5, "free memory not available on this system (install psutil)",
                   transform=b.transAxes, ha="center", va="center", fontsize=8, color="#8a8880")
        b.set_ylabel("free memory (GB)")
        if rows == 3:
            c = axes[2]
            disk = [float("nan") if s[3] is None else s[3] for s in self.samples]
            first = next((v for v in disk if v == v), 0.0)
            c.plot(t, [first - v for v in disk], color="#8e5cc8", lw=1.2)
            c.set_ylabel("disk written (GB)")
            c.set_ylim(bottom=0)
        axes[-1].set_xlabel(f"{unit} since start")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return Path(path)


def step_labels(steps: list[dict], total_s: float, per_char: float = 0.0085,
                pad: float = 0.010) -> list[tuple[float, str, bool]]:
    """Where to name each step along the top of the memory plot: its middle,
    its name up to any parenthesis ("qc (and the one-time copy ...)" is "qc"),
    and whether to turn it upright because its band is narrow. A step wide
    enough for its name is written flat; a narrower one is written turned;
    only a sliver, under 0.8% of the run, is left unnamed. Every step that
    can be seen is named. ``per_char`` is the share of the run's width one
    character takes (0.0085 for the 7-inch plot at 6.5 points; a narrower
    panel needs more), and ``pad`` the room a flat name leaves around it."""
    out = []
    for s in steps:
        share = s["seconds"] / max(total_s, 1e-9)
        if share < 0.008:
            continue
        name = s["step"].split(" (")[0]
        flat = share >= per_char * len(name) + pad
        out.append((s["start_s"] + s["seconds"] / 2, name, not flat))
    return out


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _filesystem_type(path: str | Path) -> str | None:
    """The file system a path lives on (Linux): the mount point that is its
    longest prefix in /proc/mounts."""
    try:
        real = os.path.realpath(path)
        best, kind = "", None
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                inside = real == parts[1] or real.startswith(parts[1].rstrip("/") + "/")
                if inside and len(parts[1]) > len(best):
                    best, kind = parts[1], parts[2]
        return kind
    except OSError:
        return None


def describe_machine(path: str | Path | None = None) -> dict:
    """What the run ran on, to state in a methods section: processor, cores,
    memory, operating system, the scheduler's allocation if any, and the file
    system and free space where the project lives. No host or site names."""
    info: dict = {"os": platform.platform(), "python": platform.python_version()}
    cpu = None
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as fh:
                cpu = next((ln.split(":", 1)[1].strip() for ln in fh
                        if ln.startswith("model name")), None)
        elif sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        cpu = None
    info["cpu"] = cpu or platform.processor() or None
    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count()
        info["memory_gb"] = round(psutil.virtual_memory().total / GB, 1)
    except ImportError:
        info["cores_logical"] = os.cpu_count()
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            info["memory_gb"] = round(pages / GB, 1)
    alloc = {k: os.environ[v] for k, v in (("cpus", "SLURM_CPUS_PER_TASK"),
                                            ("memory_mb", "SLURM_MEM_PER_NODE")) if v in os.environ}
    if alloc:
        info["scheduler_allocation"] = alloc
    if path is not None:
        info["project_disk_free_gb"] = round(shutil.disk_usage(path).free / GB, 1)
        fs = _filesystem_type(path)
        if fs:
            info["project_filesystem"] = fs
    return info
