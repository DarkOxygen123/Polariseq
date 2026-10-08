"""What this machine has, and what Polariseq is allowed to use.

Polariseq's premise is that a laptop should be able to analyse a
multi-million-cell dataset. That only works if the library knows what it is
running on: how much memory is actually free (not installed), how many cores
are worth using, and how much scratch space the spill directory has.

Two objects:

- :class:`Resources` — what was *detected*, plus how each number was obtained
  (``available_source``), because an estimated figure and a measured one
  deserve different amounts of trust.
- :class:`Budget` — what Polariseq may *use*. Detected by default, overridable
  in whole or in part, and the single thing the planner and the kernels
  consult.

Detection is pure standard library: no new dependency, no ``unsafe``, and it
works the same on macOS, Linux and Windows. Where the standard library cannot answer
(available memory on macOS), the fallbacks are tried in order and the winner
is recorded rather than hidden.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["Resources", "Budget", "detect_resources"]

_GB = 1_000_000_000


def _total_ram_bytes() -> int:
    """Installed RAM. `sysconf` covers macOS and Linux, the kernel32 call
    covers Windows (which has no `os.sysconf`); 0 if unknown."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Windows":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                    "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual",
                )
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            pass
    return 0


def _available_ram_bytes(total: int) -> tuple[int, str]:
    """Memory that could be allocated now, and where the number came from.

    Tried in order of trustworthiness. The last resort is a fraction of the
    installed total, which is a guess and is labelled as one.
    """
    # 1. psutil, if the user happens to have it: the best answer on every OS.
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().available), "psutil"
    except Exception:
        pass

    # 2. Linux: MemAvailable is the kernel's own estimate and accounts for
    #    reclaimable page cache.
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024, "meminfo"
    except OSError:
        pass

    # 3. POSIX free pages (Linux; not implemented on macOS).
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), "sysconf"
    except (ValueError, OSError, AttributeError):
        pass

    # 4. macOS: vm_stat. Free + inactive + speculative is what a large
    #    allocation can realistically claim (inactive pages are evictable).
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
            page = 4096
            counts: dict[str, int] = {}
            for line in out.splitlines():
                if line.startswith("Mach Virtual Memory Statistics"):
                    if "page size of" in line:
                        page = int(line.split("page size of")[1].split()[0])
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.strip().rstrip(".")
                    if val.isdigit():
                        counts[key.strip()] = int(val)
            pages = (
                counts.get("Pages free", 0)
                + counts.get("Pages inactive", 0)
                + counts.get("Pages speculative", 0)
            )
            if pages:
                return pages * page, "vm_stat"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    # 5. Nothing worked: assume most of the machine is usable, and say so.
    return int(total * 0.7), "estimated"


def _performance_cores(logical: int) -> int:
    """Cores worth scheduling memory-bound work on.

    Apple Silicon reports efficiency cores alongside performance ones; on
    memory-bound kernels they finish late and gate the epoch (measured in
    phase 3: with large chunks, 8 threads performed the same as 4). Falls
    back to the logical count when the distinction does not apply.
    """
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            if out.isdigit() and 0 < int(out) <= logical:
                return int(out)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return logical


def _usable_cores() -> int:
    """Cores this process may run on (respects cgroup/taskset pinning)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class Resources:
    """What was detected on this machine."""

    total_ram_bytes: int
    available_ram_bytes: int
    #: How ``available_ram_bytes`` was obtained: ``psutil``, ``meminfo``,
    #: ``sysconf``, ``vm_stat`` or ``estimated``.
    available_source: str
    logical_cores: int
    performance_cores: int
    spill_dir: str
    spill_free_bytes: int
    platform: str
    machine: str

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Resources(ram={self.total_ram_bytes / _GB:.1f} GB total, "
            f"{self.available_ram_bytes / _GB:.1f} GB available [{self.available_source}], "
            f"cores={self.performance_cores}P/{self.logical_cores}, "
            f"spill={self.spill_free_bytes / _GB:.0f} GB free at {self.spill_dir})"
        )

    def summary(self) -> dict[str, Any]:
        """The detected values as a plain dict."""
        return {
            "total_ram_gb": round(self.total_ram_bytes / _GB, 2),
            "available_ram_gb": round(self.available_ram_bytes / _GB, 2),
            "available_source": self.available_source,
            "logical_cores": self.logical_cores,
            "performance_cores": self.performance_cores,
            "spill_dir": self.spill_dir,
            "spill_free_gb": round(self.spill_free_bytes / _GB, 2),
            "platform": self.platform,
            "machine": self.machine,
        }


def free_disk_bytes(path: str) -> int:
    """Free space on the disk holding ``path``, measured at its nearest
    existing ancestor so the folder need not exist yet (0 if unreadable)."""
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or "/").free
    except OSError:
        return 0


def detect_resources(spill_dir: str | None = None) -> Resources:
    """Inspect the machine. Cheap (a few milliseconds) and side-effect free."""
    spill = spill_dir or os.path.join(tempfile.gettempdir(), "polariseq_spill")
    total = _total_ram_bytes()
    avail, source = _available_ram_bytes(total)
    logical = _usable_cores()
    free = free_disk_bytes(spill)
    return Resources(
        total_ram_bytes=total,
        available_ram_bytes=min(avail, total) if total else avail,
        available_source=source,
        logical_cores=logical,
        performance_cores=_performance_cores(logical),
        spill_dir=spill,
        spill_free_bytes=free,
        platform=platform.system(),
        machine=platform.machine(),
    )


#: Fraction of *available* memory a detected budget will claim. The rest is
#: headroom for the interpreter, the page cache and whatever else the user is
#: running — exceeding available memory does not fail cleanly, it swaps, and a
#: swapping run is worse than a slower one that fits.
DEFAULT_RAM_FRACTION = 0.75


@dataclass(frozen=True)
class Budget:
    """What Polariseq may use for one workload.

    Attributes:
        ram_bytes: Ceiling for in-memory data structures.
        threads: Worker threads for parallel kernels.
        spill_dir: Where out-of-core scratch files are written.
        spill_bytes: Ceiling for those files.
        allow_lossy: Whether the planner may trade accuracy for fit. Off by
            default: Polariseq will choose a slower exact plan over a faster
            approximate one unless told otherwise, and any approximation it
            does make is reported.
        detected: The :class:`Resources` this was derived from, if any.
    """

    ram_bytes: int
    threads: int
    spill_dir: str
    spill_bytes: int
    allow_lossy: bool = False
    detected: Resources | None = None

    @classmethod
    def detect(
        cls,
        *,
        ram_fraction: float = DEFAULT_RAM_FRACTION,
        spill_dir: str | None = None,
        allow_lossy: bool = False,
        prefer_performance_cores: bool = True,
    ) -> Budget:
        """Build a budget from the current machine."""
        res = detect_resources(spill_dir)
        threads = res.performance_cores if prefer_performance_cores else res.logical_cores
        return cls(
            ram_bytes=int(res.available_ram_bytes * ram_fraction),
            threads=max(1, threads),
            spill_dir=res.spill_dir,
            # Leave a tenth of the free space alone; filling a disk is a much
            # worse failure than declining to start.
            spill_bytes=int(res.spill_free_bytes * 0.9),
            allow_lossy=allow_lossy,
            detected=res,
        )

    def with_overrides(
        self,
        *,
        ram_gb: float | None = None,
        threads: int | None = None,
        spill_dir: str | None = None,
        spill_gb: float | None = None,
        allow_lossy: bool | None = None,
    ) -> Budget:
        """A copy with the given fields replaced; the rest stays detected."""
        changes: dict[str, Any] = {}
        if ram_gb is not None:
            changes["ram_bytes"] = int(ram_gb * _GB)
        if threads is not None:
            changes["threads"] = max(1, int(threads))
        if spill_dir is not None:
            changes["spill_dir"] = spill_dir
            if spill_gb is None:
                # The ceiling follows the folder to its disk, which may not be
                # the one it was detected on (a project on an external SSD).
                changes["spill_bytes"] = int(free_disk_bytes(spill_dir) * 0.9)
        if spill_gb is not None:
            changes["spill_bytes"] = int(spill_gb * _GB)
        if allow_lossy is not None:
            changes["allow_lossy"] = bool(allow_lossy)
        return replace(self, **changes)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / _GB

    @property
    def spill_gb(self) -> float:
        return self.spill_bytes / _GB

    def __repr__(self) -> str:  # pragma: no cover - display only
        lossy = "lossy allowed" if self.allow_lossy else "exact only"
        return (
            f"Budget(ram={self.ram_gb:.1f} GB, threads={self.threads}, "
            f"spill={self.spill_gb:.0f} GB at {self.spill_dir}, {lossy})"
        )

    def summary(self) -> dict[str, Any]:
        """The budget as a plain dict."""
        return {
            "ram_gb": round(self.ram_gb, 2),
            "threads": self.threads,
            "spill_dir": self.spill_dir,
            "spill_gb": round(self.spill_gb, 2),
            "allow_lossy": self.allow_lossy,
        }
