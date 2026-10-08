"""The run record: what a measurement is, and what has to travel with it.

A number without its provenance is not a measurement. Every record here
carries the machine, the library versions, the thread settings, the git
commit, the exact parameters, and the page-cache state, because a
performance claim in the README or the results has to be re-derivable a year
later by someone who was not there.

The schema is versioned (:data:`SCHEMA`). Add fields freely; change the
meaning of one and you must bump the version, because ``analyze`` and the
figure code read records written weeks earlier.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "polariseq.telemetry/1"

#: Thread-count environment variables. Every arm is launched with these set
#: identically: if one arm gets 8 threads and the other 1, the comparison is
#: about the launcher, not the libraries. `scanpy` in particular defaults to
#: `n_jobs=1`, which pins numba inside pynndescent — a known 10x penalty
#: (measured in the fairness audit of the arms) that is a configuration artefact
#: rather than a property of the library.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARISEQ_THREADS",
)

_VERSIONED = (
    "polariseq",
    "scanpy",
    "anndata",
    "numpy",
    "scipy",
    "pandas",
    "numba",
    "igraph",
    "leidenalg",
    "umap-learn",
    "pynndescent",
    "psutil",
    "h5py",
)


def _cpu_brand() -> str:
    try:
        if platform.system() == "Windows":
            # `platform.processor()` on Windows is the CPUID family string
            # ("AMD64 Family 25 Model 80 ..."), not the model name a methods
            # section needs; the registry holds the marketing name.
            import winreg

            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                ) as key:
                    return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            except OSError:
                return platform.processor() or "unknown"
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _git_sha(repo: Path | None = None) -> str:
    root = repo or Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {"python": platform.python_version()}
    for name in _VERSIONED:
        with contextlib.suppress(Exception):
            out[name] = md.version(name)
    return out


def _power_plugged() -> bool | None:
    """Mains power, for a laptop; None where there is no battery to ask.

    A laptop on battery lowers its clocks, so a record taken unplugged is
    not comparable with one taken on the charger.
    """
    import psutil

    try:
        battery = psutil.sensors_battery()
    except (AttributeError, NotImplementedError, OSError, RuntimeError):
        return None
    return None if battery is None else bool(battery.power_plugged)


def capture_env() -> dict[str, Any]:
    """Everything about *this* process that could change a number."""
    import psutil

    vm = psutil.virtual_memory()
    try:
        # Windows: the pagefile, so RAM + this is the commit limit past which
        # an allocation fails outright (the 2026-09-19 laptop campaign: 2 GB,
        # and three runs refused at 13.3-14.5 GB). macOS/Linux: swap in use
        # or configured, which only slows an oversubscribed run.
        swap_total = int(psutil.swap_memory().total)
    except (OSError, RuntimeError):
        swap_total = 0
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "logical_cores": os.cpu_count() or 0,
        "total_ram_bytes": int(vm.total),
        "swap_total_bytes": swap_total,
        "available_ram_bytes_at_start": int(vm.available),
        "git_sha": _git_sha(),
        "versions": _versions(),
        "thread_env": {k: os.environ.get(k, "") for k in THREAD_ENV_VARS},
        "executable": sys.executable,
        "hostname": platform.node(),
        "power_plugged": _power_plugged(),
    }


@dataclass
class Dataset:
    """The input, described well enough to be re-fetched and re-checked."""

    label: str
    path: str
    n_cells: int = 0
    n_genes: int = 0
    nnz: int = 0
    file_bytes: int = 0
    #: ``warm`` (the file was read once before the run so every arm starts
    #: from the same page-cache state), or ``as-is``. Dropping caches needs
    #: root on macOS, so *warming for everyone* is the reproducible choice —
    #: and it is the conservative one here, since a cold cache would penalise
    #: whichever arm reads the file most, which is ours.
    cache_state: str = "warm"


@dataclass
class Params:
    """The pipeline, pinned. Identical across arms or the comparison is void."""

    n_hvg: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    resolution: float = 1.0
    target_sum: float = 1e4
    min_genes: int = 200
    seed: int = 0
    with_umap: bool = True
    threads: int = 0  # 0 = "whatever the harness detected"; recorded either way
    #: RAM ceiling for the budget, pinned per campaign. 0 = detect at
    #: import — which makes the spill arm's block size (and with it the
    #: block-fold summation order, and with *that* the HVG boundary
    #: genes) depend on what else was running. Pinned, runs reproduce.
    ram_gb: float = 0.0


@dataclass
class RunRecord:
    """One execution of one arm on one dataset."""

    arm: str
    arm_label: str
    dataset: Dataset
    params: Params
    rep: int = 1
    schema: str = SCHEMA
    run_id: str = ""
    status: str = "ok"  # ok | failed | timeout | oom | exceeds_memory | skipped
    note: str = ""
    wall_s: float = 0.0
    env: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    child: dict[str, Any] = field(default_factory=dict)
    parent: dict[str, Any] = field(default_factory=dict)
    peaks: dict[str, int] = field(default_factory=dict)
    disk: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    def finalize_id(self) -> RunRecord:
        if not self.run_id:
            stamp = self.timestamp.replace("-", "").replace(":", "")
            self.run_id = f"{self.dataset.label}__{self.arm}__rep{self.rep}__{stamp}"
        return self

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, directory: str | Path) -> Path:
        self.finalize_id()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.run_id}.json"
        p.write_text(json.dumps(self.to_json(), indent=1))
        return p


def load_records(directory: str | Path) -> list[dict[str, Any]]:
    """Every record in a results directory, oldest first. Unknown schema versions
    are skipped loudly rather than silently mixed into a table."""
    out: list[dict[str, Any]] = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[telemetry] unreadable record {p.name}: {exc}", file=sys.stderr)
            continue
        if rec.get("schema") != SCHEMA:
            print(
                f"[telemetry] skipping {p.name}: schema {rec.get('schema')!r} "
                f"!= {SCHEMA!r}",
                file=sys.stderr,
            )
            continue
        rec["_path"] = str(p)
        out.append(rec)
    return out
