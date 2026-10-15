"""Shared pieces of the showdown drivers (standard library only).

The contract every driver follows, so that monitor.py can measure any tool the same way:

* ``STEP <name> <t_begin> <t_end>`` on stderr after each step, times in epoch seconds
* the last line of stdout is one JSON object (the driver's own summary)
* artifacts (labels, UMAP, flags) are written after the last step, outside the timed region
* a cell is identified by its global row: the position in the concatenation of the input files in natural order
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "params.json").read_text())


def natural(name: str) -> list:
    """Sort key: 2.h5ad before 10.h5ad."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def shard_files(spec: str) -> list[str]:
    """Comma-separated files, folders (every .h5ad in them) or patterns, in natural order."""
    out: list[str] = []
    for part in [s for s in spec.split(",") if s]:
        p = Path(part)
        if p.is_dir():
            out += sorted((str(f) for f in p.glob("*.h5ad")), key=natural)
        elif any(c in part for c in "*?["):
            out += sorted(glob.glob(part), key=natural)
        else:
            out.append(part)
    missing = [f for f in out if not Path(f).is_file()]
    if not out or missing:
        raise SystemExit(f"no such input: {missing[:1] or spec}")
    return out


def shard_sizes(files: list[str]) -> list[int]:
    """Cells per file, read from the files' X shape (h5py)."""
    import h5py
    sizes = []
    for f in files:
        with h5py.File(f, "r") as h:
            sizes.append(int(h["X"].attrs["shape"][0]) if "shape" in h["X"].attrs else int(h["X"].shape[0]))
    return sizes


class Steps:
    """``with steps("name"):`` prints the STEP line on exit (also when the block raises)."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.done: list[tuple[str, float, float]] = []

    @contextmanager
    def __call__(self, name: str):
        b = time.time()
        try:
            yield
        finally:
            e = time.time()
            self.done.append((name, b, e))
            print(f"STEP {name} {b:.3f} {e:.3f}", file=sys.stderr, flush=True)


def emit(result: dict) -> None:
    """The driver's summary: the last line of stdout."""
    print(json.dumps(result, default=str), flush=True)
