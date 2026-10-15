#!/usr/bin/env python
"""Download the Tahoe-100M parquet files of the chosen plates (login node: compute nodes have no internet).

    python tahoe_fetch.py --plates plate1,plate2 --index ../data/tahoe_files.csv --out DIR [--parallel 8]

Reads the index written by tahoe_index.py, downloads every file of the plates named (with curl, retried,
resumed), checks that each ends with the parquet magic bytes, and also fetches metadata/gene_metadata.parquet.
Prints the cells and gigabytes to expect first. Files already complete are skipped, so it can be rerun.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/"


def complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 12:
        return False
    with path.open("rb") as f:
        f.seek(-4, 2)
        return f.read(4) == b"PAR1"


def fetch(rel: str, out: Path) -> tuple[str, bool]:
    dest = out / Path(rel).name
    for _ in range(5):
        if complete(dest):
            return rel, True
        subprocess.run(["curl", "-sSL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", str(dest), BASE + rel],
                       check=False)
        if not complete(dest) and dest.exists():
            dest.unlink()   # a truncated file cannot be resumed reliably through the CDN redirect
    return rel, complete(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", required=True, help="comma-separated, e.g. plate1,plate2")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    plates = set(a.plates.split(","))
    rows = [r for r in csv.DictReader(open(a.index)) if r["plate"] in plates]
    if not rows:
        raise SystemExit(f"no files for {plates} in {a.index}")
    cells = sum(int(r["cells"]) for r in rows)
    print(f"{len(rows)} files, {cells:,} cells, about {len(rows) * 0.085:.0f} GB of parquet "
          f"(about {cells * 1300 * 8 / 1e9:.0f} GB once converted to .h5ad)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = ["metadata/gene_metadata.parquet"] + [f"data/{r['file']}" for r in rows]
    bad = []
    with ThreadPoolExecutor(a.parallel) as ex:
        for k, (rel, ok) in enumerate(ex.map(lambda r: fetch(r, out), todo)):
            if not ok:
                bad.append(rel)
            if k % 50 == 0:
                print(f"{k}/{len(todo)} {rel} {'ok' if ok else 'FAILED'}", flush=True)
    print(f"done: {len(todo) - len(bad)} complete, {len(bad)} failed {bad[:5]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
