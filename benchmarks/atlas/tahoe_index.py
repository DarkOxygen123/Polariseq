#!/usr/bin/env python
"""Index Tahoe-100M's 3,388 parquet files by plate without downloading them (login node, network only).

    python tahoe_index.py --out ../data/tahoe_files.csv [--threads 16]

Each file holds one sub-library of one plate. Its footer (the last kilobytes, fetched with an HTTP range
request) gives the number of cells and, from the column statistics, the plate and the sub-library.
Writes file,plate,sublibrary,cells. Files that fail are listed with plate "?" and retried on a second run.
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/data/"
N_FILES = 3388


def name(i: int) -> str:
    return f"train-{i:05d}-of-{N_FILES:05d}.parquet"


def tail(url: str, n: int) -> bytes:
    with urlopen(Request(url, headers={"Range": f"bytes=-{n}"}), timeout=120) as r:
        return r.read()


def footer(i: int) -> dict:
    url = BASE + name(i)
    for attempt in range(4):
        try:
            data = tail(url, 262144)
            flen = struct.unpack("<I", data[-8:-4])[0]
            if data[-4:] != b"PAR1":
                raise ValueError("not a parquet tail")
            if flen + 8 > len(data):
                data = tail(url, flen + 8)
            md = pq.read_metadata(pa.BufferReader(b"PAR1" + data[-(flen + 8):]))
            stats = {}
            rg = md.row_group(0)
            for j in range(rg.num_columns):
                c = rg.column(j)
                if c.path_in_schema in ("plate", "BARCODE_SUB_LIB_ID") and c.statistics is not None:
                    stats[c.path_in_schema] = (c.statistics.min, c.statistics.max)
            plate = stats.get("plate", ("?", "?"))
            lib = stats.get("BARCODE_SUB_LIB_ID", ("?", "?"))
            return {"file": name(i), "plate": plate[0] if plate[0] == plate[1] else f"{plate[0]}..{plate[1]}",
                    "sublibrary": str(lib[0]).split("-")[-1], "cells": md.num_rows}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"file": name(i), "plate": "?", "sublibrary": "?", "cells": 0, "error": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    done = {}
    if out.exists():
        done = {r["file"]: r for r in csv.DictReader(out.open()) if r["plate"] != "?"}
    todo = [i for i in range(N_FILES) if name(i) not in done]
    print(f"{len(done)} indexed, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for k, r in enumerate(ex.map(footer, todo)):
            done[r["file"]] = r
            if k % 200 == 0:
                print(k, r, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "plate", "sublibrary", "cells"], extrasaction="ignore")
        w.writeheader()
        for f in sorted(done):
            w.writerow(done[f])
    bad = sum(1 for r in done.values() if r["plate"] == "?")
    plates: dict[str, int] = {}
    for r in done.values():
        plates[r["plate"]] = plates.get(r["plate"], 0) + int(r["cells"])
    for p in sorted(plates, key=lambda s: (len(s), s)):
        print(f"{p}\t{plates[p]:,} cells", flush=True)
    print(f"failed: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
