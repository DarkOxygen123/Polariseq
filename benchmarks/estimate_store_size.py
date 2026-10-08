"""Bytes per stored value of a count matrix under several lossless layouts.

    python estimate_store_size.py file.h5ad [cells] [block_cells]

Reads the first `cells` rows (default 60,000) of X (CSR) in blocks, one block
at a time, and sizes each layout from the block's own numbers: nothing is
written. General-purpose compressors are measured on the byte planes of each
block (zlib 6 stands in for zstd, lzma 6 for the best a general compressor does).
"""
import lzma
import sys
import zlib

import h5py
import numpy as np

path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
blk = int(sys.argv[3]) if len(sys.argv) > 3 else 8_192

f = h5py.File(path, "r")
X = f["X"]
n_rows = len(X["indptr"]) - 1
n_cols = int(X.attrs["shape"][1])
rows = min(want, n_rows)
ip = X["indptr"][: rows + 1].astype(np.int64)

tot = dict(nnz=0, cur=0, u8_u16=0, u8_dlt8=0, bp=0, zl=0, xz=0, ent_bits=0.0)
stat = dict(v_gt255=0, v_gt65535=0, v_nonint=0, v_one=0, unsorted=0, esc=0)
hv = np.zeros(1 << 16, np.int64)   # histogram of values (clipped)
hd = np.zeros(1 << 16, np.int64)   # histogram of index gaps (clipped)


def groups_bits(a: np.ndarray) -> int:
    """Bit-packed size in bits of `a` in groups of 128: each group at its own
    widest width, plus an 8-bit width header."""
    pad = (-len(a)) % 128
    g = np.concatenate([a, np.zeros(pad, a.dtype)]).reshape(-1, 128)
    m = g.max(axis=1).astype(np.uint64)
    w = np.where(m == 0, 0, np.floor(np.log2(np.maximum(m, 1))).astype(np.int64) + 1)
    return int(w.sum() * 128 + 8 * len(w))


for s in range(0, rows, blk):
    e = min(s + blk, rows)
    a, b = ip[s], ip[e]
    val = X["data"][a:b]
    idx = X["indices"][a:b].astype(np.int64)
    n = b - a
    if n == 0:
        continue
    starts = (ip[s:e] - a)[ip[s + 1 : e + 1] > ip[s:e]]
    gap = idx.copy()
    gap[1:] -= idx[:-1] + 1
    gap[starts] = idx[starts]
    if (gap < 0).any():
        stat["unsorted"] += int((gap < 0).sum())
        gap = np.abs(gap)
    nonint = int((val != np.floor(val)).sum())
    mx = float(val.max())
    stat["v_nonint"] += nonint
    stat["v_gt255"] += int((val > 255).sum())
    stat["v_gt65535"] += int((val > 65535).sum())
    stat["v_one"] += int((val == 1).sum())
    vw = 1 if (mx <= 255 and not nonint) else 2 if (mx <= 65535 and not nonint) else 4
    tot["nnz"] += n
    tot["cur"] += 8 * n
    tot["u8_u16"] += (2 + vw) * n                                  # u16 gap + narrowest value
    esc = int((gap >= 255).sum())
    stat["esc"] += esc
    tot["u8_dlt8"] += (1 + vw) * n + 2 * esc                       # u8 gap, 255 then u16 on escape
    vi = np.minimum(val, 65535).astype(np.int64) if not nonint else np.zeros(0, np.int64)
    gi = np.minimum(gap, 65535)
    hv += np.bincount(vi, minlength=1 << 16)[: 1 << 16] if len(vi) else 0
    hd += np.bincount(gi, minlength=1 << 16)[: 1 << 16]
    if not nonint:
        tot["bp"] += (groups_bits(gap.astype(np.uint64)) + groups_bits(val.astype(np.uint64))) / 8
    g16 = gi.astype(np.uint16)
    planes = [(g16 & 0xFF).astype(np.uint8).tobytes(), (g16 >> 8).astype(np.uint8).tobytes(),
              np.minimum(val, 255).astype(np.uint8).tobytes()]
    if vw > 1:
        planes.append((np.minimum(val, 65535).astype(np.uint16) >> 8).astype(np.uint8).tobytes())
    tot["zl"] += sum(len(zlib.compress(p, 6)) for p in planes)
    tot["xz"] += sum(len(lzma.compress(p, preset=6)) for p in planes)


def entropy(h):
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


n = tot["nnz"]
H = entropy(hd) + (entropy(hv) if hv.sum() else float("nan"))
print(f"{path}")
print(f"rows read {rows:,} of {n_rows:,}; genes {n_cols:,}; stored values {n:,} "
      f"({n / rows:,.0f} per cell, density {n / rows / n_cols:.1%})")
print(f"values: share equal to 1 {stat['v_one'] / n:.1%}; above 255: {stat['v_gt255']:,}; "
      f"above 65535: {stat['v_gt65535']:,}; not whole numbers: {stat['v_nonint']:,}")
print(f"gaps needing an escape (>=255): {stat['esc'] / n:.2%}; "
      f"rows out of order: {stat['unsorted']:,}")
rows_out = [
    ("now: f32 value + u32 gene id", tot["cur"] / n),
    ("u8 value + u16 gene id (or u16 gap)   [compressed.rs]", tot["u8_u16"] / n),
    ("u8 value + u8 gap, escape to u16", tot["u8_dlt8"] / n),
    ("bit-packed gap + value, groups of 128 (BPCells-like)", tot["bp"] / n),
    ("u8/u16 planes + zlib 6 per block (zstd-like)", tot["zl"] / n),
    ("u8/u16 planes + lzma 6 per block (ceiling)", tot["xz"] / n),
    ("zeroth-order entropy of gap + value (bound)", H / 8),
]
print(f"\n{'layout':58s} {'bytes/value':>11s} {'smaller by':>11s}")
for name, b in rows_out:
    print(f"{name:58s} {b:11.2f} {8 / b:10.1f}x")
