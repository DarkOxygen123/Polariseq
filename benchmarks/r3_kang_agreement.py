"""Polariseq's pseudobulk test against edgeR + limma, on the same sums.

    .venv/bin/python benchmarks/r3_kang_agreement.py [folder]

Merges Polariseq's per-gene results (joint.csv, from r3_kang.py) with the reference's (limma.csv, from
r3_kang_limma.R) on cell type and gene, and writes limma_agreement.json: the number of genes tested and,
for each of the fold change, the t statistic, the p-value and the adjusted p-value, the largest
difference relative to the reference value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "benchmarks/results/r3/kang"


def main() -> dict:
    mine = pd.read_csv(K / "joint.csv")
    ref = pd.read_csv(K / "limma.csv")
    m = mine.merge(ref, on=["cell_type", "gene"], suffixes=("_ps", "_ref"))
    # (Polariseq's column, the reference's column) after the merge; the t statistic is in both tables
    pairs = {"log2_fold_change": ("log2_fold_change", "logFC"), "t": ("t_ps", "t_ref"),
             "p_value": ("p_value", "P.Value"), "adj_p_value": ("adj_p_value", "adj.P.Val")}
    out = {"rows": int(len(m))}
    for name, (a, b) in pairs.items():
        x, y = m[a].to_numpy(float), m[b].to_numpy(float)
        out[name] = float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300)))
    (K / "limma_agreement.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    print(main())
