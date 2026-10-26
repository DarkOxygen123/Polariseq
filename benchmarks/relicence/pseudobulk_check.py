#!/usr/bin/env python
"""Polariseq's pseudobulk test against the edgeR and limma reference of pseudobulk_reference.R.

    python pseudobulk_check.py REF_DIR [--out report.json]

Each case's samples x genes counts go through `tl.compare_conditions` with one "cell" per sample
(min_cells=1). Reports, over all cases, whether the same genes pass the filter and the largest
relative differences of log fold change, t, p-value, adjusted p-value and the
prior degrees of freedom, and the seconds spent.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time

import numpy as np


def rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    d = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    d[both_inf] = 0
    return float(np.nanmax(d)) if d.size else 0.0


def main() -> int:
    import pandas as pd
    import polariseq as ps
    import scipy.sparse as sp
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--out")
    a = ap.parse_args()
    worst = {k: 0.0 for k in ("logFC", "t", "p", "adjp", "df_prior")}
    where = {}
    same_genes, cases, seconds = 0, 0, 0.0
    for f in sorted(glob.glob(f"{a.ref}/case_*.json")):
        c = json.load(open(f))
        y = np.asarray(c["counts"], dtype=np.float64)
        cond = np.asarray(c["condition"]).astype(str)
        donor = np.asarray(c["donor"]).astype(str)
        g = y.shape[1]
        ad = ps.AnnData(X=sp.csr_matrix(y.astype(np.float32)),
                        obs={"cell_type": np.repeat("all", len(cond)), "condition": cond,
                             "donor": np.char.add("d", donor)},
                        var_names=[f"g{j}" for j in range(g)])
        t0 = time.perf_counter()
        got = ps.tl.compare_conditions(ad, "condition", "0", groupby="cell_type", sample_key="donor",
                                       min_cells=1)
        seconds += time.perf_counter() - t0
        cases += 1
        info = ad.uns["compare_conditions"]["groups"].iloc[0]
        keep = [f"g{j}" for j in c["keep"]]
        genes_ok = True
        for comp in c.get("comparisons", []):
            sub = got[got["condition"].astype(str) == str(np.ravel(comp["condition"])[0])].set_index("gene")
            if set(sub.index) != set(keep):
                genes_ok = False
                continue
            sub = sub.loc[keep]
            for k, col in (("logFC", "log2_fold_change"), ("t", "t"), ("p", "p_value"),
                           ("adjp", "adj_p_value")):
                r = rel(sub[col].to_numpy(), comp[k])
                if r > worst[k]:
                    worst[k], where[k] = r, f
        r = rel([info["df_prior"]], c["df_prior"])
        if r > worst["df_prior"]:
            worst["df_prior"], where["df_prior"] = r, f
        same_genes += genes_ok
    rep = {"cases": cases, "cases_with_the_same_genes": same_genes, "max_relative_difference": worst,
           "worst_case": where, "seconds_total": round(seconds, 2)}
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
