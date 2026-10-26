"""Regenerate the ps.plan() decision grid.

`plan_grid.json` is the model's answers, not a measurement: for each
(dataset size, budget) pair it records what `ps.plan()` would choose before
any data is read — storage mode, rows per block, predicted peak, the
working-set floor and whether the plan fits. It can be set against what
the runs then actually did.
"""
from __future__ import annotations

import json
from pathlib import Path

import polariseq as ps

OUT = Path("benchmarks/figures/out/plan_grid.json")

# The sweep's sizes and budgets: 50K shows the cheap end where everything is
# 'ram', 300K the middle, 800K the size the budget sweep measured.
SIZES = {
    "brain50k": "benchmarks/results/validation_data/brain50k.h5ad",
    "brain300k": "benchmarks/results/validation_data/brain300k.h5ad",
    "brain800k": "benchmarks/results/validation_data/brain800k.h5ad",
}
BUDGETS = [1, 2, 4, 6, 8, 16]


def main() -> None:
    rows = []
    for label, path in SIZES.items():
        for gb in BUDGETS:
            ps.set_budget(ram_gb=gb, threads=4)
            p = ps.plan(path, verbose=False)
            rows.append({
                "cells": p.n_cells,
                "budget": gb,
                "storage": p.storage,
                "rows_per_block": p.rows_per_block,
                "predicted_peak_gb": round(p.peak_gb, 3),
                "min_budget_gb": round(p.min_budget_gb, 3),
                "fits": p.fits,
                "limiting_step": p.limiting_step,
                "warnings": list(p.warnings),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT} ({len(rows)} cells)")
    for r in rows:
        print(f"  {r['cells']:>9,} @ {r['budget']:>2} GB -> {r['storage']:<10} "
              f"rpb {r['rows_per_block']:>6,}  peak {r['predicted_peak_gb']:5.2f}  "
              f"floor {r['min_budget_gb']:5.2f}  fits {r['fits']}")


if __name__ == "__main__":
    main()
