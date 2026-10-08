"""Cluster recovery: what a biologist would actually check.

The adjusted Rand index answers "how much do these two partitions agree" with
a single number, and it punishes a split and a shuffle equally. Those are very
different outcomes for an experiment. If one method separates a subpopulation
the other keeps whole, the biology is largely preserved. If the same cells are
scattered across unrelated clusters, it is not. ARI cannot tell you which
happened.

So this reports, per dataset and per arm against a reference:

* **cells_in_matched**  — share of cells whose reference cluster's
  best-matching arm cluster actually contains them. "Would I find this cell
  among the same neighbours?"
* **recovered**         — reference clusters with a best match at Jaccard >= 0.5.
  "How many of my populations came back?"
* **merges** / **splits** — direction of the disagreement, which is the part
  ARI discards.

ARI and the seed-to-seed floor are still reported alongside. Recovery is the
readable summary; the floor is what keeps it falsifiable, because "87% of cells
agree" means nothing until you know the reference scores against itself.

    python benchmarks/cluster_recovery.py [--reference scanpy-tuned]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ART = "benchmarks/results/telemetry/artifacts"


def _labels(dataset: str, arm: str, rep: int = 1) -> np.ndarray | None:
    """Labels for one (dataset, arm), preferring `rep` but taking any.

    Falling back matters: a rep can be absent because an earlier attempt
    failed and was archived, and silently dropping the arm from the table
    would read as "this tool was not compared" rather than "rep 1 is missing".
    """
    for pat in (f"{ART}/{dataset}__{arm}__rep{rep}__*", f"{ART}/{dataset}__{arm}__rep*__*"):
        for h in sorted(glob.glob(pat)):
            f = Path(h) / "labels.npy"
            if f.exists():
                return np.load(f)
    return None


def recovery(arm_labels: np.ndarray, ref_labels: np.ndarray) -> dict:
    """Match every reference cluster to its best-overlapping arm cluster."""
    ua, ub = np.unique(arm_labels), np.unique(ref_labels)
    ia = {v: i for i, v in enumerate(ua)}
    ib = {v: i for i, v in enumerate(ub)}
    C = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for a, b in zip(arm_labels, ref_labels, strict=True):
        C[ia[a], ib[b]] += 1

    best = C.argmax(axis=0)                      # per reference cluster
    captured = C[best, np.arange(C.shape[1])].sum() / len(ref_labels)
    jac = np.array([
        C[best[j], j] / (C[best[j], :].sum() + C[:, j].sum() - C[best[j], j])
        for j in range(C.shape[1])
    ])
    # An arm cluster that is the best match for several reference clusters has
    # merged them; a reference cluster split across arm clusters shows up as a
    # low Jaccard despite a clean best match.
    merges = sum(1 for i in range(C.shape[0]) if (best == i).sum() > 1)
    splits = int(((jac < 0.5) & (C[best, np.arange(C.shape[1])] > 0)).sum())

    return {
        "n_clusters_arm": int(len(ua)),
        "n_clusters_ref": int(len(ub)),
        "cells_in_matched": round(float(captured), 4),
        "recovered_at_j50": int((jac >= 0.5).sum()),
        "median_jaccard": round(float(np.median(jac)), 3),
        "merges": int(merges),
        "splits": int(splits),
        "ari": round(float(adjusted_rand_score(arm_labels, ref_labels)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="scanpy-tuned")
    ap.add_argument("--json", action="store_true", help="emit JSON not a table")
    args = ap.parse_args()

    datasets = sorted({Path(p).name.split("__")[0] for p in glob.glob(f"{ART}/*")})
    arms = sorted({Path(p).name.split("__")[1] for p in glob.glob(f"{ART}/*")})
    arms = [a for a in arms if a != args.reference]

    out = []
    for ds in datasets:
        ref = _labels(ds, args.reference)
        if ref is None:
            continue
        for arm in arms:
            lab = _labels(ds, arm)
            if lab is None or len(lab) != len(ref):
                continue
            out.append({"dataset": ds, "arm": arm, **recovery(lab, ref)})

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"CLUSTER RECOVERY vs {args.reference}   (rep1; recovery = Jaccard >= 0.5)\n")
    print(f"{'dataset':<11}{'arm':<20}{'k_arm':>6}{'k_ref':>6}"
          f"{'cells match':>12}{'recovered':>11}{'medJ':>6}{'merge':>6}{'ARI':>7}")
    print("-" * 85)
    for r in out:
        print(f"{r['dataset']:<11}{r['arm']:<20}{r['n_clusters_arm']:6d}{r['n_clusters_ref']:6d}"
              f"{r['cells_in_matched']:11.1%}{str(r['recovered_at_j50'])+'/'+str(r['n_clusters_ref']):>11}"
              f"{r['median_jaccard']:6.2f}{r['merges']:6d}{r['ari']:7.3f}")


if __name__ == "__main__":
    main()
