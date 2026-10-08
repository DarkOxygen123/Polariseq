"""The three-way Tier 1 comparison: Polariseq, scanpy, Seurat.

Reports medians with spread, and keeps the two things that make the numbers
interpretable next to them — cluster counts (which differ because resolution
is not comparable across tools) and cluster recovery (which says whether the
same populations came back).
"""
from __future__ import annotations
import glob, json, statistics as st, collections

ORDER = ["polariseq-ram", "polariseq-spill", "scanpy-default", "scanpy-tuned",
         "scanpy-tuned-seed1", "seurat-default", "seurat-tuned", "seurat-bpcells"]
DS_ORDER = ["pbmc3k", "pbmc10k", "neuron10k", "pbmc20k"]


def load():
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob("benchmarks/results/telemetry/*.json"):
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("status") != "ok": continue
        k = (r["dataset"]["label"], r["arm"])
        by[k]["wall"].append(r["wall_s"])
        by[k]["rss"].append(r["peaks"]["parent_rss"] / 1e9)
        by[k]["uss"].append(r["peaks"]["child_uss"] / 1e9)
        by[k]["k"].append(r.get("outputs", {}).get("n_clusters"))
    return by


def spread(v):
    return (max(v) - min(v)) / 2 if len(v) > 1 else 0.0


def main():
    by = load()
    datasets = [d for d in DS_ORDER if any(k[0] == d for k in by)]
    print("TIER 1 — THREE-WAY COMPARISON   (median +/- half-range, n per cell)\n")
    print(f"{'dataset':<11}{'arm':<21}{'wall s':>15}{'peak RSS GB':>14}{'clusters':>10}{'n':>4}")
    print("-" * 76)
    for ds in datasets:
        base = None
        for arm in ORDER:
            v = by.get((ds, arm))
            if not v: continue
            w, r = v["wall"], v["rss"]
            ks = [x for x in v["k"] if x is not None]
            mw = st.median(w)
            if arm == "polariseq-ram": base = mw
            spd = f"  ({base/mw:.2f}x)" if base and arm != "polariseq-ram" and mw else ""
            print(f"{ds:<11}{arm:<21}{mw:8.1f}+/-{spread(w):4.1f}{st.median(r):10.2f}"
                  f"{'':4}{st.median(ks) if ks else 0:8.0f}{len(w):4d}")
        print()
    print("Speed is reported as Polariseq's median over the arm's median, so a")
    print("value below 1.0 would mean the arm is faster than Polariseq.\n")
    print("Cluster counts differ partly because `resolution` is not comparable")
    print("across tools: Seurat's Louvain at 0.8 and its Leiden at 1.0 give")
    print("different granularity on the same data. Cluster *recovery*")
    print("(cluster_recovery.py) is the comparable quantity.")


if __name__ == "__main__":
    main()
