"""Comparing conditions on Kang et al. (2018): validation against independent answers.

    .venv/bin/python benchmarks/r3_kang.py            # Polariseq, writes results
    Rscript benchmarks/r3_kang_limma.R                 # edgeR + limma on the same sums
    <scrublet venv>/python benchmarks/r3_kang.py --reference   # reference Scrublet

Data: GSE96583 batch 2 (Kang, H. M. et al. Nat. Biotechnol. 36, 89-94, 2018),
built into two .h5ad files by examples/05_comparing_conditions.py (run it
once first). Results go to benchmarks/results/r3/kang/.

Checks, each against an independent answer:

- doublets: Polariseq's Scrublet against the reference implementation
  (scrublet 0.2.3) and against the authors' genotype calls (demuxlet);
- the pseudobulk test: against edgeR 4.10 and limma 3.68 on the same sums;
- the analysis: in memory against out of core (identical), and the two
  conditions analyzed together against each analyzed on its own;
- the biology: interferon-stimulated genes up in every cell type.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/data/kang_2018"
RUNS = {"control": str(DATA / "control.h5ad"), "IFN-beta": str(DATA / "ifn_beta.h5ad")}
OUT = ROOT / "benchmarks/results/r3/kang"

#: Interferon-stimulated genes: the canonical core of the type I response.
ISG = ["ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "ISG20", "LY6E", "IFITM3", "OAS1",
       "RSAD2", "IFI44L"]


def reference() -> None:
    """Reference Scrublet on each capture (run in an environment with scrublet)."""
    import anndata as ad
    import scrublet as scr

    out = {}
    for label, path in RUNS.items():
        a = ad.read_h5ad(path)
        t0 = time.perf_counter()
        s = scr.Scrublet(a.X.tocsr(), expected_doublet_rate=0.05, sim_doublet_ratio=2.0,
                         random_state=0)
        scores, _ = s.scrub_doublets(verbose=False, n_prin_comps=30)
        out[label] = {"seconds": time.perf_counter() - t0,
                      "threshold": float(getattr(s, "threshold_", np.nan))}
        np.save(OUT / f"scrublet_reference_{label}.npy", scores)
    (OUT / "scrublet_reference.json").write_text(json.dumps(out, indent=1))


def analyze(adata, ps, markers) -> None:
    """The example's analysis, from quality control to named clusters."""
    import importlib

    sys.path.insert(0, str(ROOT / "examples"))
    ex = importlib.import_module("05_comparing_conditions")
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=200)
    ps.pp.filter_genes(adata, min_cells=3)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=2000)
    ps.pp.pca(adata, n_components=30)
    rep = "X_pca"
    if len(set(np.asarray(adata.obs["condition"]).tolist())) > 1:
        ps.pp.harmony_integrate(adata, batch_key="condition")
        rep = "X_pca_harmony"
    ps.pp.neighbors(adata, n_neighbors=15, use_rep=rep)
    ps.tl.leiden(adata, resolution=0.6)
    ps.tl.umap(adata)
    ex.name_clusters(adata, markers)


def main() -> None:
    import importlib

    import polariseq as ps
    from sklearn.metrics import roc_auc_score

    sys.path.insert(0, str(ROOT / "examples"))
    markers = importlib.import_module("05_comparing_conditions").MARKERS
    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_budget(ram_gb=8, threads=4)
    ps.project(OUT / "project")
    res: dict = {}

    # Doublets, in memory and out of core, against the genotype calls.
    t = {}
    for mode in ("memory", "out-of-core"):
        a = ps.read_h5ad(RUNS, batch_key="condition", mode=mode, name=f"kang_{mode}")
        a.X  # the matrix (or the on-disk copy) first, so the timing is the step's
        t0 = time.perf_counter()
        ps.pp.scrublet(a, batch_key="condition")
        t[mode] = time.perf_counter() - t0
        if mode == "memory":
            mem = a
        else:
            ooc = a
    same = np.array_equal(np.asarray(mem.obs["doublet_score"]), np.asarray(ooc.obs["doublet_score"]))
    dmx = np.asarray(mem.obs["demuxlet"]).astype(str)
    cond = np.asarray(mem.obs["condition"]).astype(str)
    score = np.asarray(mem.obs["doublet_score"], dtype=float)
    ok = dmx != "ambs"
    res["doublets"] = {
        "seconds_memory": t["memory"], "seconds_out_of_core": t["out-of-core"],
        "memory_equals_out_of_core": bool(same),
        "auroc": float(roc_auc_score(dmx[ok] == "doublet", score[ok])),
        "auroc_by_capture": {c: float(roc_auc_score(dmx[ok & (cond == c)] == "doublet",
                                                    score[ok & (cond == c)])) for c in RUNS},
        "called": int(np.sum(mem.obs["predicted_doublet"])),
        "genotype_doublets": int((dmx == "doublet").sum()),
        "genotype_doublets_called": int(np.sum(np.asarray(mem.obs["predicted_doublet"]) & (dmx == "doublet"))),
        "batches": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else None)
                        for kk, vv in v.items() if kk != "doublet_scores_sim"}
                    for k, v in mem.uns["scrublet"]["batches"].items()},
    }
    np.save(OUT / "scrublet_polariseq.npy", score)
    pd.DataFrame({"obs": list(mem.obs_names), "condition": cond, "demuxlet": dmx,
                  "score": score}).to_csv(OUT / "doublets.csv", index=False)

    # The joint analysis, in memory and out of core.
    tables = {}
    for mode, a in (("memory", mem), ("out-of-core", ooc)):
        keep = ~np.asarray(a.obs["predicted_doublet"]) & (np.asarray(a.obs["demuxlet"]) == "singlet")
        a = a[keep]
        t0 = time.perf_counter()
        analyze(a, ps, markers)
        t_analysis = time.perf_counter() - t0
        t0 = time.perf_counter()
        tab = ps.tl.compare_conditions(a, "condition", "control", groupby="cell_type",
                                       sample_key="donor")
        tables[mode] = (a, tab, t_analysis, time.perf_counter() - t0)
    a, joint, t_an, t_cmp = tables["memory"]
    b, joint_ooc = tables["out-of-core"][0], tables["out-of-core"][1]
    res["analysis"] = {
        "cells": int(a.n_obs), "seconds_analysis": t_an, "seconds_compare": t_cmp,
        "memory_equals_out_of_core_types": bool(np.array_equal(np.asarray(a.obs["cell_type"]),
                                                               np.asarray(b.obs["cell_type"]))),
        "memory_equals_out_of_core_table": bool(joint.equals(joint_ooc)),
        "cell_type_agreement_with_authors": float(np.mean(
            np.asarray(a.obs["cell_type"]).astype(str) == np.asarray(a.obs["published_type"]).astype(str))),
        "groups": a.uns["compare_conditions"]["groups"].to_dict("records"),
    }
    joint.to_csv(OUT / "joint.csv", index=False)
    pd.DataFrame({"x": a.obsm["X_umap"][:, 0], "y": a.obsm["X_umap"][:, 1],
                  "condition": np.asarray(a.obs["condition"]).astype(str),
                  "cell_type": np.asarray(a.obs["cell_type"]).astype(str)}).to_csv(
        OUT / "umap.csv", index=False)
    genes = ["ISG15", "IFI6", "CXCL10"]
    vals = np.asarray(a.X[:, [list(a.var_names).index(g) for g in genes]].to_scipy().toarray())
    pd.DataFrame(vals, columns=genes).assign(
        condition=np.asarray(a.obs["condition"]).astype(str),
        cell_type=np.asarray(a.obs["cell_type"]).astype(str)).to_csv(OUT / "violins.csv", index=False)
    pb = ps.get.pseudobulk(a, "donor", groupby="cell_type", condition_key="condition", min_cells=10)
    pd.DataFrame(np.asarray(pb.X), index=pb.obs_names, columns=list(pb.var_names)).to_csv(
        OUT / "pseudobulk_counts.csv")
    pd.DataFrame({k: np.asarray(v).astype(str) if k != "n_cells" else np.asarray(v)
                  for k, v in pb.obs.items()}, index=pb.obs_names).to_csv(OUT / "pseudobulk_meta.csv")

    # Cells compared cell by cell: what the replicates guard against.
    cells = ps.tl.compare_conditions(a, "condition", "control", groupby="cell_type",
                                     key_added="compare_cells")
    sig = lambda t: t[t["adj_p_value"] < 0.05].groupby("cell_type", observed=True).size()  # noqa: E731
    res["significant_genes"] = {"replicates": sig(joint).to_dict(), "cells": sig(cells).to_dict()}

    # Each condition analyzed on its own, its clusters named the same way.
    names = {}
    for label, path in RUNS.items():
        c = ps.read_h5ad({label: path}, batch_key="condition", mode="memory", name=f"alone_{label}")
        c = c[np.asarray(c.obs["demuxlet"]) == "singlet"]
        ps.pp.scrublet(c)
        c = c[~np.asarray(c.obs["predicted_doublet"])]
        analyze(c, ps, markers)
        names.update(dict(zip(c.obs_names, np.asarray(c.obs["cell_type"]).astype(str), strict=True)))
    sep = a.copy()
    sep.obs["cell_type"] = pd.Categorical([names.get(n, "unassigned") for n in sep.obs_names])
    sep = sep[np.asarray(sep.obs["cell_type"] != "unassigned")]
    separate = ps.tl.compare_conditions(sep, "condition", "control", groupby="cell_type",
                                        sample_key="donor")
    separate.to_csv(OUT / "separate.csv", index=False)
    same_type = np.mean([names.get(n) == t for n, t in
                         zip(a.obs_names, np.asarray(a.obs["cell_type"]).astype(str), strict=True)
                         if n in names])
    m = joint.merge(separate, on=["cell_type", "gene"], suffixes=("_joint", "_separate"))
    res["joint_vs_separate"] = {
        "same_cell_type": float(same_type),
        "log2fc_r": {str(t): float(np.corrcoef(d["log2_fold_change_joint"],
                                               d["log2_fold_change_separate"])[0, 1])
                     for t, d in m.groupby("cell_type", observed=True) if len(d) > 2},
    }

    # The biology: interferon-stimulated genes up in every cell type.
    up = joint[(joint["adj_p_value"] < 0.05) & (joint["log2_fold_change"] > 0)]
    types = sorted(set(joint["cell_type"].astype(str)))
    res["isg_up_in_types"] = {g: sorted(up[up["gene"] == g]["cell_type"].astype(str)) for g in ISG}
    res["isg_up_in_every_type"] = [g for g in ISG if set(res["isg_up_in_types"][g]) == set(types)]
    cx = joint[joint["gene"] == "CXCL10"].set_index("cell_type")["log2_fold_change"]
    res["cxcl10_log2fc"] = {str(k): float(v) for k, v in cx.items()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "analysis"}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    reference() if "--reference" in sys.argv else main()
