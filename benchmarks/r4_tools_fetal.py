"""Each tool's analysis of the 4.06M-cell fetal atlas on the cluster node, measured one way.

    python benchmarks/r4_tools_fetal.py --out OUT --cells CELLS_CSV_GZ --profiles REFERENCE_PROFILES_NPY \
        --run "Polariseq=ARTIFACT_DIR" --run "Scanpy=PACKED.npz" --run "Scarf=PACKED.npz" ...

A run is a harness artifact folder (umap.npy, labels.npy, obs_names.txt, hvg.txt) or a file packed by
figures/pack_embeddings.py (umap, cluster, obs_names, cell_type). Every run is measured with the
functions of the parameter sweeps (r4_sweep.py), against the authors' cell types (Main_cluster_name, joined by cell name):

  clusters   r3_scarf_ablation.scores: ARI, NMI, purity, types recovered
  layout     r3_umap_fragmentation.islands (islands, types mostly in one island), pixel purity
             (figures/r3_layout_purity.py), and the global structure: Spearman correlation of the types'
             distances in the layout with the distances of their mean log-normalized profiles over all
             genes (REFERENCE_PROFILES_NPY, written by benchmarks/r4_sweep.py for the same atlas, rows in
             the order of the types' names)
  raster     figures/embedding_raster.py, by cell type, 600 px

The authors' own UMAP (Global_umap_1/2) is measured the same way, over the cells of the first run.
Writes OUT/tools_fetal4m.json, OUT/fetal4m__<tool>__cell_types.npz and OUT/genes_<tool>.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))


def load(src: str):
    p = Path(src)
    if p.is_dir():
        names = np.array([s for s in (p / "obs_names.txt").read_text().split("\n") if s])
        genes = (p / "hvg.txt").read_text().split() if (p / "hvg.txt").exists() else None
        return names, np.load(p / "umap.npy").astype(np.float32), np.load(p / "labels.npy").astype(np.int64), genes
    z = np.load(p, allow_pickle=False)
    return z["obs_names"].astype(str), z["umap"].astype(np.float32), z["cluster"].astype(np.int64), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--run", action="append", required=True, help="TOOL=PATH; PATH may also be TOOL's hvg.txt via TOOL:genes=PATH")
    ap.add_argument("--genes", action="append", default=[], help="TOOL=hvg.txt for a packed run")
    a = ap.parse_args()
    import embedding_raster as ER
    from r3_layout_purity import purity
    from r3_scarf_ablation import scores
    from r3_umap_fragmentation import islands
    from r4_sweep import global_vs_profiles

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(a.cells, usecols=["cell", "Main_cluster_name", "Global_umap_1", "Global_umap_2"],
                      dtype={"Main_cluster_name": "category"}).set_index("cell")
    types = list(tab.Main_cluster_name.cat.categories)
    ref = np.load(a.profiles)
    assert ref.shape[0] == len(types), (ref.shape, len(types))
    extra_genes = dict(g.split("=", 1) for g in a.genes)
    res = {"cells_csv": a.cells, "profiles": a.profiles, "types": len(types), "tools": {}}

    def measure(tool, names, xy, clusters, genes, source):
        t = tab.reindex(names)
        lab = t.Main_cluster_name.cat.codes.to_numpy().astype(np.int64)
        ok = lab >= 0
        r = {"source": source, "cells": int(len(names)), "cells_with_type": int(ok.sum())}
        if clusters is not None:
            r["clusters_vs_types"] = scores(clusters[ok], lab[ok])
        r["layout"] = islands(xy[ok], lab[ok])
        r["layout"]["pixel_label_purity"] = round(purity(xy[ok], lab[ok]), 4)
        r["layout"]["global_spearman_vs_type_profiles"] = round(global_vs_profiles(ref, xy[ok], lab[ok]), 4)
        ER.save(out / f"fetal4m__{tool}__cell_types.npz", ER.raster(xy[ok], lab[ok]), types,
                run_id=source, cells=int(ok.sum()))
        if genes:
            (out / f"genes_{tool}.txt").write_text("\n".join(genes))
            r["genes"] = len(genes)
        res["tools"][tool] = r
        (out / "tools_fetal4m.json").write_text(json.dumps(res, indent=1))
        print(tool, json.dumps(r), flush=True)
        return names

    first = None
    for spec in a.run:
        tool, path = spec.split("=", 1)
        names, xy, cl, genes = load(path)
        assert len(names) == len(xy) == len(cl) and np.isfinite(xy).all(), tool
        if genes is None and tool in extra_genes:
            genes = Path(extra_genes[tool]).read_text().split()
        measure(tool, names, xy, cl, genes, path)
        first = names if first is None else first
    t = tab.reindex(first)
    auth = t[["Global_umap_1", "Global_umap_2"]].to_numpy(dtype=np.float32)
    has = np.isfinite(auth).all(axis=1)
    measure("authors", first[has], auth[has], None, None, "Cao et al. 2020, Global_umap_1/2")


if __name__ == "__main__":
    main()
