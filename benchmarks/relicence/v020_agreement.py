"""Agreement of the tools on the compute node, with Polariseq's final 0.2.0 runs.

    .venv/bin/python benchmarks/relicence/v020_agreement.py

Writes benchmarks/figures/source_data/cluster_agreement_v020.csv in the layout of cluster_agreement_r4.csv:
- clusters, adjusted Rand index: every pair of Polariseq (in memory and out of core, 0.2.0, packed by
  v020_pack_and_compare.py), Scanpy and Scarf (packed by figures/pack_embeddings.py), over the cells all of them
  kept, matched by name;
- selected genes, Jaccard: Polariseq's genes of 0.2.0 are identical to the r4 runs' on both datasets (checked
  here against the r4 lists), so the r4 overlaps with Scanpy and Scarf are the ones of 0.2.0;
- principal components: Polariseq's two modes (modes_cmp_v020.json).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "benchmarks/figures/source_data"
DER = ROOT / "benchmarks/results/v020_cluster/derived"
PACKED = {
    "brain1306k": {"Polariseq out-of-core": DER / "cluster_embeddings_v020/brain1306k__polariseq-spill__20261014T125551.npz",
                   "Polariseq in-memory": DER / "cluster_embeddings_v020/brain1306k__polariseq-ram__20261014T125449.npz",
                   "Scanpy": ROOT / "benchmarks/results/cluster_embeddings/brain1306k__scanpy-tuned-int64__20261003T171918.npz",
                   "Scarf": ROOT / "benchmarks/results/cluster_embeddings/brain1306k__scarf__20261004T150332.npz"},
    "fetal4062k": {"Polariseq out-of-core": DER / "cluster_embeddings_v020/fetal4062k__polariseq-spill__20261014T125749.npz",
                   "Polariseq in-memory": DER / "cluster_embeddings_v020/fetal4062k__polariseq-ram__20261014T125637.npz",
                   "Scanpy": ROOT / "benchmarks/results/cluster_embeddings/fetal4062k__scanpy-tuned-int64__20261004T130538.npz",
                   "Scarf": ROOT / "benchmarks/results/cluster_embeddings/fetal4062k__scarf__20261004T125231.npz"}}
GENES_V020 = {"brain1306k": DER / "hvg_v020_brain1306k__polariseq-spill__rep1__20261014T125551.txt",
              "fetal4062k": DER / "hvg_v020_fetal4062k__polariseq-spill__rep1__20261014T125749.txt"}
GENES_R4 = {"brain1306k": SRC / "r4_sweeps/cluster_brain_ooc/genes_seurat_floor+scarf.txt",
            "fetal4062k": SRC / "r4_sweeps/cluster_tools_fetal4m/genes_Polariseq.txt"}


def main() -> None:
    r4 = pd.read_csv(SRC / "cluster_agreement_r4.csv")
    modes = json.loads((DER / "modes_cmp_v020.json").read_text())
    rows = []
    for ds, runs in PACKED.items():
        assert set(GENES_V020[ds].read_text().split()) == set(GENES_R4[ds].read_text().split()), ds
        z = {t: np.load(p, allow_pickle=False) for t, p in runs.items()}
        common = sorted(set.intersection(*(set(v["obs_names"].astype(str)) for v in z.values())))
        lab = {t: pd.Series(v["cluster"], index=v["obs_names"].astype(str)).reindex(common).to_numpy()
               for t, v in z.items()}
        for t, v in z.items():
            rows += [{"dataset": ds, "measure": "cells_kept", "a": t, "b": "", "value": len(v["obs_names"])},
                     {"dataset": ds, "measure": "clusters", "a": t, "b": "", "value": len(np.unique(v["cluster"]))},
                     {"dataset": ds, "measure": "genes_selected", "a": t, "b": "", "value": 2000}]
        g = r4[(r4.dataset == ds) & (r4.measure == "selected genes, Jaccard")]
        rows += g.to_dict("records")
        for a, b in itertools.combinations(z, 2):
            rows.append({"dataset": ds, "measure": "clusters, adjusted Rand index", "a": a, "b": b,
                         "value": float(ari(lab[a], lab[b]))})
        k = next(k for k in modes if k.startswith(ds) and "('polariseq-ram', 4)" in k)
        rows.append({"dataset": ds, "measure": "principal components, largest angle (deg)",
                     "a": "Polariseq out-of-core", "b": "Polariseq in-memory", "value": modes[k]["pc_max_angle_deg"]})
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "cluster_agreement_v020.csv", index=False)
    print(out[out.measure == "clusters, adjusted Rand index"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
