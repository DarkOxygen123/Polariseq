"""Each tool's clusters of the fetal atlas against the authors' cell types, from the packed embeddings.

    python benchmarks/r4_score_tools.py EMBEDDINGS_DIR OUT_JSON

EMBEDDINGS_DIR holds the files benchmarks/figures/pack_embeddings.py wrote, one per reported cluster run
(fetal4062k__<arm>__<timestamp>.npz, with each cell's own cluster and published cell type). Scored with
the function of the sweeps (r3_scarf_ablation.scores: ARI, NMI, purity, types recovered), over the cells
with a published type.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_scarf_ablation import scores  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {"source": str(src), "measure": "r3_scarf_ablation.scores, Leiden/Scarf clusters vs Main_cluster_name",
       "runs": {}}
for f in sorted(glob.glob(str(src / "fetal4062k__*.npz"))):
    z = np.load(f, allow_pickle=False)
    ct = z["cell_type"].astype(np.int64)
    ok = ct >= 0
    s = scores(z["cluster"].astype(np.int64)[ok], ct[ok])
    s["cells"], s["cells_with_type"] = int(len(ct)), int(ok.sum())
    res["runs"][Path(f).stem] = s
    print(Path(f).stem, s, flush=True)
out.write_text(json.dumps(res, indent=1))
