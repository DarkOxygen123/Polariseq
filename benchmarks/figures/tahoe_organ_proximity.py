"""Do Tahoe-100M cell lines from one organ of origin lie together on the map?

    python benchmarks/figures/tahoe_organ_proximity.py RASTERS_DIR CELL_LINES_CSV OUT_CSV

For each organ with two lines or more, the mean distance between its lines' centres on the UMAP (the median
position of each line's cells, from the raster of every cell) against the same mean for 5,000 random sets of as
many lines (seed 0). RASTERS_DIR is raster_tahoe100m.py's output folder; CELL_LINES_CSV is
benchmarks/atlas/tahoe_all/cell_lines.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embedding_raster as ER  # noqa: E402

r = ER.load(Path(sys.argv[1]) / "tahoe100M__polariseq-spill__cell_lines.npz")
meta = pd.read_csv(sys.argv[2]).drop_duplicates("cell_line_id").set_index("cell_line_id")
c = pd.DataFrame(r["centres"], columns=["group", "cells", "x", "y"])
c["line"] = [str(r["group_names"][int(g)]) for g in c.group]
c["organ"] = c.line.map(meta.organ)
xy = c[["x", "y"]].to_numpy()
dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))


def mean_pair(idx) -> float:
    sub = dist[np.ix_(idx, idx)]
    return float(sub[np.triu_indices(len(idx), 1)].mean())


rng = np.random.default_rng(0)
everyone = mean_pair(np.arange(len(c)))
rows = []
for organ, grp in c.groupby("organ"):
    if len(grp) < 2:
        continue
    m = mean_pair(grp.index.to_numpy())
    null = np.array([mean_pair(rng.choice(len(c), len(grp), replace=False)) for _ in range(5000)])
    rows.append({"organ": organ, "lines": len(grp), "mean_distance": round(m, 3),
                 "relative_to_all_lines": round(m / everyone, 3),
                 "p_random_as_close": round(float((np.sum(null <= m) + 1) / (len(null) + 1)), 4)})
out = pd.DataFrame(rows).sort_values("p_random_as_close")
out.to_csv(SRC / "organ_proximity.csv", index=False)
print(out.to_string(index=False))
