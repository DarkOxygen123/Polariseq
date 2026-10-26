#!/bin/bash
# Leiden of 0.1.1 (env py-polariseq-011) and of 0.2.0 (env py-polariseq-020dev) on the Tahoe-100M run's graph,
# on one node, alternating, seeds 0 and 1; then their agreement (ARI between runs and with the 50 cell lines)
# from the run's per-cell table. Results: $SD/results/leiden-100m/.
#SBATCH -J sd-leiden-100m
#SBATCH -p cpu,standard
#SBATCH -c 16
#SBATCH --mem=170G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs
G=$SD/work/tahoe-all/project/runs/0+96_more_febf85ff/results/neighbors
OUT=$SD/results/leiden-100m; mkdir -p $OUT
SRC=$SD/tahoe_all
echo "node $(hostname) $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg)"
for seed in 0 1; do
  for v in 011 020dev; do
    R=$OUT/$v-seed$seed; mkdir -p $R
    echo "== $v seed $seed $(date '+%F %T')"
    $E/py-score/bin/python $SD/node/monitor.py --out $R --name run -- \
        $E/py-polariseq-$v/bin/python $SRC/leiden_param.py $G --seed $seed --out $R/labels.npy > $R/log.txt 2>&1
    echo "   status $? $(tail -1 $R/log.txt)"
  done
done
$E/py-score/bin/python - <<'PY'
import json, glob, numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
OUT = "$B/showdown/results/leiden-100m"
runs = {p.split("/")[-1]: np.load(p + "/labels.npy") for p in sorted(glob.glob(OUT + "/*-seed*"))}
cells = pd.read_csv("$B/showdown/results/tahoe-all/polariseq/figures/cells.tsv.gz",
                    sep="\t", usecols=["cell_line_id"])
_, line = np.unique(cells.cell_line_id.to_numpy().astype(str), return_inverse=True)
out = {}
for k, v in runs.items():
    m = json.load(open(f"{OUT}/{k}/labels.json")); mon = json.load(open(f"{OUT}/{k}/run.monitor.json"))
    out[k] = {**m, "peak_anon_gb": mon["peak_anon_gb"], "ari_cell_lines": round(ari(line, v), 4)}
keys = sorted(runs)
out["pairs"] = {f"{a} vs {b}": round(ari(runs[a], runs[b]), 4) for i, a in enumerate(keys) for b in keys[i + 1:]}
json.dump(out, open(OUT + "/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
