#!/bin/bash
# Full Tahoe-100M, step 4: scores from the per-cell table of step 3 (ARI, NMI, purity, cell lines recovered, UMAP
# scores, clusters held mostly by one plate) and the cell-line x cluster heatmap. Reads cells.tsv.gz in chunks.
#SBATCH -J sd-tahoe-all-score
#SBATCH -p cpu,standard
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs; RUN=${RUN:-tahoe-all}; R=$SD/results/$RUN/polariseq
export MPLBACKEND=Agg
echo "tahoe-all score job ${SLURM_JOB_ID:-none} node $(hostname) $(date '+%F %T')" >> $SD/results/$RUN/jobs.txt
$E/py-score/bin/python $REPO/benchmarks/atlas/tahoe_all/score_cells.py --cells $R/figures/cells.tsv.gz --out $R/scores
STATUS=$?
echo "[tahoe-all] score status $STATUS $(date '+%F %T')"
exit $STATUS
