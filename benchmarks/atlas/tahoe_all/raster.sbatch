#!/bin/bash
# Full Tahoe-100M, step 5: UMAP rasters by cell line and cluster (benchmarks/figures/raster_tahoe100m.py)
# from the per-cell table of step 3.
#SBATCH -J sd-tahoe-all-raster
#SBATCH -p cpu,standard
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs; RUN=${RUN:-tahoe-all}; R=$SD/results/$RUN/polariseq
echo "tahoe-all raster job ${SLURM_JOB_ID:-none} node $(hostname) $(date '+%F %T')" >> $SD/results/$RUN/jobs.txt
cd "${REPO:?set REPO to your clone of the reproduce branch}"/benchmarks/figures
${PYR:-$E/py-polariseq/bin/python} -u raster_tahoe100m.py $R/figures/cells.tsv.gz $R/rasters $REPO/benchmarks/atlas/tahoe_all/cell_lines.csv
STATUS=$?
echo "[tahoe-all] raster status $STATUS $(date '+%F %T')"
exit $STATUS
