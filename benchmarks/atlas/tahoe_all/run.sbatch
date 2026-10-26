#!/bin/bash
# Full Tahoe-100M, step 3: Polariseq, end to end, with the repository's one-call example (examples/06_large_run.py,
# the script of the paper's 22-million-cell analysis): QC, normalization, 2,000 genes, PCA, neighbours, Leiden,
# UMAP, then every figure, the per-cell table and the run summary. No doublet step. The reported run used Polariseq 0.1.1.
#SBATCH -J sd-tahoe-all-polariseq
#SBATCH -p cpu,standard
#SBATCH -c 16
#SBATCH --mem=170G
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs; N=${REPO:?set REPO to your clone of the reproduce branch}/benchmarks/atlas
D=$B/data/tahoe/tahoe-all; R=$SD/results/tahoe-all; W=$SD/work/tahoe-all
export HDF5_USE_FILE_LOCKING=FALSE TMPDIR=/tmp MPLBACKEND=Agg
mkdir -p $R/polariseq $W
echo "tahoe-all polariseq job ${SLURM_JOB_ID:-none} node $(hostname) $(date '+%F %T') polariseq $($E/py-polariseq/bin/python -c 'import polariseq; print(polariseq.__version__)')" >> $R/jobs.txt
$E/py-score/bin/python $N/monitor.py --out $R/polariseq --name run --watch $W/project -- \
    $E/py-polariseq/bin/python $REPO/examples/06_large_run.py "$D/h5ad" \
        --out $R/polariseq/figures --project $W/project --ram-gb 150 --threads 16 \
        --keep-obs plate,sample,cell_line_id,drug,sublibrary --color-by cell_line_id,plate
STATUS=$?
echo "[tahoe-all] polariseq status $STATUS $(date '+%F %T')"
exit $STATUS
