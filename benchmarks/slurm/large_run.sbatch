#!/bin/bash -l
# One large dataset end to end with examples/06_large_run.py (the 22.2-million-cell scTab atlas:
# 100 .h5ad files, out of core, 8 threads, a 100 GB budget). Edit the settings, then:
#     sbatch benchmarks/slurm/large_run.sbatch
#SBATCH --job-name=polariseq-large-run
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=large-run-%j.log

# ---- settings ---------------------------------------------------------------------------------
REPO=$HOME/polariseq                       # this repository
ENV=$HOME/polariseq-env                    # a Python environment with polariseq installed
INPUT=$REPO/benchmarks/results/scTab/h5ad  # a folder of .h5ad files, a pattern or one file
OUT=$REPO/benchmarks/results/scTab_22M     # report, figures and the per-cell table
PROJECT=$TMPDIR/polariseq-scTab            # block store: put it on fast local or parallel storage
RAM_GB=100
THREADS=8
# -----------------------------------------------------------------------------------------------

source $ENV/bin/activate
echo "node: $(hostname)  start: $(date)"
python -u $REPO/examples/06_large_run.py "$INPUT" --out "$OUT" --project "$PROJECT" \
    --ram-gb $RAM_GB --threads $THREADS --mode out-of-core \
    --keep-obs soma_joinid,cell_type,donor_id,dataset_id,assay,tissue_general,disease --color-by cell_type
echo "finished: $(date)"
