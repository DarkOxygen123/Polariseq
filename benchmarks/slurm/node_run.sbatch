#!/bin/bash -l
# One benchmark run on one compute node: one tool ("arm"), one dataset, one repeat.
# Edit the settings, then:  sbatch benchmarks/slurm/node_run.sbatch
# Run each job alone on its node (other users' jobs are fine); two of your own runs on one node
# compete for its memory bandwidth and disk.
#SBATCH --job-name=polariseq-node-run
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=node-run-%j.log

# ---- settings ---------------------------------------------------------------------------------
REPO=$HOME/polariseq                       # this repository, branch "reproduce"
ENV=$HOME/polariseq-env                    # a Python environment with polariseq and the tools installed
DATASET=$REPO/benchmarks/results/fetal_data/fetal4062k.h5ad   # or .../validation_data/brain1306k.h5ad
ARM=polariseq-spill                        # polariseq-spill (out of core), polariseq-ram (in memory),
                                           # scanpy-tuned-int64, scarf
THREADS=4
RAM_GB=100                                 # Polariseq's budget; other tools ignore it
RESULTS=$REPO/benchmarks/results/node/$(basename $DATASET .h5ad)_${ARM}_t${THREADS}
# -----------------------------------------------------------------------------------------------

source $ENV/bin/activate
export TMPDIR=${TMPDIR:-$REPO/benchmarks/results/tmp}; mkdir -p $TMPDIR
echo "node: $(hostname)  start: $(date)  load: $(cut -d' ' -f1-3 /proc/loadavg)"
cd $REPO
python -u benchmarks/run_telemetry.py --datasets $DATASET --arms $ARM --reps 1 \
    --threads $THREADS --ram-gb $RAM_GB --timeout 259000 --gate off --cache warm --results $RESULTS
echo "finished: $(date)"
