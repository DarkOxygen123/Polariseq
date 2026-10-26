#!/bin/bash -l
# The r4 cluster run fetal4062k_polariseq-spill_t4 (benchmarks/figures/source_data/r4_cluster/job_scripts) again with Polariseq
# 0.2.0: the same harness, arm, threads, budget, timeout, gate, cache, memory request and partition; the
# build of commit $COMMIT (env py-polariseq-020) and its own copy of benchmarks/.
#SBATCH --job-name=v020-fetal4062k-polariseq-spill-t4
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=170G
#SBATCH --time=3-00:00:00
#SBATCH --output=v020-fetal4062k_polariseq-spill_t4-%j.log
B=${B:?set B to your working folder (REPRODUCE.md)}; E=$B/envs
COMMIT=$(cat $E/py-polariseq-020/COMMIT); SRC=$B/dev/build-$COMMIT
export TMPDIR=$B/runs/v020/tmp HDF5_USE_FILE_LOCKING=FALSE; mkdir -p $TMPDIR
unset CONDA_PREFIX CONDA_DEFAULT_ENV
echo "node: $(hostname)  start: $(date)  load: $(cut -d' ' -f1-3 /proc/loadavg)  commit: $COMMIT"
free -g | head -2
cd $SRC
$E/py-polariseq-020/bin/python -u benchmarks/run_telemetry.py --datasets $B/data/paper/fetal_data/fetal4062k.h5ad \
    --arms polariseq-spill --reps 1 --threads 4 --ram-gb 100 --timeout 259000 --gate off --cache warm \
    --results $B/runs/v020/fetal4062k_polariseq-spill_t4
echo "finished: $(date)"
