#!/bin/bash
# Memory of the neighbours step against the number of cells, on random slices of the Tahoe-100M embedding
# (Polariseq 0.1.1, the code of the 100M run). One run per size, smallest first, each under the memory
# monitor with NN-Descent's phase trace. Results: $SD/results/neighbors-memory/n<cells>/.
#SBATCH -J sd-nbr-memory
#SBATCH -p cpu,standard
#SBATCH -c 16
#SBATCH --mem=170G
#SBATCH -t 06:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs
EMB=$SD/work/tahoe-all/project/runs/0+96_more_febf85ff/results/pca/X_pca.c367708a.npy
export HDF5_USE_FILE_LOCKING=FALSE TMPDIR=/tmp POLARISEQ_TRACE=1
for N in ${SIZES:-5000000 10000000 20000000 40000000}; do
    R=$SD/results/neighbors-memory/n$N; W=$SD/work/neighbors-memory/n$N
    mkdir -p $R $W
    echo "== $N cells, node $(hostname), $(date '+%F %T')"
    $E/py-score/bin/python $SD/node/monitor.py --out $R --name run -- \
        $E/py-polariseq-011/bin/python $REPO/benchmarks/relicence/neighbors_memory.py $EMB $N --project $W/project \
        --threads 16 --ram-gb 150 > $R/trace.log 2>&1
    echo "   status $? $(date '+%F %T')"
    rm -rf $W
done
