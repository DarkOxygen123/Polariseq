#!/bin/bash
# Memory of the neighbours step against the number of threads, at 20M cells of the Tahoe-100M embedding.
# N, THREADS, PYENV (default py-polariseq-011), OUTNAME (default neighbors-threads) and RAMGB (the budget,
# default 150) from the environment.
#SBATCH -J sd-nbr-threads
#SBATCH -p cpu,standard
#SBATCH -c 16
#SBATCH --mem=170G
#SBATCH -t 06:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs
EMB=$SD/work/tahoe-all/project/runs/0+96_more_febf85ff/results/pca/X_pca.c367708a.npy
export HDF5_USE_FILE_LOCKING=FALSE TMPDIR=/tmp POLARISEQ_TRACE=1
N=${N:-20000000}; PYENV=${PYENV:-py-polariseq-011}; OUTNAME=${OUTNAME:-neighbors-threads}; RAMGB=${RAMGB:-150}
for T in ${THREADS:-4 8 12}; do
    R=$SD/results/$OUTNAME/t$T; W=$SD/work/$OUTNAME/t$T
    mkdir -p $R $W
    echo "== $N cells, $T threads, $PYENV, node $(hostname), $(date '+%F %T'), load $(cut -d' ' -f1 /proc/loadavg)"
    $E/py-score/bin/python $SD/node/monitor.py --out $R --name run -- \
        $E/$PYENV/bin/python $REPO/benchmarks/relicence/neighbors_memory.py $EMB $N --project $W/project \
        --threads $T --ram-gb $RAMGB > $R/trace.log 2>&1
    echo "   status $? $(date '+%F %T')"
    rm -rf $W
done
