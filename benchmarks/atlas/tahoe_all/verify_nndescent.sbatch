#!/bin/bash
# NN-Descent with the forest built a group of trees at a time (env py-polariseq-020dev) against 0.1.1
# (env py-polariseq-011), on one node: the neighbours step on 20M and 40M random cells of the Tahoe-100M
# embedding with each version (memory, time, and a SHA-256 of every result array, which must be equal), then
# on all 95.6M cells with the new version, whose arrays must equal those the 0.1.1 run wrote. Results:
# $SD/results/nndescent-verify/<size>-<version>/ and $SD/results/nndescent-verify/hashes.txt.
#SBATCH -J sd-nn-verify
#SBATCH -p cpu,standard
#SBATCH -c 16
#SBATCH --mem=170G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs
RUN=$SD/work/tahoe-all/project/runs/0+96_more_febf85ff
EMB=$RUN/results/pca/X_pca.c367708a.npy
OUT=$SD/results/nndescent-verify; mkdir -p $OUT
export HDF5_USE_FILE_LOCKING=FALSE TMPDIR=/tmp POLARISEQ_TRACE=1
echo "node $(hostname) $(date '+%F %T'); new = $(cat $E/py-polariseq-020dev/COMMIT)" | tee -a $OUT/hashes.txt

hashes() {  # SHA-256 of each result array of the neighbours step, by name without the random suffix
    $E/py-score/bin/python - "$1" <<'PY'
import hashlib, sys, pathlib
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.npy")):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    print(f.name.split(".")[0], h.hexdigest())
PY
}

step() {  # step SIZE VERSION ENV
    local N=$1 V=$2 PY=$3/bin/python R=$OUT/$1-$2 W=$SD/work/nndescent-verify/$1-$2
    mkdir -p $R $W
    echo "== $N cells, $V, $(date '+%F %T')"
    $E/py-score/bin/python $SD/node/monitor.py --out $R --name run -- \
        $PY $REPO/benchmarks/relicence/neighbors_memory.py $EMB $N --project $W/project --threads 16 --ram-gb 150 \
        > $R/trace.log 2>&1
    echo "   status $? $(date '+%F %T')"
    hashes $(ls -d $W/project/runs/*/results/neighbors | head -1) | sed "s/^/$N $V /" | tee -a $OUT/hashes.txt
    rm -rf $W
}

for N in 20000000 40000000; do
    step $N old $E/py-polariseq-011
    step $N new $E/py-polariseq-020dev
done
step 95596109 new $E/py-polariseq-020dev
echo "the 0.1.1 run's own arrays:" | tee -a $OUT/hashes.txt
hashes $RUN/results/neighbors | sed "s/^/95596109 run-011 /" | tee -a $OUT/hashes.txt
