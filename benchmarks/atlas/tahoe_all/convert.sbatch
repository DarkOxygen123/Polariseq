#!/bin/bash
# Full Tahoe-100M, step 2: the downloaded parquet files of the 12 other plates become .h5ad shards 14... and
# Tahoe-A's 14 shards (0-13) are hard-linked beside them, so tahoe-all/h5ad holds all 95,596,109 cells.
# Submit once tahoe_fetch.py has finished without failures (REPRODUCE.md).
#SBATCH -J sd-tahoe-all-convert
#SBATCH -p cpu,standard
#SBATCH -c 16
#SBATCH --mem=170G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}; SD=$B/showdown; E=$B/envs; N=${REPO:?set REPO to your clone of the reproduce branch}/benchmarks/atlas
D=$B/data/tahoe/tahoe-all
export HDF5_USE_FILE_LOCKING=FALSE TMPDIR=/tmp
mkdir -p $D/h5ad $SD/results/tahoe-all/convert
$E/py-score/bin/python $N/monitor.py --out $SD/results/tahoe-all/convert --name convert -- \
    $E/py-score/bin/python $N/tahoe_to_h5ad.py --parquet $D/parquet --out $D/h5ad --start-index 14 --workers 16 --delete-parquet
for i in $(seq 0 13); do ln -f $B/data/tahoe/tahoe-A/h5ad/$i.h5ad $D/h5ad/$i.h5ad; done
$E/py-score/bin/python - <<PY
import glob, h5py
n = 0
files = sorted(glob.glob("$D/h5ad/*.h5ad"))
for f in files:
    with h5py.File(f, "r") as h:
        n += int(h["X"].attrs["shape"][0])
print(len(files), "shards,", n, "cells")
assert len(files) == 97 and n == 95596109, "unexpected shard or cell count"
PY
echo "[tahoe-all] converted and checked $(date '+%F %T')"
