#!/bin/bash
# The fetal atlas by one clustering at five resolutions, with markers, from the final 0.2.0 node run
# (benchmarks/relicence/v020_fetal_resolutions.py). Results: runs/v020/derived/fetal.
#SBATCH -J v020-fetal-res
#SBATCH -p cpu,standard
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}
export POLARISEQ_THREADS=8 TMPDIR=/tmp HDF5_USE_FILE_LOCKING=FALSE
cd /tmp && $B/envs/py-polariseq-020-final/bin/python -u $B/runs/v020/v020_fetal_resolutions.py $B/runs/v020/derived/fetal
