#!/bin/bash
# With the final 0.2.0: the fetal analysis handed to Scanpy's PAGA, and Paris on
# the same graph (v020_downstream_paris.py). Scanpy's environment (scanpy 1.12.4, anndata 0.13.4, NumPy 2.2.6, as
# the node runs) with an overlay on the path that holds only the 0.2.0 package (a link) and igraph 0.11.9, the
# version the Scanpy node runs recorded (envs/overlay-paga-020), so Polariseq and Scanpy share one process.
#SBATCH -J v020-downstream
#SBATCH -p cpu,standard
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}
export POLARISEQ_THREADS=4 TMPDIR=/tmp HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONPATH=$B/envs/overlay-paga-020
SPILL=$B/runs/v020/downstream_spill
mkdir -p $SPILL
cd /tmp && $B/envs/py-scanpy/bin/python -u $B/runs/v020/v020_downstream_paris.py $B/runs/v020/derived/downstream $SPILL
