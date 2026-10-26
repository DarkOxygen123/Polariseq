#!/bin/bash
# Leiden's seed-to-seed agreement on the graphs of the final 0.2.0's node runs (benchmarks/relicence/v020_seed_spread.py).
#SBATCH -J v020-seeds
#SBATCH -p cpu,standard
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}
export POLARISEQ_THREADS=8 TMPDIR=/tmp
cd /tmp && $B/envs/py-polariseq-020-final/bin/python -u $B/runs/v020/v020_seed_spread.py $B/runs/v020/derived/seed_spread_v020.json
