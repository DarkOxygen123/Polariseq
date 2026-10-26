#!/bin/bash
# Paris on the graph that v020_downstream_paris.py saved (that run stopped there: the port takes float64 weights).
#SBATCH -J v020-paris
#SBATCH -p cpu,standard
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}
export POLARISEQ_THREADS=4 TMPDIR=/tmp
cd /tmp && $B/envs/py-polariseq-020-final/bin/python -u $B/runs/v020/v020_downstream_paris.py _paris $B/runs/v020/derived/downstream
cat $B/runs/v020/derived/downstream/fetal_paris.json
