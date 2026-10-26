#!/bin/bash
# Packed embeddings of the final 0.2.0's compute-node runs and the two modes compared
# (benchmarks/relicence/v020_pack_and_compare.py). Results: runs/v020/derived.
#SBATCH -J v020-pack
#SBATCH -p cpu,standard
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${B:?set B to your working folder (REPRODUCE.md)}
$B/envs/py-score/bin/python -u $B/runs/v020/v020_pack_and_compare.py $B/runs/v020/derived
