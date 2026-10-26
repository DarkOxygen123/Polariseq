#!/usr/bin/env bash
# Does the configured memory budget actually hold?
#
# Every Polariseq run in the campaign so far used ram_gb=6. The budget is the
# distinctive claim of the design, and it has never been varied, so nothing in
# the record shows whether a user who asks for 2 GB gets 2 GB. This sweeps the
# budget at two sizes and records what the run actually demanded.
#
# Three outcomes are all informative. The peak tracks the budget, which is the
# claim. The peak ignores the budget, which is a defect worth knowing about
# before publication. Or the run fails below some budget, which locates the
# floor and is the more useful number for a reader deciding whether their
# machine can do this at all.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -u +%Y%m%dT%H%M%S)
LOG="benchmarks/results/logs/budget_sweep_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"
say () { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

BUDGETS="${BUDGETS:-1 2 3 4 6 8}"
REPS="${REPS:-2}"
say "=== budget sweep, ram_gb in [$BUDGETS] ==="
n=0; for b in $BUDGETS; do n=$((n+2*REPS)); done
echo "EXPECTED_RUNS $n" | tee -a "$LOG"

for size in brain400k brain800k; do
  DATA="benchmarks/results/validation_data/${size}.h5ad"
  [ -f "$DATA" ] || { say "missing $DATA, skipping"; continue; }
  for b in $BUDGETS; do
    say "--- $size / polariseq-spill / ram_gb=${b} x${REPS} ---"
    python3 benchmarks/run_telemetry.py \
      --datasets "$DATA" --arms polariseq-spill --reps "$REPS" \
      --threads 4 --ram-gb "$b" --no-umap --cache warm \
      --gate calibrate --gate-timeout 420 --timeout 2400 \
      >>"$LOG" 2>&1 || say "    (failure recorded, continuing)"
  done
done

say "=== sweep done ==="
