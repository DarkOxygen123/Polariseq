#!/bin/bash
# The laptop runs of the final Polariseq 0.2.0 (graph steps read their inputs into memory when they fit) at
# 1.3M and 4.06M cells, after a restart (swap empty), with the charger in, the lid open and nothing else
# running. About 1.5 hours. Run from the repository, in Terminal:
#
#     bash benchmarks/relicence/run_laptop_final.sh
#
# 1. The timed laptop runs at 1.3M and 4.06M cells (benchmarks/relicence/laptop_v020.py: the r4 laptop campaign's harness and
#    settings, with UMAP): brain 1.3M in memory (which this laptop declines) and out of core x3, fetal 4.06M
#    out of core x3. Records: benchmarks/results/telemetry_v020_final_umap (the 2026-10-13 records of the
#    build before this change stay in telemetry_v020_umap).
# 2. The full analyses (examples/06_large_run.py, 4 threads, 6 GB, UMAP, every figure, the per-cell table and
#    the run summary, which says where each graph step read its inputs): benchmarks/results/v020_laptop_final.
set -u
cd "$(dirname "$0")/../.."
OUT=benchmarks/results/v020_laptop_final; mkdir -p "$OUT"
LOG=$OUT/run_$(date +%Y%m%dT%H%M%S).log
caffeinate -dimsu -w $$ &
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
say "start; $(sysctl -n vm.swapusage); polariseq $(.venv/bin/python -c 'import polariseq; print(polariseq.__version__)') at $(git rev-parse --short=8 HEAD)"
say "timed runs -> telemetry_v020_final_umap"
.venv/bin/python benchmarks/relicence/laptop_v020.py --allow-dirty --results telemetry_v020_final_umap >> "$LOG" 2>&1 \
    || say "  the timed runs exited $? (see the log)"
for ds in fetal_data/fetal4062k validation_data/brain1306k; do
    name=$(basename $ds)
    say "full analysis $name"
    rm -rf "${OUT:?}/${name:?}" "${OUT:?}/${name:?}_project"
    .venv/bin/python -u examples/06_large_run.py benchmarks/results/$ds.h5ad --out "$OUT/$name" \
        --project "$OUT/${name}_project" --ram-gb 6 --threads 4 --umap >> "$LOG" 2>&1 \
        || say "  $name exited $? (see the log)"
    rm -rf "${OUT:?}/${name:?}_project"
    grep -h -e "cells in the file" -e "Inputs of the graph steps" "$OUT/$name/run_summary.md" 2>/dev/null | tee -a "$LOG"
done
say "all done; log $LOG"
