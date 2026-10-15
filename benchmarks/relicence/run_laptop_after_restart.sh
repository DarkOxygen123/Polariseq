#!/bin/bash
# The laptop runs of Polariseq 0.2.0 at 1.3M and 4.06M cells, after a restart (swap empty), with the
# charger in, the lid open and nothing else running. About 2 to 2.5 hours. Run from the repository:
#
#     bash benchmarks/relicence/run_laptop_after_restart.sh
#
# 1. The timed laptop runs at 1.3M and 4.06M cells (benchmarks/relicence/laptop_v020.py: the r4 laptop campaign's harness and
#    settings, with UMAP): brain 1.3M out of core x3 (and the in-memory arm, which this laptop declines) and
#    fetal 4.06M out of core x3, with 0.2.0 (.venv) and, alternating repeat by repeat in the same session,
#    0.1.1 (.venv-011) as the control, so that a difference between them is the code and not the machine.
#    Records: benchmarks/results/telemetry_v020_umap and telemetry_v011_umap_control.
# 2. The full analyses with 0.2.0 (examples/06_large_run.py, 4 threads, 6 GB, UMAP, every figure, the
#    per-cell table and the run summary): benchmarks/results/v020_laptop/{fetal4062k,brain1306k}.
set -u
cd "$(dirname "$0")/../.."
OUT=benchmarks/results/v020_laptop; mkdir -p $OUT
LOG=$OUT/after_restart_$(date +%Y%m%dT%H%M%S).log
caffeinate -dimsu -w $$ &
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
say "start; $(sysctl -n vm.swapusage); polariseq 0.2.0 = $(.venv/bin/python -c 'import polariseq; print(polariseq.__version__)'), control = $(.venv-011/bin/python -c 'import polariseq; print(polariseq.__version__)')"
# the records made before the restart, with swap almost full, are kept aside, not mixed in
if [ -d benchmarks/results/telemetry_v020_umap ]; then
    mv benchmarks/results/telemetry_v020_umap benchmarks/results/telemetry_v020_umap_before_restart_$(date +%Y%m%d%H%M)
fi
for rep in 1 2 3; do
    for env in .venv-011 .venv; do
        res=$([ $env = .venv ] && echo telemetry_v020_umap || echo telemetry_v011_umap_control)
        say "timed runs, repeat $rep, $env -> $res"
        $env/bin/python benchmarks/relicence/laptop_v020.py --allow-dirty --results $res --reps-upto $rep >> "$LOG" 2>&1 \
            || say "  the timed runs exited $? (see the log)"
    done
done
for ds in fetal_data/fetal4062k validation_data/brain1306k; do
    name=$(basename $ds)
    say "full analysis $name (0.2.0)"
    rm -rf $OUT/$name $OUT/${name}_project
    .venv/bin/python -u examples/06_large_run.py benchmarks/results/$ds.h5ad --out $OUT/$name \
        --project $OUT/${name}_project --ram-gb 6 --threads 4 --umap >> "$LOG" 2>&1 \
        || say "  $name exited $? (see the log)"
    rm -rf $OUT/${name}_project
    grep -h "cells in the file" $OUT/$name/run_summary.md 2>/dev/null | tee -a "$LOG"
done
say "all done; log $LOG"
