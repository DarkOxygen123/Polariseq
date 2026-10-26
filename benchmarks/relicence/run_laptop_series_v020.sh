#!/bin/bash
# The Polariseq laptop series with Polariseq 0.2.0 : every stage of the
# r4 laptop campaign and the budget sweep at 800,000 cells (benchmarks/relicence/laptop_series_v020.py).
# Run it after a restart (swap empty), with the charger in, the lid open and nothing else running; about
# 3 to 4 hours plus a few minutes to build, and it can be stopped and started again (finished runs are kept). From the repository, in
# Terminal:
#
#     bash benchmarks/relicence/run_laptop_series_v020.sh
#
# Records: benchmarks/results/telemetry_v020 and benchmarks/results/v020_budget; log in
# benchmarks/results/logs/laptop_series_v020_*.log.
set -u
cd "$(dirname "$0")/../.."
free_gb=$(df -g . | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 15 ]; then
    echo "only ${free_gb} GB free on this disk; the series needs about 15 GB (records and one run's block store)"
    exit 1
fi
# Uncommitted changes to tracked files stop the series (the records name the commit they ran), except
# changes under docs/, which no run reads.
code_changes=$(git status --porcelain --untracked-files=no | grep -v ' docs/')
if [ -n "$code_changes" ]; then
    echo "uncommitted changes outside docs/; commit or stash them first:"; echo "$code_changes"; exit 1
fi
# Build the library from this commit first, so the series runs exactly this code (a few minutes).
echo "$(date '+%F %T')  building Polariseq (release) from $(git rev-parse --short=8 HEAD)"
.venv/bin/maturin develop --release -q || { echo "build failed"; exit 1; }
.venv/bin/python -c "
import polariseq
from polariseq import planning
assert polariseq.__version__.startswith('0.2.0') and hasattr(planning, 'IN_MEMORY_VIEW_CHUNKS_MAX'), polariseq.__version__
print('build is polariseq', polariseq.__version__)" || { echo "this is not the 0.2.0 build of 2026-10-15"; exit 1; }
caffeinate -dimsu -w $$ &
echo "$(date '+%F %T')  start; ${free_gb} GB free; $(sysctl -n vm.swapusage); polariseq" \
     "$(.venv/bin/python -c 'import polariseq; print(polariseq.__version__)') at $(git rev-parse --short=8 HEAD)"
.venv/bin/python benchmarks/relicence/laptop_series_v020.py --allow-dirty
status=$?
echo "$(date '+%F %T')  laptop_series_v020.py exited ${status}"
exit $status
