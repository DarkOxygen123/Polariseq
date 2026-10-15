#!/bin/bash
# One tool on one dataset: convert, run, score, clean up. Submit with sbatch (REPRODUCE.md, section 5):
#
#   sbatch -J <tool>-<dataset> --exclusive -c 16 --mem=170G -t 4-00:00:00 -o %x-%j.out \
#          $REPO/benchmarks/atlas/run_tool.sh TOOL DATASET INPUT
#
# --exclusive: on some clusters, when one of your jobs ends on a node, your other jobs on that node are
# killed too; exclusive jobs also keep the timings clean.
#
# TOOL     tahoe-h5ad | truth | polariseq | scarf | bpcells | compare (score each run, compare the tools)
# DATASET  a short name, e.g. scTab-22M; results go to $B/showdown/results/DATASET/
# INPUT    folder of .h5ad files, a pattern, or comma-separated files
#
# Environment: B (working folder), REPO (this clone), the environments in $B/envs (REPRODUCE.md);
# PS_ENV (default $B/envs/py-polariseq) is the Polariseq environment and PS_BUILD a label for it.
# KEEP_STORE=1 keeps the tool's store in $B/showdown/work after the run (default: deleted after a
# successful run, its size recorded first). LABEL_COL (default cell_type) names the published labels;
# BATCH_COLS (default dataset_id,donor_id). PS_MODE (default auto) is Polariseq's mode.
set -u
TOOL=$1; DATASET=$2; INPUT=$3; DBL=${4:-none}
B=${B:?set B to your working folder (REPRODUCE.md)}
SD=$B/showdown
E=$B/envs
N=${REPO:?set REPO to your clone of the reproduce branch}/benchmarks/atlas
THREADS=${SLURM_CPUS_PER_TASK:-16}
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-174080} / 1024 ))
RAM_GB=${RAM_GB:-$(( MEM_GB - 10 ))}
BATCH_COLS=${BATCH_COLS:-dataset_id,donor_id}
LABEL_COL=${LABEL_COL:-cell_type}
PS_MODE=${PS_MODE:-auto}
PS_ENV=${PS_ENV:-$E/py-polariseq}
export PS_BUILD=${PS_BUILD:-"$($PS_ENV/bin/python -c 'import polariseq; print(polariseq.__version__)' 2>/dev/null)"}
export LD_LIBRARY_PATH=$E/r/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=$E/tmp
export HDF5_USE_FILE_LOCKING=FALSE   # several jobs read the same files on Lustre
MON="$E/py-score/bin/python $N/monitor.py"
SCORE="$E/py-score/bin/python $N/score.py"
NAME=$TOOL; [ "$DBL" = remove ] && NAME=$TOOL-remove
R=$SD/results/$DATASET/$NAME
W=$SD/work/$DATASET/$NAME      # flag and remove runs of a tool never share a store
FLAGS=$SD/results/$DATASET/scrublet-ref/flags
mkdir -p "$R" "$W"
case $TOOL in polariseq|scarf|bpcells)
  # A rerun (after a failure, or on an hm node) starts clean: the earlier attempt's records move to
  # attempt-N/ (kept for the sheet's notes) and its store is deleted, so nothing half-computed is reused.
  if [ -f "$R/run.resources.csv" ] || [ -f "$R/convert.resources.csv" ]; then
    n=1; while [ -d "$R/attempt-$n" ]; do n=$((n+1)); done
    mkdir -p "$R/attempt-$n"
    if [ "${RESUME_RUN:-0}" = 1 ]; then
      # RESUME_RUN=1: only the failed analysis is repeated; the finished conversion and its store stay
      # (the conversion record keeps the first attempt's time and memory).
      find "$R" -maxdepth 1 -type f -name 'run.*' -exec mv {} "$R/attempt-$n/" \;
      [ -d "$R/art" ] && mv "$R/art" "$R/attempt-$n/"
      echo "[run_tool] RESUME_RUN: analysis records of the earlier attempt moved to $R/attempt-$n; conversion kept"
    else
      find "$R" -maxdepth 1 -type f ! -name jobs.txt -exec mv {} "$R/attempt-$n/" \;
      [ -d "$R/art" ] && mv "$R/art" "$R/attempt-$n/"
      rm -rf "$W"; mkdir -p "$W"
      echo "[run_tool] earlier attempt moved to $R/attempt-$n"
    fi
  fi
  ;;
esac
echo "[run_tool] $TOOL on $DATASET ($DBL) node $(hostname) threads $THREADS ram $RAM_GB GB start $(date '+%F %T')"
echo "$TOOL $DATASET $INPUT $DBL job ${SLURM_JOB_ID:-none} node $(hostname) $(date '+%F %T')" >> "$R/jobs.txt"

no_remove() {
  if [ "$DBL" = remove ]; then echo "[run_tool] $TOOL has no doublet detection: no removal run"; exit 3; fi
}

case $TOOL in
tahoe-h5ad)
  # INPUT is the folder of downloaded parquet files; shards go to its sibling h5ad/
  $MON --out "$R" --name convert -- $E/py-score/bin/python $N/tahoe_to_h5ad.py --parquet "$INPUT" \
       --out "$(dirname "$INPUT")/h5ad" --workers "$THREADS" --delete-parquet
  ;;
truth)
  $MON --out "$R" --name truth -- $E/py-score/bin/python $N/build_truth.py --input "$INPUT" \
       --out $SD/results/$DATASET/truth.npz --label-col "$LABEL_COL" --workers "$THREADS"
  ;;
polariseq)
  PY=$PS_ENV/bin/python
  echo "polariseq build: $PS_BUILD ($PS_ENV)" >> "$R/jobs.txt"
  $MON --out "$R" --name convert --watch "$W/project" -- $PY $N/drv_polariseq.py convert --input "$INPUT" --mode "$PS_MODE" \
       --project "$W/project" --out "$R/art" --threads "$THREADS" --ram-gb "$RAM_GB" || exit 1
  $MON --out "$R" --name run --watch "$W/project" -- $PY $N/drv_polariseq.py run --input "$INPUT" --mode "$PS_MODE" \
       --project "$W/project" --out "$R/art" --threads "$THREADS" --ram-gb "$RAM_GB" --doublets "$DBL" \
       --batch-cols "$BATCH_COLS" || exit 1
  [ -f $SD/results/$DATASET/truth.npz ] && $SCORE run --truth $SD/results/$DATASET/truth.npz --art "$R/art" --out "$R/score.json"
  ;;
scarf)
  no_remove
  PY=$E/py-scarf/bin/python
  $MON --out "$R" --name convert --watch "$W/data.zarr" -- $PY $N/drv_scarf.py convert --input "$INPUT" \
       --zarr "$W/data.zarr" || exit 1
  $MON --out "$R" --name run --watch "$W/data.zarr" -- $PY $N/drv_scarf.py run --zarr "$W/data.zarr" \
       --out "$R/art" --threads "$THREADS" || exit 1
  [ -f $SD/results/$DATASET/truth.npz ] && $SCORE run --truth $SD/results/$DATASET/truth.npz --art "$R/art" \
       --out "$R/score.json"
  ;;
bpcells)
  no_remove
  if [ "${RESUME_RUN:-0}" != 1 ]; then
    $MON --out "$R" --name convert --watch "$W/matrix" -- $E/r/bin/Rscript $N/drv_bpcells.R convert \
         --input "$INPUT" --dir "$W/matrix" || exit 1
  fi
  $MON --out "$R" --name run --watch "$W/matrix" -- $E/r/bin/Rscript $N/drv_bpcells.R run --dir "$W/matrix" \
       --out "$R/art" --threads "$THREADS" || exit 1
  [ -f $SD/results/$DATASET/truth.npz ] && $SCORE run --truth $SD/results/$DATASET/truth.npz --art "$R/art" \
       --out "$R/score.json"
  ;;
compare)
  # after the tools: score any finished run not yet scored, then compare the tools
  RD=$SD/results/$DATASET
  ARTS=""
  for t in polariseq scarf bpcells polariseq-remove scarf-remove bpcells-remove; do
    if [ -f $RD/$t/art/labels.npy ] || [ -f $RD/$t/art/labels.i32 ]; then
      if [ ! -f $RD/$t/score.json ]; then
        FL=""; case $t in scarf*|bpcells*) [ -f $FLAGS.npz ] && FL="--flags $FLAGS.npz";; esac
        $SCORE run --truth $RD/truth.npz --art $RD/$t/art $FL --out $RD/$t/score.json > /dev/null
      fi
      case $t in *-remove) ;; *) ARTS="$ARTS --art $t=$RD/$t/art";; esac
    fi
  done
  RF=""; [ -f $FLAGS.npz ] && RF="--ref-flags $FLAGS.npz"
  $SCORE compare $ARTS $RF --out $RD/compare.json
  ;;
*) echo "unknown tool $TOOL"; exit 2 ;;
esac
STATUS=$?
if [ "${KEEP_STORE:-0}" != 1 ] && [ "$TOOL" != truth ] && [ "$TOOL" != tahoe-h5ad ] && [ "$TOOL" != compare ]; then
  rm -rf "$W"
fi
echo "[run_tool] done $(date '+%F %T') status $STATUS"
exit $STATUS
