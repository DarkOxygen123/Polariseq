#!/bin/bash
# Run on the cluster's login node (nohup): submit the v020 runs one at a time; each excludes
# the nodes your running jobs occupy, and the next is submitted once the previous has started.
#   nohup bash launch.sh LOGFILE name1 name2 ... > /dev/null 2>&1 &
LOG=$1; shift
B=${B:?set B to your working folder (REPRODUCE.md)}
R=$B/runs/v020
for name in "$@"; do
  nodes=$(squeue -h -u $USER -t R -o %N | tr '\n' ',' | sed 's/,$//')
  ex="$nodes"
  id=$(cd $R && sbatch --parsable ${ex:+--exclude=$ex} "${REPO:?set REPO to your clone of the reproduce branch}/benchmarks/relicence/cluster/v020_$name.sbatch")
  echo "$(date '+%F %T') submitted $name as $id excluding $ex" >> $LOG
  while true; do
    st=$(squeue -h -j $id -o %T 2>/dev/null)
    [ "$st" = "RUNNING" ] && break
    [ -z "$st" ] && break
    sleep 60
  done
  echo "$(date '+%F %T') $name ($id) started on $(squeue -h -j $id -o %N)" >> $LOG
done
echo "$(date '+%F %T') all submitted" >> $LOG
