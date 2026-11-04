#!/bin/bash
# Resumable queue: units "cmd setting seed npm arg stage" from $1; 4 workers x 1 torch thread.
#   train: arg = mask name(s);  masks: arg = BUILDERS:NS (e.g. TB:2000,8000)
# Every unit writes its JSON atomically and skips finished outputs, so rerunning the queue resumes it.
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=${TOYPREF_RUNS:-$PWD/runs}
mkdir -p explore/TRKNG/logs
grep -v '^#' "$1" | xargs -P ${WORKERS:-4} -L 1 bash -c '
  if [ "$0" = masks ]; then a="--builders ${4%%:*} --N ${4##*:}"; else a="--mask $4"; fi
  t0=$(date +%s)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 3000 python explore/TRKNG/trkng.py $0 --setting $1 --seed $2 --npm $3 $a --stage $5 >> explore/TRKNG/logs/$1_s$2_n$3.log 2>&1
  echo "unit $0 $1 s$2 n$3 $4 exit $? $(( $(date +%s) - t0 ))s"'
echo QUEUE_DONE
