#!/bin/bash
# E4 queue: 7 settings x npm {1,4,16,64} x seeds 5-9 = 140 cells, 4 workers, 1 thread each.
# One unit = one cell (B0, B1a, method; skip-if-exists per method), so the queue is resumable:
# rerun this script after any interruption and finished JSONs are skipped.
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=${TOYPREF_RUNS:-$PWD/runs}
mkdir -p explore/E4/logs
for s in 5 6 7 8 9; do for n in 4 16 1 64; do for st in k1 k2 k4 contig alt pairs b3p1; do
  echo "$st $s $n"; done; done; done |
  xargs -P 4 -L 1 bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 3000 python explore/E4/e4.py --setting $0 --seed $1 --npm $2 >> explore/E4/logs/$0_s$1_n$2.log 2>&1; echo "unit $0 s$1 n$2 exit $?"'
echo QUEUE_DONE
