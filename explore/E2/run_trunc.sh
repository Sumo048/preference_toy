#!/bin/bash
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=$PWD/runs OMP_NUM_THREADS=1
for s in "$@"; do for L in random alt; do for n in 4 16; do echo "$s $L $n"; done; done; done | \
  xargs -P 2 -L 1 sh -c 'python explore/E2/b1a_trunc.py --seed $0 --layout $1 --npm $2 >> explore/E2/logs/trunc.log 2>&1; echo "trunc $0 $1 $2 rc=$?" >> explore/E2/logs/queue_done.log'
