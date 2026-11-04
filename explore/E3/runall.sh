#!/bin/bash
# usage: runall.sh "<methods>" [seeds] [settings] [npms]   -- 4 workers, one cell per process
M="$1"; SEEDS="${2:-0 1 2}"; SETS="${3:-k4 alt}"; NPMS="${4:-4 16}"
cd "$(dirname "$0")/../.."
mkdir -p explore/E3/logs
for st in $SETS; do for s in $SEEDS; do for n in $NPMS; do echo "$st $s $n"; done; done; done |
  xargs -P 4 -L 1 bash -c 'OMP_NUM_THREADS=1 python explore/E3/e3.py --setting $0 --seed $1 --npm $2 --methods "'"$M"'" >> explore/E3/logs/$0_s$1_n$2.log 2>&1'
