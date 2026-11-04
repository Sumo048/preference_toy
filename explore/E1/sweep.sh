#!/bin/bash
# Usage: P=4 SEEDS="0 1 2" ./sweep.sh <tag> [graft.py options...]   -- runs k4 + alt, npm 4 and 16
cd "$(dirname "$0")"
tag=$1; shift
for s in ${SEEDS:-0 1 2}; do for L in random alt; do for n in 4 16; do
  nm=$([ $L = random ] && echo k4 || echo alt)_s${s}_n${n}
  [ -f results/graft_${tag}_${nm}.json ] || echo "$s $L $n"
done; done; done | xargs -P ${P:-4} -L 1 bash -c 'OMP_NUM_THREADS=1 python graft.py --seed $0 --layout $1 --npm $2 --tag '"$tag $*"' > logs/'"$tag"'_$1_s$0_n$2.log 2>&1'
