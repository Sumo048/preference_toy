#!/bin/bash
# Like sweep.sh but npm 4 only (the fit does not depend on npm once every bad track is named; see eval_n16.py)
cd "$(dirname "$0")"
tag=$1; shift
for s in ${SEEDS:-0 1 2 3 4}; do for L in random alt; do
  nm=$([ $L = random ] && echo k4 || echo alt)_s${s}_n4
  [ -f results/graft_${tag}_${nm}.json ] || pgrep -f "seed $s --layout $L --npm 4 .*--tag $tag" >/dev/null || echo "$s $L 4"
done; done | xargs -P ${P:-4} -L 1 bash -c 'OMP_NUM_THREADS=1 python graft.py --seed $0 --layout $1 --npm $2 --tag '"$tag $*"' > logs/'"$tag"'_$1_s$0_n$2.log 2>&1'
