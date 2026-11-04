#!/bin/bash
# B0 and B1a (frozen recipe) on the E1 cells via the repo's own script. Skip-if-exists.
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=$PWD/runs OMP_NUM_THREADS=1
jobs_list=()
for s in 0 1 2 3 4; do for L in random alt; do for n in 4 16; do jobs_list+=("$s $L $n"); done; done; done
printf '%s\n' "${jobs_list[@]}" | xargs -P ${P:-3} -L 1 bash -c 'python scripts/cond_cell.py --seed $0 --n_forget 4 --n_neg $2 --layout $1 --arms B0_frozen,R_hopB_bsup_neg_AMdist >> explore/E1/logs/baselines.log 2>&1'
