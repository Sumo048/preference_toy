#!/bin/bash
# B0 and B1a on the E2 cells via the repo's own cond_cell.py (skip-if-exists).
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=$PWD/runs OMP_NUM_THREADS=1
jobs_list() {
for s in 0 1 2 3 4; do for L in random alt; do for n in 4 16; do
  echo "--seed $s --n_forget 4 --n_neg $n --layout $L --arms B0_frozen,R_hopB_bsup_neg_AMdist"
done; done; done
}
mkdir -p explore/E2/logs
jobs_list | xargs -P 4 -I{} sh -c 'python scripts/cond_cell.py {} >> explore/E2/logs/baselines.log 2>&1'
echo ALLDONE >> explore/E2/logs/baselines.log
