#!/bin/bash
# full-weight control for the local-anchor result: the repo's R_hopB_bsup_neg_AL arm via cond_cell.py
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=$PWD/runs OMP_NUM_THREADS=1
for s in 0 1 2; do for L in random alt; do for n in 4 16; do
  echo "--seed $s --n_forget 4 --n_neg $n --layout $L --arms R_hopB_bsup_neg_AL"; done; done; done | \
  xargs -P 2 -I{} sh -c 'python scripts/cond_cell.py {} >> explore/E2/logs/al_control.log 2>&1'
echo ALLDONE >> explore/E2/logs/al_control.log
