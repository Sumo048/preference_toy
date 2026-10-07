#!/bin/bash
#SBATCH --job-name=toypref_smokeN
#SBATCH --output=logs/%j_smoke_next.log
#SBATCH --error=logs/%j_smoke_next.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --partition=all
# Smoke for the next-step experiments: 50-step fine-tunes into a scratch root.
set -eo pipefail
cd /home/soumojib/repos/toypref
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
S=/grogu/datasets/rollout/soumojib/toypref/runs_smoke_next
rm -rf $S
$PY -m pytest -q tests
FILT=$($PY -c "import sys; sys.path.insert(0,'src'); from toypref.arms import FILTERED_ARMS; print(','.join(FILTERED_ARMS))")
$PY scripts/run_cell.py --seed 0 --n_forget 2 --n_neg 2 --steps 50 --arms B0_frozen,B1_filtered,R_ours_both,$FILT --out $S/cells
$PY scripts/run_cell.py --seed 1 --n_forget 4 --n_neg 1 --steps 50 --arms $FILT --out $S/cells
$PY scripts/run_cell.py --seed 0 --n_forget 2 --n_neg 2 --steps 50 --n_pos 4 \
   --arms B0_frozen,B1_filtered,B3_npo_b10,B5_kto_b10,DPO_cpl_b1,RM_bon8,B4_naive_ga,R_ours_both,R_ours_filt_both,R_spread_filt_nhood_neg_kl1 \
   --out $S/cells_npos4
$PY scripts/aggregate.py --cells $S/cells,$S/cells_npos4 --out $S/results
TOYPREF_NEXT_RESULTS=$S/results $PY scripts/report_next_steps.py
echo "SMOKE_NEXT DONE"
