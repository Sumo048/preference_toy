#!/bin/bash
#SBATCH --job-name=toypref_pilot
#SBATCH --output=logs/%j_pilot.log
#SBATCH --error=logs/%j_pilot.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --partition=shubhamlong
# Pilot: full base for seed 0, then one cell with a few arms at three peak LRs.
# Used ONLY to pick one fine-tune LR for every arm (see docs/DESIGN.md).
set -eo pipefail
cd /home/soumojib/repos/toypref
mkdir -p logs
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/runs_pilot
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
$PY -m pytest -q tests
$PY scripts/train_base.py --seed 0
ARMS=B0_frozen,B1_filtered,R_ours_both,R_spread_nhood_both,R_spread_nhood_neg,B3_npo_b10,B5_kto_b10
for LR in 1e-4 3e-4 1e-3; do
  $PY scripts/run_cell.py --seed 0 --n_forget 2 --n_neg 8 --arms $ARMS --lr $LR \
      --out $TOYPREF_RUNS/cells_lr$LR
done
echo "PILOT DONE $(date)"
