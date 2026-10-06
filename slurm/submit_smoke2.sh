#!/bin/bash
#SBATCH --job-name=toypref_smoke2
#SBATCH --output=logs/%j_smoke2.log
#SBATCH --error=logs/%j_smoke2.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --partition=all
set -eo pipefail
cd /home/soumojib/repos/toypref
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
$PY -m pytest -q tests
$PY scripts/run_cell.py --seed 0 --n_forget 1 --n_neg 1 --steps 600 \
  --arms B0_frozen,B1_filtered,RM_bon8,R_spread_nhood_neg_kl0.3,R_online2_spread_nhood_neg,B5_kto_b100_negonly,DPO_cpl_b25 \
  --out /grogu/datasets/rollout/soumojib/toypref/runs_smoke2/cells
TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/runs_smoke2 $PY scripts/aggregate.py --out /grogu/datasets/rollout/soumojib/toypref/runs_smoke2/results
$PY scripts/plot.py --results /grogu/datasets/rollout/soumojib/toypref/runs_smoke2/results --cells /grogu/datasets/rollout/soumojib/toypref/runs_smoke2/cells --sample_cells s0_k1_n1
echo "SMOKE2 DONE"
