#!/bin/bash
#SBATCH --job-name=r2d_base
#SBATCH --output=logs/%A_%a_r2d_base.log
#SBATCH --error=logs/%A_%a_r2d_base.log
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --partition=shubhamlong
# task 0: unconditional base seed 0; task 1: conditional base seed 0
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/report_2d
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then $PY scripts/train_base.py --seed 0; else $PY scripts/cond_base.py --seed 0; fi
echo "=== DONE ==="
