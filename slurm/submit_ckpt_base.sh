#!/bin/bash
#SBATCH --job-name=toypref_ckpt
#SBATCH --output=logs/%A_%a_ckpt.log
#SBATCH --error=logs/%A_%a_ckpt.log
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-1-20,grogu-0-25
# Retrain conditional pi_ref seeds 0-9 for ckpt/ (originals pruned; s0 retrain checks determinism).
set -eo pipefail
cd /home/soumojib/repos/toypref-cond
mkdir -p logs
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/runs_ckpt
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/cond_base.py --seed "$SLURM_ARRAY_TASK_ID"
