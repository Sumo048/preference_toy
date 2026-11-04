#!/bin/bash
#SBATCH --job-name=toypref_ckptu
#SBATCH --output=logs/%j_ckptu.log
#SBATCH --error=logs/%j_ckptu.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-1-20,grogu-0-25
# Unconditional default-width base, seed 0 (smoke-test base_k2) for ckpt/.
set -eo pipefail
cd /home/soumojib/repos/toypref-cond
mkdir -p logs
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/runs_ckpt
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/train_base.py --seed 0
