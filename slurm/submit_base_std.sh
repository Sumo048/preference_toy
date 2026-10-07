#!/bin/bash
#SBATCH --job-name=toypref_basestd
#SBATCH --output=logs/%A_%a_basestd.log
#SBATCH --error=logs/%A_%a_basestd.log
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --partition=shubhamlong
# Base policies for the overlapping-mode regimes: task i -> std (0.45, 0.6)[i // 5], seed i % 5
set -eo pipefail
cd /home/soumojib/repos/toypref-rwin
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
STDS=(0.45 0.6)
STD=${STDS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((SLURM_ARRAY_TASK_ID % 5))
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/train_base.py --seed $SEED --mode_std $STD
