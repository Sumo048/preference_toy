#!/bin/bash
#SBATCH --job-name=toypref_cbase1014
#SBATCH --output=logs/%A_%a_cbase.log
#SBATCH --error=logs/%A_%a_cbase.log
#SBATCH --array=10-14
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --partition=shubhamlong
set -eo pipefail
cd /home/soumojib/repos/toypref-novel
mkdir -p logs
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/cond_base.py --seed "$SLURM_ARRAY_TASK_ID"
