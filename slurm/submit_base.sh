#!/bin/bash
#SBATCH --job-name=toypref_base
#SBATCH --output=logs/%A_%a_base.log
#SBATCH --error=logs/%A_%a_base.log
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --partition=shubhamlong
set -eo pipefail
cd /home/soumojib/repos/toypref
mkdir -p logs
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/train_base.py --seed "$SLURM_ARRAY_TASK_ID"
