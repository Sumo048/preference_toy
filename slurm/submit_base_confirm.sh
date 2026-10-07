#!/bin/bash
#SBATCH --job-name=toypref_base59
#SBATCH --output=logs/%A_%a_base59.log
#SBATCH --error=logs/%A_%a_base59.log
#SBATCH --array=5-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --partition=shubhamlong
# Fresh-seed base policies (seeds 5-9, default mode width) for the confirmation phase.
set -eo pipefail
cd /home/soumojib/repos/toypref-rwin
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/train_base.py --seed "$SLURM_ARRAY_TASK_ID"
