#!/bin/bash
#SBATCH --job-name=toypref_sweep
#SBATCH --output=logs/%A_%a_sweep.log
#SBATCH --error=logs/%A_%a_sweep.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --partition=shubhamlong
# --array is passed on the command line (size = rows of results/manifest.csv).
# run_cell.py skips arms whose result file exists, so a resubmit resumes.
set -eo pipefail
cd /home/soumojib/repos/toypref
mkdir -p logs
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/run_task.py "$SLURM_ARRAY_TASK_ID"
echo "=== TASK $SLURM_ARRAY_TASK_ID DONE $(date) ==="
