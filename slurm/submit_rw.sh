#!/bin/bash
#SBATCH --job-name=toypref_rw
#SBATCH --output=logs/%A_%a_rw.log
#SBATCH --error=logs/%A_%a_rw.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=06:00:00
#SBATCH --partition=shubhamlong
# Round rw. MANIFEST=<path> sbatch --array=0-N%K ... ; run_cell skips arms already on disk.
set -eo pipefail
cd /home/soumojib/repos/toypref-rwin
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/run_task.py "$SLURM_ARRAY_TASK_ID" "$MANIFEST"
echo "=== RW TASK $SLURM_ARRAY_TASK_ID DONE $(date) ==="
