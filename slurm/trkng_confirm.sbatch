#!/bin/bash
#SBATCH --job-name=trkng-confirm
#SBATCH --output=explore/TRKNG/logs/slurm_%j.out
#SBATCH --error=explore/TRKNG/logs/slurm_%j.out
#SBATCH --partition=shubhamlong,all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
# TRKNG confirmation queue (frozen prereg: docs/TRKNG_PREREG.md). CPU only.
# Submit from the repo root (log path is relative):  cd /home/soumojib/repos/preference_toy && sbatch slurm/trkng_confirm.sbatch
# Resubmit note: run_queue.sh skips any unit whose JSON already exists, so if this job
# runs out of time, resubmitting the same script continues the queue (no --dependency needed).
set -eo pipefail
cd /home/soumojib/repos/preference_toy
source $HOME/miniconda3/bin/activate base
export TOYPREF_RUNS=$PWD/runs PYTHONUNBUFFERED=1
WORKERS=$SLURM_CPUS_PER_TASK bash explore/TRKNG/run_queue.sh explore/TRKNG/queue_confirm.txt
python explore/TRKNG/analyze.py > explore/TRKNG/analyze_out.txt 2>&1
echo "=== TRKNG CONFIRM DONE ==="
