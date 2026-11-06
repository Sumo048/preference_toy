#!/bin/bash
#SBATCH --job-name=trkng-k1s9n1
#SBATCH --output=explore/TRKNG/logs/slurm_%j.out
#SBATCH --error=explore/TRKNG/logs/slurm_%j.out
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-1-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
# Rerun of the k1 s9 n1 cell only (masks failed the hooked-decode assertion on grogu-1-20 in job 4069868).
# Same frozen units as queue_confirm.txt; the masks unit runs before the train units (which need them).
# Submit with: sbatch --chdir=/home/soumojib/repos/preference_toy slurm/trkng_rerun_k1_s9_n1.sbatch
set -eo pipefail
cd /home/soumojib/repos/preference_toy
source $HOME/miniconda3/bin/activate base
export TOYPREF_RUNS=$PWD/runs PYTHONUNBUFFERED=1
Q=/home/soumojib/repos/preference_toy/.claude/worktrees/trkng-confirm-slurm/slurm/queue_rerun_k1_s9_n1.txt
WORKERS=1 bash explore/TRKNG/run_queue.sh <(head -n 1 $Q)
WORKERS=3 bash explore/TRKNG/run_queue.sh <(tail -n +2 $Q)
python explore/TRKNG/analyze.py confirm > explore/TRKNG/analyze_out.txt 2>&1
echo "=== TRKNG k1 s9 n1 RERUN DONE ==="
