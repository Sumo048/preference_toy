#!/bin/bash
#SBATCH --job-name=toypref_smoke
#SBATCH --output=logs/%j_smoke.log
#SBATCH --error=logs/%j_smoke.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --partition=all
set -eo pipefail
cd /home/soumojib/repos/toypref
mkdir -p logs
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOYPREF_RUNS=/home/soumojib/repos/toypref/runs_smoke
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
$PY scripts/train_base.py --seed 0 --steps 2000
$PY scripts/run_cell.py --seed 0 --n_forget 2 --n_neg 2 --steps 50
echo "SMOKE DONE $(date)"
