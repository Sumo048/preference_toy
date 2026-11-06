#!/bin/bash
#SBATCH --job-name=trkng-render
#SBATCH --output=explore/TRKNG/logs/slurm_%j.out
#SBATCH --error=explore/TRKNG/logs/slurm_%j.out
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-1-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
# TRKNG rendered outputs: retrains B1a, B6, TC (frozen path) for alt s5 n4 and k4 s5 n4 and plots them.
# Submit from the repo root: sbatch slurm/trkng_render.sbatch
set -eo pipefail
source $HOME/miniconda3/bin/activate base
export TOYPREF_RUNS=/home/soumojib/repos/preference_toy/runs PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p explore/TRKNG/logs
python explore/TRKNG/render.py cell alt 5 4 &
python explore/TRKNG/render.py cell k4 5 4 &
wait
echo "=== TRKNG RENDER DONE ==="
