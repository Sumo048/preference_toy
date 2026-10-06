#!/bin/bash
#SBATCH --job-name=toypref_agg
#SBATCH --output=logs/%j_agg.log
#SBATCH --error=logs/%j_agg.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --partition=all
set -eo pipefail
cd /home/soumojib/repos/toypref
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
$PY scripts/aggregate.py
$PY scripts/plot.py
