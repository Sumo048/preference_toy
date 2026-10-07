#!/bin/bash
#SBATCH --job-name=toypref_aggN
#SBATCH --output=logs/%j_agg_next.log
#SBATCH --error=logs/%j_agg_next.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --partition=all
# Aggregates the n_pos=512 cells (incl. the filtered arms) and the n_pos sweep into
# results/next_steps/ ONLY; the main results/ outputs are never written here.
set -eo pipefail
cd /home/soumojib/repos/toypref
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
R=/grogu/datasets/rollout/soumojib/toypref/runs
$PY scripts/aggregate.py --cells $R/cells,$R/cells_npos4,$R/cells_npos16,$R/cells_npos64 --out results/next_steps
$PY scripts/report_next_steps.py
