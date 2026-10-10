#!/bin/bash
#SBATCH --job-name=toypref_sdpoc
#SBATCH --output=logs/%A_%a_cond.log
#SBATCH --error=logs/%A_%a_cond.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=06:00:00
#SBATCH --partition=shubhamlong,all
# MANIFEST=results/cond/manifest_<phase>.csv; per-arm outputs are skip-if-exists, so preemption is cheap.
set -eo pipefail
cd /home/soumojib/repos/toypref-sdpo
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
ROW=$($PY -c "import csv,sys;r=list(csv.DictReader(open('$MANIFEST')))[$SLURM_ARRAY_TASK_ID];print(r['seed'],r['n_forget'],r['n_neg'],r['arms'],r.get('layout') or 'random',r.get('stress') or '-')")
read S K N A L ST <<< "$ROW"
if [ "$ST" = "-" ]; then ST=""; fi
$PY scripts/cond_cell.py --seed $S --n_forget $K --n_neg $N --arms $A --layout $L --stress "$ST"
echo "=== COND TASK $SLURM_ARRAY_TASK_ID DONE ==="
