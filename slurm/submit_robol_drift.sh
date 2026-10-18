#!/bin/bash
#SBATCH --job-name=robol_drift
#SBATCH --output=logs/%A_%a_robol_drift.log
#SBATCH --error=logs/%A_%a_robol_drift.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --partition=all,shubhamlong
#SBATCH --exclude=grogu-0-25,grogu-1-20
# MANIFEST rows: seed,arm,labels,ns[,steps]. Never the shubhamlong A6000 nodes.
set -eo pipefail
cd /home/soumojib/repos/toypref-robo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
ROW=$($PY -c "import csv;r=list(csv.DictReader(open('$MANIFEST')))[$SLURM_ARRAY_TASK_ID];print(r['seed'],r['arm'],r['labels'],r['ns'],(r.get('steps') or '50;100;200;500;1000;2000;5000').replace(';',','))")
read S A L N ST <<< "$ROW"
$PY scripts/robol_drift.py --seed $S --arm $A --labels $L --ns $N --steps $ST
echo "=== ROBOL DRIFT TASK $SLURM_ARRAY_TASK_ID DONE ==="
