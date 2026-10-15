#!/bin/bash
#SBATCH --job-name=robol_cell
#SBATCH --output=logs/%A_%a_robol_cell.log
#SBATCH --error=logs/%A_%a_robol_cell.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-25,grogu-1-20
# MANIFEST=results/robol/manifest_*.csv ; one task per row (seed, arms, labels, ns, budget). Skip-if-exists per arm.
set -eo pipefail
cd /home/soumojib/repos/toypref-robo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
ROW=$($PY -c "import csv;r=list(csv.DictReader(open('$MANIFEST')))[$SLURM_ARRAY_TASK_ID];print(r['seed'],r['arms'],r['labels'],r['ns'],r['budget'])")
read S A L N B <<< "$ROW"
$PY scripts/robol_cell.py --seed $S --arms $A --labels $L --ns $N --budget $B
echo "=== ROBOL TASK $SLURM_ARRAY_TASK_ID DONE ==="
