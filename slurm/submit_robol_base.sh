#!/bin/bash
#SBATCH --job-name=robol_base
#SBATCH --output=logs/%A_%a_robol_base.log
#SBATCH --error=logs/%A_%a_robol_base.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-25
# robol base policies: one array task per seed. Optional env: ARGS (extra args, e.g. injection rates), OUT.
set -eo pipefail
cd /home/soumojib/repos/toypref-robo
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
EXTRA=""
[ -n "$OUT" ] && EXTRA="--out $OUT"
$PY scripts/robol_base.py --seed $SLURM_ARRAY_TASK_ID $ARGS $EXTRA
echo "=== ROBOL BASE $SLURM_ARRAY_TASK_ID DONE ==="
