#!/bin/bash
#SBATCH --job-name=robo_base
#SBATCH --output=logs/%A_%a_robo_base.log
#SBATCH --error=logs/%A_%a_robo_base.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-25
# robo2d base policies: one array task per seed (SLURM_ARRAY_TASK_ID = seed).
# Optional env: Q_COL, Q_JERK (injection rates), STEPS, OUT (output root), EXTRA (extra args).
set -eo pipefail
cd /home/soumojib/repos/toypref-robo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
ARGS="--seed $SLURM_ARRAY_TASK_ID"
[ -n "$Q_COL" ] && ARGS="$ARGS --q_col $Q_COL"
[ -n "$Q_JERK" ] && ARGS="$ARGS --q_jerk $Q_JERK"
[ -n "$STEPS" ] && ARGS="$ARGS --steps $STEPS"
[ -n "$OUT" ] && ARGS="$ARGS --out $OUT"
$PY scripts/robo_base.py $ARGS $EXTRA
echo "=== ROBO BASE $SLURM_ARRAY_TASK_ID DONE ==="
