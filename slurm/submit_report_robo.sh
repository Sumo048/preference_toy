#!/bin/bash
#SBATCH --job-name=rep_robo_base
#SBATCH --output=/home/soumojib/repos/toypref-report/logs/%j_rep_robo_base.log
#SBATCH --error=/home/soumojib/repos/toypref-report/logs/%j_rep_robo_base.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=06:00:00
#SBATCH --partition=shubhamlong
# Report: retrain the robo2d seed-0 base with the repo's own recipe (robo_base.py defaults: 30000 steps,
# q_col 0.015, q_jerk 0.025). Submit with: sbatch --exclude=grogu-1-20,grogu-0-25 slurm/submit_report_robo.sh
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
OUT=/grogu/datasets/rollout/soumojib/toypref/report_robo
mkdir -p $OUT/base
$PY scripts/robo_base.py --seed 0 --out $OUT/base
echo "=== REPORT ROBO BASE DONE ==="
