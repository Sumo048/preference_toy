#!/bin/bash
#SBATCH --job-name=rep_robo_arms
#SBATCH --output=/home/soumojib/repos/toypref-report/logs/%j_rep_robo_arms.log
#SBATCH --error=/home/soumojib/repos/toypref-report/logs/%j_rep_robo_arms.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=06:00:00
#SBATCH --partition=shubhamlong
# Report: train + save the seed-0 arms (native, 5000 steps, m64). Submit with --exclude=grogu-1-20,grogu-0-25 (and --dependency=afterok:<base>).
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
OUT=/grogu/datasets/rollout/soumojib/toypref/report_robo
$PY scripts/report/robo_train.py --base $OUT/base --out $OUT/arms
