#!/bin/bash
#SBATCH --job-name=rep_robol_pdata
#SBATCH --output=logs/%j_rep_robol_pdata.log
#SBATCH --error=logs/%j_rep_robol_pdata.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --partition=shubhamlong
# Submit with --exclude=grogu-1-20,grogu-0-25 on the command line.
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
$PY scripts/report/robol_policy_data.py
