#!/bin/bash
#SBATCH --job-name=rep_robol_tr
#SBATCH --output=logs/%A_%a_rep_robol_train.log
#SBATCH --error=logs/%A_%a_rep_robol_train.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=1-12:00:00
#SBATCH --partition=shubhamlong
# array 0..6 -> npo kto fresh pin pin5000 sn8 sn64. Submit with --exclude=grogu-1-20,grogu-0-25 on the command line.
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
NAMES=(npo kto fresh pin pin5000 sn8 sn64)
$PY scripts/report/robol_train_policy.py --name ${NAMES[$SLURM_ARRAY_TASK_ID]}
