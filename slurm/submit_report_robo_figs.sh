#!/bin/bash
#SBATCH --job-name=rep_robo_figs
#SBATCH --output=/home/soumojib/repos/toypref-report/logs/%j_rep_robo_figs.log
#SBATCH --error=/home/soumojib/repos/toypref-report/logs/%j_rep_robo_figs.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --partition=shubhamlong
# Report: draw robo_setup / robo_chunkfan / robo_rollouts from the saved seed-0 base + arm policies.
# Submit with --exclude=grogu-1-20,grogu-0-25. Optional env ONLY=setup,fan,rollouts
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
OUT=/grogu/datasets/rollout/soumojib/toypref/report_robo
$PY scripts/report/robo_figs.py --base $OUT/base --arms $OUT/arms --facts $OUT/fig_facts.json --only ${ONLY:-setup,fan,rollouts}
