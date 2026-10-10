#!/bin/bash
#SBATCH --job-name=robo_sweep
#SBATCH --output=logs/%A_%a_robo_sweep.log
#SBATCH --error=logs/%A_%a_robo_sweep.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --partition=shubhamlong
#SBATCH --exclude=grogu-0-25
# MANIFEST=results/robo/manifest_full.csv ; one task per (seed, arm group); per-arm outputs are
# skip-if-exists (same base policy sha, steps, m_eval), so a rerun only does what is missing. Every arm
# JSON records base_hash (the CRN base rollout); robo_report.py refuses seeds whose arms disagree. Optional env: SMOKE=N (restrict setups).
set -eo pipefail
cd /home/soumojib/repos/toypref-robo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
ROW=$($PY -c "import csv;r=list(csv.DictReader(open('$MANIFEST')))[$SLURM_ARRAY_TASK_ID];print(r['seed'],r['arms'],r['steps'])")
read S A ST <<< "$ROW"
EXTRA=""
[ -n "$SMOKE" ] && EXTRA="--smoke $SMOKE"
$PY scripts/robo_cell.py --seed $S --arms $A --steps $ST $EXTRA
echo "=== ROBO TASK $SLURM_ARRAY_TASK_ID DONE ==="
