#!/bin/bash
#SBATCH --job-name=toypref_rerun
#SBATCH --output=logs/%A_%a_rerun.log
#SBATCH --error=logs/%A_%a_rerun.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --partition=shubhamlong
# Re-run ONE arm (--force) in every cell. ARM=<name> sbatch --array=0-19 ...
# Task i handles cells i, i+20, i+40, ... of the sweep.
set -eo pipefail
cd /home/soumojib/repos/toypref
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
$PY - "$SLURM_ARRAY_TASK_ID" "$ARM" <<'PYEOF'
import subprocess, sys, os
sys.path.insert(0, "src")
from toypref.config import DEFAULT
tid, arm = int(sys.argv[1]), sys.argv[2]
sw = DEFAULT.sweep
cells = [(s, k, n) for s in sw.seeds for k in sw.n_forget for n in sw.n_neg]
for s, k, n in cells[tid::20]:
    subprocess.check_call([sys.executable, "scripts/run_cell.py", "--seed", str(s), "--n_forget", str(k),
                           "--n_neg", str(n), "--arms", arm, "--force"])
PYEOF
echo "=== RERUN $ARM TASK $SLURM_ARRAY_TASK_ID DONE ==="
