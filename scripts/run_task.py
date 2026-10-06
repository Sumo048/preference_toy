"""Array-task entry point: run manifest row $SLURM_ARRAY_TASK_ID."""
import csv
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(os.path.join(ROOT, "results", "manifest.csv")) as f:
    row = list(csv.DictReader(f))[tid]
cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_cell.py"), "--seed", row["seed"],
       "--n_forget", row["n_forget"], "--n_neg", row["n_neg"], "--arms", row["arms"]]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
