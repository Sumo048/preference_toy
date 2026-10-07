"""Array-task entry point: run manifest row $SLURM_ARRAY_TASK_ID.

usage: run_task.py [task_id] [manifest.csv]   (default manifest: results/manifest.csv)
Optional manifest columns ``n_pos`` and ``out`` are passed through to run_cell.py.
"""
import csv
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
manifest = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "manifest.csv")
with open(manifest) as f:
    row = list(csv.DictReader(f))[tid]
cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_cell.py"), "--seed", row["seed"],
       "--n_forget", row["n_forget"], "--n_neg", row["n_neg"], "--arms", row["arms"]]
if row.get("n_pos"):
    cmd += ["--n_pos", row["n_pos"]]
if row.get("out"):
    cmd += ["--out", row["out"]]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
