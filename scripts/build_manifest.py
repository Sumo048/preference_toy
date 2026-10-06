"""One row per array task: a sweep cell and a chunk of arms."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from toypref.arms import ARMS
from toypref.config import DEFAULT, ROOT

N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
names = list(ARMS)
chunks = [names[i::N_CHUNKS] for i in range(N_CHUNKS)]   # interleaved: balances cost
sw = DEFAULT.sweep
rows = []
for s in sw.seeds:
    for k in sw.n_forget:
        for n in sw.n_neg:
            for c in chunks:
                rows.append({"task": len(rows), "seed": s, "n_forget": k, "n_neg": n, "arms": ",".join(c)})
path = os.path.join(ROOT, "results", "manifest.csv")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} tasks -> {path}")
