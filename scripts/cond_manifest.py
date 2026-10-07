"""Manifests for the conditional round: exploration (seeds 0-4) or confirmation (seeds 5-9, frozen arms)."""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from toypref.cond.experiment import ARMS
phase = sys.argv[1]
if phase == "explore":
    seeds, arms, nch = range(0, 5), list(ARMS), 2
else:
    seeds, arms, nch = range(5, 10), sys.argv[2].split(","), 1
    assert all(a in ARMS for a in arms)
rows = []
for s in seeds:
    for k in (1, 2, 4):
        for n in (1, 4, 16, 64):
            for c in [arms[i::nch] for i in range(nch)]:
                rows.append({"task": len(rows), "seed": s, "n_forget": k, "n_neg": n, "arms": ",".join(c)})
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", f"manifest_{phase}.csv")
os.makedirs(os.path.dirname(p), exist_ok=True)
w = csv.DictWriter(open(p, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(p, len(rows))
