"""Manifests for the two next-step experiments (2026-10-07).

results/next_steps/manifest_filt512.csv  label-filtered reroute arms on the 120 n_pos=512 cells
results/next_steps/manifest_npos.csv     every T_p-dependent arm at n_pos in {4, 16, 64}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from toypref.arms import ARMS, FILTERED_ARMS, depends_on_tp
from toypref.config import DEFAULT, ROOT

N_POS = (4, 16, 64)
N_CHUNKS = 2
out_dir = os.path.join(ROOT, "results", "next_steps")
os.makedirs(out_dir, exist_ok=True)
sw = DEFAULT.sweep
cells = [(s, k, n) for s in sw.seeds for k in sw.n_forget for n in sw.n_neg]


def write(name, rows):
    with open(os.path.join(out_dir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "n_forget", "n_neg", "n_pos", "arms"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"task": i, **r})
    print(f"{len(rows)} tasks -> {name}")


write("manifest_filt512.csv", [{"seed": s, "n_forget": k, "n_neg": n, "n_pos": DEFAULT.n_pos,
                                 "arms": ",".join(FILTERED_ARMS)} for s, k, n in cells])
tp_arms = [a for a, sp in ARMS.items() if depends_on_tp(sp)]
chunks = [tp_arms[i::N_CHUNKS] for i in range(N_CHUNKS)]
print(f"{len(tp_arms)} T_p-dependent arms: {tp_arms}")
print("not rerun (no T_p):", [a for a in ARMS if a not in tp_arms])
write("manifest_npos.csv", [{"seed": s, "n_forget": k, "n_neg": n, "n_pos": p, "arms": ",".join(c)}
                            for p in N_POS for s, k, n in cells for c in chunks])
