"""Manifest for the robo2d sweep: one row per (seed, arm group). Every arm of a seed writes into
cells/full_st{steps}_m64/s{seed}/; the hop cache is built once per seed (lock file) and shared.

    python scripts/robo_manifest.py --seeds 0,1,2 --out results/robo/manifest_full.csv
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from toypref.robo import methods as MT

GROUPS = [["B0_frozen", "NPO_b10_neg", "NPO_b100_neg", "KTO_b10_neg", "KTO_b100_neg"],
          ["R_hop_bsup_neg", "R_hop_bsup_neg_AL", "R_ours_neg"],
          ["SDPO_hop_pin_b1", "SDPO_fresh_b1", "SDPO_fresh_b25"],
          ["SDPO_pin_b1", "SDPO_pin_b25"]]
assert sorted(sum(GROUPS, [])) == sorted(MT.ARMS), "every arm exactly once"

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--out", default="results/robo/manifest_full.csv")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "arms", "steps"])
    for s in a.seeds.split(","):
        for g in GROUPS:
            w.writerow([s, ",".join(g), a.steps])
print(f"{a.out}: {len(a.seeds.split(',')) * len(GROUPS)} tasks")
