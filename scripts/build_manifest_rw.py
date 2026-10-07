"""Manifests for round rw (docs/REROUTE_WINS_PREREG.md), exploration phase.

results/rw/manifest_A.csv  base (new arms), cover=0.5, cover=0.25, contam=0.2
results/rw/manifest_B.csv  std=0.45, std=0.6 (need the std base policies first)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from toypref.arms import ARMS, RW_ARMS

SEEDS = (0, 1, 2, 3, 4)
N_FORGET = (1, 2, 4)
N_NEG = (1, 4, 16)
TP_ARMS = ["B0_frozen", "B1_filtered", "B3_npo_b1", "B3_npo_b10", "B3_npo_b100", "B5_kto_b10",
           "B5_kto_b100", "DPO_cpl_b1", "DPO_cpl_b25", "RM_bon8", "R_ours_both", "R_spread_log_both",
           "R_spread_nhood_both", "R_spread_filt_log_both", "R_spread_nhood_neg_kl1",
           "R_near_nf_log_both", "R_near_nf_nhood_both", "R_spread_nf_log_both", "R_ours_nf_both",
           "R_near_filt_log_both"]
OLD_KEY = ["B0_frozen", "ORACLE_retain_bc", "B1_filtered", "B3_npo_b1", "B3_npo_b10", "B3_npo_b100",
           "B3_npo_b100_negonly", "B5_kto_b10", "B5_kto_b100", "B5_kto_b100_negonly", "DPO_cpl_b1",
           "DPO_cpl_b25", "RM_bon8", "R_ours_both", "R_ours_neg", "R_spread_log_both", "R_spread_log_neg",
           "R_spread_nhood_both", "R_spread_nhood_neg", "R_spread_filt_log_both", "R_spread_nhood_neg_kl1"]
for a in TP_ARMS + OLD_KEY + RW_ARMS:
    assert a in ARMS, a


def chunks(names, n):
    return [names[i::n] for i in range(n)]


def rows(regimes):
    out = []
    for regime, arms, nch in regimes:
        for s in SEEDS:
            for k in N_FORGET:
                for n in N_NEG:
                    for c in chunks(arms, nch):
                        out.append({"task": len(out), "seed": s, "n_forget": k, "n_neg": n,
                                    "regime": regime, "arms": ",".join(c)})
    return out


A = rows([("", list(RW_ARMS), 4), ("cover=0.5", TP_ARMS, 2), ("cover=0.25", TP_ARMS, 2),
          ("contam=0.2", TP_ARMS, 2)])
B = rows([("std=0.45", OLD_KEY + list(RW_ARMS), 6), ("std=0.6", OLD_KEY + list(RW_ARMS), 6)])
root = os.path.join(os.path.dirname(__file__), "..", "results", "rw")
os.makedirs(root, exist_ok=True)
for name, rs in (("manifest_A.csv", A), ("manifest_B.csv", B)):
    with open(os.path.join(root, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    print(name, len(rs), "tasks")
