"""Round-2 confirmation (V2, V3 of docs/COND_REROUTE_PREREG.md), seeds 5-9."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from toypref.config import RUNS
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip", "neg_still_forget_near",
     "neg_still_forget_far", "rows_target_frac_in_forget"]
ARMS = ["R_hopB_bsup_neg", "R_hop_bsup_neg", "NPO_b100_neg", "NPO_b10_neg_A", "B0_frozen"]
rs = [json.load(open(p)) for p in glob.glob(os.path.join(RUNS, "cond", "cells", "*", "*.json"))]
df = pd.DataFrame([{k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F} for r in rs
                   if r["seed"] >= 5 and r["arm"] in ARMS])
out = [f"# Round 2 (noise veto) confirmation, seeds 5-9 ({len(df)} runs)", ""]
W = "R_hopB_bsup_neg"


def paired(d, a, b):
    z = (d[d.arm == a].set_index(["seed", "n_neg"]).tv_to_ideal - d[d.arm == b].set_index(["seed", "n_neg"]).tv_to_ideal).dropna()
    return z.mean(), z.sem(), int((z < 0).sum()), len(z)


res = {}
for k in (1, 2, 4):
    d = df[df.n_forget == k]
    out.append(f"## n_forget={k}\n")
    out.append(d.groupby("arm")[F].mean().loc[[a for a in ARMS if a in set(d.arm)]].round(3).to_string() + "\n")
    for b in ("R_hop_bsup_neg", "NPO_b100_neg", "NPO_b10_neg_A"):
        mu, se, lo, n = paired(d, W, b)
        res[(k, b)] = mu < -2 * se
        out.append(f"- {W} - {b}: {mu:+.4f} +- {se:.4f} (lower in {lo}/{n}) {'PASS' if mu < -2 * se else 'fail'}")
    out.append("")
out.append(f"**V2 (beats plain hop at n_forget=4): {'CONFIRMED' if res[(4, 'R_hop_bsup_neg')] else 'NOT confirmed'}**")
out.append(f"**V3 (beats best neg-only baseline at n_forget=2): {'CONFIRMED' if res[(2, 'NPO_b100_neg')] and res[(2, 'NPO_b10_neg_A')] else 'NOT confirmed'}**")
out.append(f"n_forget=4 vs both baselines: {'PASS' if res[(4, 'NPO_b100_neg')] and res[(4, 'NPO_b10_neg_A')] else 'fail'}")
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", "confirm_veto.md"); open(p, "w").write("\n".join(out) + "\n")
print("\n".join(out))
