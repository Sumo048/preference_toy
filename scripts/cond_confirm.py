"""Confirmation of the conditional round (frozen in docs/COND_REROUTE_PREREG.md, 13:29)."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from toypref.config import RUNS
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "neg_still_forget_near", "neg_still_forget_far",
     "rows_target_frac_in_forget"]
rows = [json.load(open(p)) for p in glob.glob(os.path.join(RUNS, "cond", "cells", "*", "*.json"))]
df = pd.DataFrame([{k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F} for r in rows if r["seed"] >= 5])
W, BASE = "R_hop_bsup_neg", ["NPO_b100_neg", "NPO_b10_neg_A"]
out = [f"# Conditional confirmation, seeds 5-9 ({len(df)} runs)", ""]
for k in (1, 2, 4):
    d = df[df.n_forget == k]
    out.append(f"## n_forget={k}\n")
    m = d.groupby("arm")[F].mean().sort_values("tv_to_ideal")
    out.append(m.round(3).to_string() + "\n")
    passes = []
    for b in BASE:
        z = (d[d.arm == W].set_index(["seed", "n_neg"]).tv_to_ideal - d[d.arm == b].set_index(["seed", "n_neg"]).tv_to_ideal).dropna()
        ok = z.mean() < -2 * z.sem()
        passes.append(ok)
        out.append(f"- {W} - {b}: {z.mean():+.4f} +- {z.sem():.4f} (lower in {(z < 0).sum()}/{len(z)}) {'PASS' if ok else 'fail'}")
    mm = m[["neg_mass_ratio", "retain_tv", "mass_off"]]
    dom = [o for o in mm.index if o != W and (mm.loc[o] <= mm.loc[W]).all() and (mm.loc[o] < mm.loc[W]).any()]
    out.append(f"- dominated by: {dom or '-'}")
    if k == 4:
        out.append(f"\n**CONFIRMATION at n_forget=4: {'PASSED' if all(passes) and not dom else 'FAILED'}**")
    out.append("")
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", "confirm.md"); open(p, "w").write("\n".join(out) + "\n")
print("\n".join(out))
