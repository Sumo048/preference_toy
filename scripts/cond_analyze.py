"""Conditional round: decision rule of docs/COND_REROUTE_PREREG.md, per setting x n_forget."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from toypref.config import RUNS
from toypref.cond.experiment import ARMS

seeds = set(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4").split(","))
out_name = sys.argv[2] if len(sys.argv) > 2 else "exploration.md"
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip", "neg_still_forget_near",
     "neg_still_forget_far", "rows_target_frac_in_forget", "rows_target_frac_off", "rows_transplant_frac_bad"]
rows = []
for p in glob.glob(os.path.join(RUNS, "cond", "cells", "*", "*.json")):
    r = json.load(open(p))
    if r["seed"] in seeds:
        rows.append({k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F})
df = pd.DataFrame(rows)
reroute = lambda a: ARMS[a].kind == "reroute"
practical_rr = lambda a: reroute(a) and ARMS[a].practical
out = [f"# Conditional round ({out_name}), seeds {sorted(seeds)}, {len(df)} runs", ""]
for setting, anc in (("NEG", False), ("NEG+U", None)):
    for k in sorted(df.n_forget.unique()):
        d = df[df.n_forget == k]
        pool = [a for a in d.arm.unique() if ARMS[a].kind != "frozen" and (anc is None or ARMS[a].anchor == anc)]
        m = d[d.arm.isin(pool)].groupby("arm")[F].mean().sort_values("tv_to_ideal")
        b0 = d[d.arm == "B0_frozen"].tv_to_ideal.mean()
        out.append(f"## {setting}, n_forget={k} (B0 {b0:.3f})\n")
        rr = [a for a in m.index if practical_rr(a)]
        nr = [a for a in m.index if not reroute(a)]
        if rr and nr:
            x = d[d.arm == rr[0]].set_index(["seed", "n_neg"]).tv_to_ideal
            y = d[d.arm == nr[0]].set_index(["seed", "n_neg"]).tv_to_ideal
            z = (x - y).dropna()
            mm = m[["neg_mass_ratio", "retain_tv", "mass_off"]]
            dom = [o for o in m.index if o != rr[0] and (mm.loc[o] <= mm.loc[rr[0]]).all() and (mm.loc[o] < mm.loc[rr[0]]).any()]
            c1 = rr[0] == [a for a in m.index if a != "R_own_nhood_neg"][0]
            ok = c1 and z.mean() < -2 * z.sem() and not dom
            out.append(f"best practical rerouting `{rr[0]}` vs best baseline `{nr[0]}`: {z.mean():+.4f} +- {z.sem():.4f} "
                       f"(lower in {(z < 0).sum()}/{len(z)}); dominated by {dom or '-'} -> **{'PASS' if ok else 'no'}**\n")
        out.append("| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | tgt off | tx bad |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for a, r in m.iterrows():
            out.append(f"| {a}{' (R)' if reroute(a) else ''} | {r.tv_to_ideal:.3f} | {r.neg_mass_ratio:.3f} | {r.retain_tv:.3f} | "
                       f"{r.mass_off:.3f} | {r.retain_noise_flip:.2f} | {r.neg_still_forget_near:.2f}/{r.neg_still_forget_far:.2f} | "
                       f"{r.rows_target_frac_in_forget:.2f} | {r.rows_target_frac_off:.2f} | {r.rows_transplant_frac_bad:.2f} |")
        out.append("")
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", out_name)
os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write("\n".join(out) + "\n")
print("\n".join(l for l in out if l.startswith("##") or l.startswith("best")))
