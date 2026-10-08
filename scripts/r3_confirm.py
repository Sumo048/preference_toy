"""Round 3 confirmation: frozen recipe vs each setting's frozen comparator, seeds 5-9 (prereg 14:59)."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from toypref.config import RUNS
W = "R_hopB_bsup_neg_AMdist"
COMP = {("random", 1): "NPO_b10_neg_AMdist", ("random", 2): "NPO_b10_neg_AL", ("random", 4): "PSEUDO_vote_dist",
        ("contig", 4): "NPO_b10_neg_AMdist", ("alt", 4): "KTO_b10_neg_AMmap", ("pairs", 4): "PSEUDO_vote_dist",
        ("b3p1", 4): "PSEUDO_vote_dist"}
SHOW = [W, "R_hopB_bsup_mine_neg", "R_hopB_bsup_neg_AL", "R_hop_bsup_neg_AL", "NPO_b10_neg_AMdist", "NPO_b10_neg_AL",
        "PSEUDO_vote_dist", "KTO_b10_neg_AMmap", "NPO_b100_neg", "B0_frozen"]
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip", "neg_still_forget_far"]
rows = []
for root, L in [(os.path.join(RUNS, "cond", "cells"), "random")] + [(os.path.join(RUNS, "cond", f"cells_{l}"), l) for l in ("contig", "alt", "pairs", "b3p1")]:
    for p in glob.glob(os.path.join(root, "*", "*.json")):
        r = json.load(open(p))
        if r["seed"] >= 5 and r["arm"] in SHOW:
            rows.append({"L": L, **{k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F}})
df = pd.DataFrame(rows)
out = [f"# Round 3 confirmation, seeds 5-9 ({len(df)} runs). Recipe: {W}", ""]
summary = []
for (L, k), comp in COMP.items():
    d = df[(df.L == L) & (df.n_forget == k)]
    m = d.groupby("arm")[F].mean().reindex([a for a in SHOW if a in set(d.arm)])
    z = (d[d.arm == W].set_index(["seed", "n_neg"]).tv_to_ideal - d[d.arm == comp].set_index(["seed", "n_neg"]).tv_to_ideal).dropna()
    mm = m.drop(index="B0_frozen", errors="ignore")[["neg_mass_ratio", "retain_tv", "mass_off"]]
    dom = [o for o in mm.index if o != W and (mm.loc[o] <= mm.loc[W]).all() and (mm.loc[o] < mm.loc[W]).any()]
    ok = z.mean() < -2 * z.sem() and not dom
    summary.append((L, k, m.loc[W, "tv_to_ideal"], comp, m.loc[comp, "tv_to_ideal"], m.loc["B0_frozen", "tv_to_ideal"], z.mean(), z.sem(), int((z < 0).sum()), len(z), ok))
    out.append(f"## {L}, n_forget={k}\n\n{m.round(3).to_string()}\n")
    out.append(f"- {W} - {comp}: {z.mean():+.4f} +- {z.sem():.4f} (lower in {(z < 0).sum()}/{len(z)}); dominated by {dom or '-'} -> **{'PASS' if ok else 'fail'}**\n")
out.append("## Summary\n\n| layout | n_forget | recipe | comparator | comparator tv | B0 | diff | lower in | pass |\n|---|---|---|---|---|---|---|---|---|")
for L, k, wt, c, ct, b0, mu, se, lo, n, ok in summary:
    out.append(f"| {L} | {k} | {wt:.3f} | {c} | {ct:.3f} | {b0:.3f} | {mu:+.4f} +- {se:.4f} | {lo}/{n} | {'**PASS**' if ok else 'fail'} |")
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", "r3_confirm.md"); open(p, "w").write("\n".join(out) + "\n")
print("\n".join(out[out.index("## Summary\n\n| layout | n_forget | recipe | comparator | comparator tv | B0 | diff | lower in | pass |\n|---|---|---|---|---|---|---|---|---|"):]))
