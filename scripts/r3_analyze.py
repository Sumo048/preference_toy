"""Round 3 + layouts: one pool of every practical negatives-only method (with or without the
unlabelled pi_ref pool), best rerouting vs best non-rerouting, seed-paired over (seed, n_neg).

usage: r3_analyze.py <seeds, e.g. 0,1,2,3,4> <out.md>
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from toypref.config import RUNS
from toypref.cond.experiment import ARMS

seeds = {int(x) for x in sys.argv[1].split(",")}
out_name = sys.argv[2]
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip", "neg_still_forget_near",
     "neg_still_forget_far", "rows_target_frac_in_forget", "vote_precision", "vote_recall", "local_anchor_n"]
rows = []
for root, layout in [(os.path.join(RUNS, "cond", "cells"), "random")] + \
        [(os.path.join(RUNS, "cond", f"cells_{L}"), L) for L in ("contig", "alt", "pairs", "b3p1")]:
    for p in glob.glob(os.path.join(root, "*", "*.json")):
        r = json.load(open(p))
        if r["seed"] in seeds and r["arm"] in ARMS:
            rows.append({"layout": layout, **{k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F}})
df = pd.DataFrame(rows)
reroute = lambda a: ARMS[a].kind == "reroute"
eligible = lambda a: ARMS[a].kind != "frozen" and ARMS[a].practical
out = [f"# Round 3 / layouts ({out_name}), seeds {sorted(seeds)}, {len(df)} runs", ""]
verdict = []
for layout in ("random", "contig", "alt", "pairs", "b3p1"):
    for k in (1, 2, 4):
        d = df[(df.layout == layout) & (df.n_forget == k)]
        if d.empty:
            continue
        cnt = d.groupby("arm").size()
        full = cnt.max()
        pool = [a for a in cnt.index if eligible(a) and cnt[a] == full]
        m = d[d.arm.isin(pool)].groupby("arm")[F].mean().sort_values("tv_to_ideal")
        b0 = d[d.arm == "B0_frozen"].tv_to_ideal.mean()
        rr = [a for a in m.index if reroute(a)]
        nr = [a for a in m.index if not reroute(a)]
        out.append(f"## layout={layout}, n_forget={k} (B0 {b0:.3f}; {full} runs per arm)\n")
        if rr and nr:
            z = (d[d.arm == rr[0]].set_index(["seed", "n_neg"]).tv_to_ideal
                 - d[d.arm == nr[0]].set_index(["seed", "n_neg"]).tv_to_ideal).dropna()
            mm = m[["neg_mass_ratio", "retain_tv", "mass_off"]]
            dom = [o for o in mm.index if o != rr[0] and (mm.loc[o] <= mm.loc[rr[0]]).all() and (mm.loc[o] < mm.loc[rr[0]]).any()]
            ok = m.index[0] == rr[0] and z.mean() < -2 * z.sem() and not dom
            verdict.append((layout, k, rr[0], m.loc[rr[0], "tv_to_ideal"], nr[0], m.loc[nr[0], "tv_to_ideal"], z.mean(), z.sem(), ok))
            out.append(f"best rerouting `{rr[0]}` vs best other `{nr[0]}`: {z.mean():+.4f} +- {z.sem():.4f} "
                       f"(lower in {(z < 0).sum()}/{len(z)}); dominated by {dom or '-'} -> **{'PASS' if ok else 'no'}**\n")
        out.append("| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for a, r in m.head(14).iterrows():
            out.append(f"| {a}{' (R)' if reroute(a) else ''} | {r.tv_to_ideal:.3f} | {r.neg_mass_ratio:.3f} | {r.retain_tv:.3f} | "
                       f"{r.mass_off:.3f} | {r.retain_noise_flip:.2f} | {r.neg_still_forget_near:.2f}/{r.neg_still_forget_far:.2f} | "
                       f"{r.rows_target_frac_in_forget:.2f} | {r.vote_precision:.2f}/{r.vote_recall:.2f} | {r.local_anchor_n:.0f} |")
        out.append("")
out.append("## Verdict\n\n| layout | n_forget | best rerouting | tv | best other | tv | paired diff | rerouting best? |\n|---|---|---|---|---|---|---|---|")
for L, k, r_, rt, o_, ot, mu, se, ok in verdict:
    out.append(f"| {L} | {k} | {r_} | {rt:.3f} | {o_} | {ot:.3f} | {mu:+.4f} +- {se:.4f} | {'**YES**' if ok else 'no'} |")
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", out_name)
open(p, "w").write("\n".join(out) + "\n")
print("\n".join(out[out.index("## Verdict\n") if "## Verdict\n" in out else -len(verdict) - 3:]))
