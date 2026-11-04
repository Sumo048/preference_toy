"""Paired D differences (method A - method B) per setting over cells: mean ± SE, A lower in n of N."""
import sys, os, json, numpy as np
sys.argv += []
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
def get(st, s, n, m):
    if m.startswith("repo:"):
        f = os.path.join(ROOT, "runs", "cond", "cells" if st == "k4" else "cells_alt", f"s{s}_k4_n{n}", m[5:] + ".json")
        return json.load(open(f))["tv_to_ideal"] if os.path.exists(f) else None
    f = os.path.join(HERE, "results", f"{st}_s{s}_n{n}", m + ".json")
    return json.load(open(f))["D"] if os.path.exists(f) else None
B1A = "repo:R_hopB_bsup_neg_AMdist"
E = "r1_fc2.0+fc2.1+fc2.2_r4_t0.4-1"
pairs = [(E, B1A), ("b1a_t0.3-1", B1A), (E + "_aAL", "repo:R_hopB_bsup_neg_AL"), (E + "_aAL", B1A),
         (E + "_w1-0.5", "full_w1-0.5"), ("b1a_trunc_r8", B1A)]
print("| A | B | setting | seeds | mean D(A) - D(B) ± SE | A lower |"); print("|---|---|---|---|---|---|")
for a, b in pairs:
    for st in ("k4", "alt"):
        d = [get(st, s, n, a) - get(st, s, n, b) for s in range(5) for n in (4, 16)
             if get(st, s, n, a) is not None and get(st, s, n, b) is not None]
        if not d: continue
        d = np.array(d); seeds = "0-4" if len(d) == 10 else f"{len(d)} cells"
        print(f"| {a} | {b.replace('repo:','')} | {st} | {seeds} | {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f} | {(d<0).sum()}/{len(d)} |")
