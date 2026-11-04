"""E5 tables and verdicts (markdown to stdout, JSON to results/summary.json).

Graft: explore/E5/results/graft_K10c_<setting>_s<seed>_n<npm>.json.
B0, B1a, B1b: origin/art/arms results/baselines/summary.json (snapshot in baselines/arms_summary.json).
B6 (E3/E4 track-masked BC): origin/art/confirm-E4 explore/E4/results (snapshot in baselines/e4_rows.json).
Test per setting (prereg): d_i = D_graft,i - D_B1a,i over the cells i = (seed 5-9, npm 4 and 16);
PASS iff mean(d) < -2 * sd(d, ddof=1) / sqrt(N).
"""
import json, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ORDER = ["k4", "alt", "k1", "contig", "pairs", "b3p1", "k2"]
SEEDS, NPMS = range(5, 10), (4, 16)
arms = json.load(open(os.path.join(H, "baselines", "arms_summary.json")))["rows"]
e4 = json.load(open(os.path.join(H, "baselines", "e4_rows.json")))["rows"]
base = {}
for r in arms:
    base[(r["arm"], r["setting"], r["seed"], r["npm"])] = (r["D"], r["removed"], r["collateral"], r["mass_off"])
for r in e4:
    base[(r["arm"], r["setting"], r["seed"], r["npm"])] = (r["D"], r["removed"], r["collateral"], r["mass_off"])
for (S, s, n) in [(S, s, n) for S in ORDER for s in SEEDS for n in NPMS]:
    f = os.path.join(H, "results", f"graft_K10c_{S}_s{s}_n{n}.json")
    if os.path.exists(f):
        r = json.load(open(f))
        base[("graft", S, s, n)] = (r["D"], r["Removed"], r["Collateral"], r["off"])


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"))


summary = {}
rows = []
print("| Setting | N | Graft D | B1a D | B1b D | B6 D | B0 D | Graft Removed | Graft Collateral | B1a Collateral | graft − B1a (± SE) | lower in n of N | Test |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for S in ORDER:
    cells = [(s, n) for s in SEEDS for n in NPMS]
    have = [c for c in cells if ("graft", S, *c) in base and ("B1a", S, *c) in base]
    if not have:
        print(f"| {S} | 0 | – | | | | | | | | | | not run |"); continue
    g = np.array([base[("graft", S, *c)] for c in have])
    b = {a: np.array([base[(a, S, *c)] for c in have]) for a in ("B1a", "B1b", "B6", "B0")}
    d = g[:, 0] - b["B1a"][:, 0]
    dm, se = ms(d)
    ok = dm < -2 * se
    lower = int((d < 0).sum())
    # seed-clustered (npm-averaged per seed)
    ds = [np.mean([x for c, x in zip(have, d) if c[0] == s]) for s in SEEDS if any(c[0] == s for c in have)]
    dsm, dss = ms(ds)
    verdict = ("PASS" if ok else "FAIL") + ("" if len(have) == len(cells) else f" (incomplete, N={len(have)})")
    f = lambda a, i: "%.3f ± %.3f" % ms(a[:, i])
    print(f"| {S} | {len(have)} | {f(g, 0)} | {f(b['B1a'], 0)} | {f(b['B1b'], 0)} | {f(b['B6'], 0)} | {f(b['B0'], 0)} | "
          f"{f(g, 1)} | {f(g, 2)} | {f(b['B1a'], 2)} | {dm:+.3f} ± {se:.3f} | {lower} of {len(have)} | {verdict} |")
    summary[S] = {"N": len(have), "d_mean": dm, "d_se": se, "lower": lower, "verdict": verdict,
                  "d_seed_clustered": [dsm, dss, len(ds)],
                  "graft": dict(zip(("D", "Removed", "Collateral", "off"), g.mean(0).tolist())),
                  **{a: dict(zip(("D", "Removed", "Collateral", "off"), b[a].mean(0).tolist())) for a in b},
                  "graft_minus_B6": list(ms(g[:, 0] - b["B6"][:, 0])), "graft_lower_than_B6": int((g[:, 0] < b["B6"][:, 0]).sum()),
                  "graft_minus_B1b": list(ms(g[:, 0] - b["B1b"][:, 0])), "graft_lower_than_B1b": int((g[:, 0] < b["B1b"][:, 0]).sum())}

print("\nDescriptive (not tested): graft − B1b and graft − B6 in D, paired over the same cells; seed-clustered graft − B1a.\n")
print("| Setting | graft − B1b | lower than B1b | graft − B6 | lower than B6 | graft − B1a, seed-clustered (n seeds) |")
print("|---|---|---|---|---|---|")
for S, v in summary.items():
    print(f"| {S} | {v['graft_minus_B1b'][0]:+.3f} ± {v['graft_minus_B1b'][1]:.3f} | {v['graft_lower_than_B1b']} of {v['N']} | "
          f"{v['graft_minus_B6'][0]:+.3f} ± {v['graft_minus_B6'][1]:.3f} | {v['graft_lower_than_B6']} of {v['N']} | "
          f"{v['d_seed_clustered'][0]:+.3f} ± {v['d_seed_clustered'][1]:.3f} ({v['d_seed_clustered'][2]}) |")

print("\nPer cell D (graft / B1a / B1b / B6):\n")
print("| Setting | Seed | npm 4 | npm 16 |\n|---|---|---|---|")
for S in ORDER:
    for s in SEEDS:
        cs = []
        for n in NPMS:
            k = ("graft", S, s, n)
            cs.append(" / ".join("%.3f" % base[(a, S, s, n)][0] if (a, S, s, n) in base else "–" for a in ("graft", "B1a", "B1b", "B6")))
        if ("graft", S, s, 4) in base:
            print(f"| {S} | {s} | {cs[0]} | {cs[1]} |")
json.dump(summary, open(os.path.join(H, "results", "summary.json"), "w"), indent=1)
