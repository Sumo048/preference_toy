"""Round 4: every pre-registered decision rule of docs/NOVEL_PREREG.md.

usage: r4_analyze.py <seeds e.g. 10,11,12,13,14> <out.md>
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from toypref.config import RUNS

seeds = {int(x) for x in sys.argv[1].split(",")}
out_name = sys.argv[2]
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip", "neg_still_forget_far",
     "mask_bad_precision", "mask_bad_recall", "mask_good_frac", "trk_refusals", "trk_n_components", "joint_kc"]
NEW = ["TRACK_bc", "TRACK_bc_sortc", "TRACK_bc_M16", "TRACK_bc_M256", "RTRACK_bc", "JOINT_bc", "ATYP_dist", "PRA_trk", "REACH_recipe"]
OLD = ["R_hopB_bsup_neg_AMdist", "R_hop_bsup_neg", "PSEUDO_vote_dist", "NPO_b10_neg_AMdist", "KTO_b10_neg_AMmap"]
RECIPE = "R_hopB_bsup_neg_AMdist"
SETTINGS = [("random", 1), ("random", 2), ("random", 4), ("contig", 4), ("alt", 4), ("pairs", 4), ("b3p1", 4)]


def load(root, tag):
    rows = []
    for p in glob.glob(os.path.join(root, "*", "*.json")):
        r = json.load(open(p))
        if r["seed"] in seeds:
            rows.append({"set": tag, **{k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F}})
    return rows


rows = []
for L in ("random", "contig", "alt", "pairs", "b3p1"):
    rows += load(os.path.join(RUNS, "cond", "cells" if L == "random" else f"cells_{L}"), L)
for st in ("flip0.1", "nodes256", "nodes64"):
    rows += load(os.path.join(RUNS, "cond", f"cells_stress_{st}"), f"stress_{st}")
df = pd.DataFrame(rows)


def cell(setname, k):
    return df[(df["set"] == setname) & (df.n_forget == k)]


def paired(d, a, b):
    x = d[d.arm == a].set_index(["seed", "n_neg"]).tv_to_ideal
    y = d[d.arm == b].set_index(["seed", "n_neg"]).tv_to_ideal
    z = (x - y).dropna()
    return z.mean(), z.sem(), int((z < 0).sum()), len(z)


out = [f"# Round 4 ({out_name}), seeds {sorted(seeds)}, {len(df)} runs", ""]
summ = {}
for L, k in SETTINGS:
    d = cell(L, k)
    m = d.groupby("arm")[F].mean()
    summ[(L, k)] = (d, m)
    out.append(f"## {L}, n_forget={k}\n")
    show = [a for a in NEW + OLD + ["B0_frozen"] if a in m.index]
    out.append(m.loc[show, ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip",
                            "neg_still_forget_far", "mask_bad_precision", "mask_bad_recall"]].round(3).to_string() + "\n")

# ---- TRACK_bc vs recipe (WIN/ADOPT/ABANDON) and vs ATYP
out.append("## Decision: TRACK_bc\n")
out.append("| setting | TRACK | recipe | diff | lower in | WIN | vs ATYP | ATYP >= 0.03 |")
out.append("|---|---|---|---|---|---|---|---|")
wins, worse, atyp_ok = 0, False, True
for L, k in SETTINGS:
    d, m = summ[(L, k)]
    if "TRACK_bc" not in m.index or RECIPE not in m.index:
        out.append(f"| {L} {k} | missing | | | | | | |")
        continue
    mu, se, lo, n = paired(d, "TRACK_bc", RECIPE)
    win = mu <= -0.02 and lo >= 16
    wins += win
    worse |= mu > 0.01
    ma, sa, _, _ = paired(d, "TRACK_bc", "ATYP_dist") if "ATYP_dist" in m.index else (np.nan, np.nan, 0, 0)
    aok = ma <= -0.03
    atyp_ok &= bool(aok)
    out.append(f"| {L} {k} | {m.loc['TRACK_bc', 'tv_to_ideal']:.3f} | {m.loc[RECIPE, 'tv_to_ideal']:.3f} | {mu:+.4f} +- {se:.4f} | "
               f"{lo}/{n} | {'WIN' if win else '-'} | {ma:+.4f} | {'yes' if aok else 'NO'} |")
verdict = "ADOPT" if (wins >= 6 and not worse and atyp_ok) else ("ABANDON" if wins <= 3 else "neither adopt nor abandon")
out.append(f"\nTRACK_bc wins {wins}/7; any setting worse by > 0.01: {worse}; beats ATYP by >= 0.03 everywhere: {atyp_ok} -> **{verdict}**\n")

# ---- general standard: each new arm vs best non-new arm per setting
out.append("## General standard: each new arm vs the best non-new arm (2 sem, not dominated)\n")
out.append("| setting | best non-new | " + " | ".join(NEW) + " |")
out.append("|---|---|" + "---|" * len(NEW))
for L, k in SETTINGS:
    d, m = summ[(L, k)]
    olds = [a for a in OLD if a in m.index]
    if not olds:
        continue
    bo = m.loc[olds, "tv_to_ideal"].idxmin()
    cells = []
    for a in NEW:
        if a not in m.index:
            cells.append("-")
            continue
        mu, se, lo, n = paired(d, a, bo)
        mm = m[["neg_mass_ratio", "retain_tv", "mass_off"]]
        pool = [x for x in mm.index if x != "B0_frozen"]
        dom = any((mm.loc[o] <= mm.loc[a]).all() and (mm.loc[o] < mm.loc[a]).any() for o in pool if o != a)
        ok = mu < -2 * se and not dom
        cells.append(f"{mu:+.3f}{' PASS' if ok else ''}")
    out.append(f"| {L} {k} | {bo} ({m.loc[bo, 'tv_to_ideal']:.3f}) | " + " | ".join(cells) + " |")

# ---- JOINT non-inferiority, PRA delivery, REACH alt
out.append("\n## Decision: JOINT_bc, PRA_trk, REACH_recipe\n")
jn, pn, rr = 0, 0, {}
for L, k in SETTINGS:
    d, m = summ[(L, k)]
    if {"JOINT_bc", "TRACK_bc"} <= set(m.index):
        jn += paired(d, "JOINT_bc", "TRACK_bc")[0] <= 0.01
    if {"PRA_trk", "TRACK_bc"} <= set(m.index):
        ok = (m.loc["PRA_trk", "tv_to_ideal"] <= m.loc["TRACK_bc", "tv_to_ideal"] + 0.005
              and m.loc["PRA_trk", "retain_noise_flip"] <= 0.6 * m.loc["TRACK_bc", "retain_noise_flip"])
        pn += ok
    if {"REACH_recipe", RECIPE} <= set(m.index):
        rr[(L, k)] = paired(d, "REACH_recipe", RECIPE)[0]
out.append(f"- JOINT_bc non-inferior to TRACK_bc (<= +0.01) in {jn}/7 -> **{'PROMOTE' if jn >= 6 else 'no'}** "
           f"(plateau k_c per cell is in the table's joint_kc column of the raw data)")
out.append(f"- PRA_trk within TRACK+0.005 with flips <= 0.6x in {pn}/7 -> **{'ADOPT delivery' if pn >= 6 else 'keep BC'}**")
if rr:
    alt = rr.get(("alt", 4), np.nan)
    others = [v for kk, v in rr.items() if kk != ("alt", 4)]
    keep = alt <= -0.05 and all(v <= 0.01 for v in others)
    out.append(f"- REACH_recipe minus recipe: alt {alt:+.3f}; others {', '.join(f'{v:+.3f}' for v in others)} -> **{'KEEP' if keep else 'drop'}**")

# ---- stress
out.append("\n## Stress cells (random layout, n_forget 2 and 4, n_neg 16 and 64)\n")
out.append("| stress | TRACK | RTRACK | recipe | B0 | TRACK - recipe | lower in | RTRACK - TRACK |")
out.append("|---|---|---|---|---|---|---|---|")
for st in ("flip0.1", "nodes256", "nodes64"):
    d = df[df["set"] == f"stress_{st}"]
    if d.empty:
        continue
    m = d.groupby("arm").tv_to_ideal.mean()
    x = d[d.arm == "TRACK_bc"].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    y = d[d.arm == RECIPE].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    r = d[d.arm == "RTRACK_bc"].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    z, zr = (x - y).dropna(), (r - x).dropna()
    out.append(f"| {st} | {m.get('TRACK_bc', np.nan):.3f} | {m.get('RTRACK_bc', np.nan):.3f} | {m.get(RECIPE, np.nan):.3f} | "
               f"{m.get('B0_frozen', np.nan):.3f} | {z.mean():+.4f} +- {z.sem():.4f} | {(z < 0).sum()}/{len(z)} | {zr.mean():+.4f} +- {zr.sem():.4f} |")
d0 = df[(df["set"] == "random") & df.n_forget.isin([2, 4]) & df.n_neg.isin([16, 64])]
if not d0.empty:
    x = d0[d0.arm == "TRACK_bc"].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    r = d0[d0.arm == "RTRACK_bc"].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    out.append(f"\nClean reference (same cells): TRACK {x.mean():.3f}, RTRACK {r.mean():.3f}, RTRACK clean cost {(r - x).mean():+.4f}")
p = os.path.join(os.path.dirname(__file__), "..", "results", "cond", out_name)
os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write("\n".join(out) + "\n")
print("\n".join(l for l in out if not l.startswith("|") or "WIN" in l or "PASS" in l or "stress" in l.lower()))
print("->", p)
