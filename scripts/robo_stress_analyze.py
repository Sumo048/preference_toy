"""Stress round: every decision rule of docs/ROBO_STRESS_PREREG.md.

    python scripts/robo_stress_analyze.py [--tag full_st5000_m64]     -> results/robo_stress/RESULTS.md

G = arm - B0 in clean success, seed-paired, mean +- sem over valid seeds (B0 neg clean success >= 0.5).
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robo")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full_st5000_m64")
ap.add_argument("--out", default="results/robo_stress/RESULTS.md")
a = ap.parse_args()

CORE = ["SDPO_fresh_b25", "SDPO_fresh_b10", "SDPO_pin_b25", "NPO_b100_neg", "KTO_b100_neg", "R_hop_bsup_neg_AL", "FILTER_rej"]
CONDS = ["oracle", "native", "pad6", "pad18", "flip0.1", "flip0.25", "flip0.5", "placebo", "recall0.5", "cover0.25"]
GROUPS = ["neg", "noneg", "held"]
FIELDS = [f"{g}_{m}" for g in GROUPS for m in ("success_clean", "bad", "collision", "jerk", "good_broken", "hom_tv")]

recs = []
for d in glob.glob(os.path.join(ROBO, "cells", a.tag + "*", "s*")):
    tagdir = os.path.basename(os.path.dirname(d))
    if tagdir == a.tag:
        cond = "native"
    elif (m := re.fullmatch(re.escape(a.tag) + r"_L(.+)", tagdir)):
        cond = m.group(1)
    else:
        continue
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(f) == "separability.json":
            continue
        r = json.load(open(f))
        recs.append({"cond": cond, "arm": r["arm"], "seed": r["seed"], "edit_norm": r.get("edit_norm", np.nan),
                     "base_hash": r.get("base_hash"), **{k: r.get(k, np.nan) for k in FIELDS}})
df = pd.DataFrame(recs)
b0 = df[(df.arm == "B0_frozen") & (df.cond == "native")].set_index("seed")
valid = sorted(int(s) for s in b0.index[b0.neg_success_clean >= 0.5])
dropped = sorted(int(s) for s in b0.index[b0.neg_success_clean < 0.5])
df = df[df.seed.isin(valid)]
# every arm of a seed must share the base rollout
for s in valid:
    hs = set(df[df.seed == s].base_hash.dropna())
    assert len(hs) == 1, f"seed {s}: arms scored against different base rollouts {hs}"

out = [f"# Robo stress round: results", "",
       f"Valid seeds {valid}; dropped {dropped} (frozen base clean success < 0.5). Runs: {len(df)}.", ""]


def gain(arm, cond, group="neg", metric="success_clean"):
    x = df[(df.arm == arm) & (df.cond == cond)].set_index("seed")[f"{group}_{metric}"]
    y = b0.loc[valid, f"{group}_{metric}"]
    z = (x - y).dropna()
    return z.mean(), z.sem() if len(z) > 1 else np.nan, len(z), z


def diff(arm1, c1, arm2, c2, group="neg"):
    z = (gain(arm1, c1, group)[3] - gain(arm2, c2, group)[3]).dropna()
    return z.mean(), z.sem() if len(z) > 1 else np.nan, len(z)


def fmt(m, s, n):
    if n == 0 or np.isnan(m):
        return "—"
    tag = " ▲" if m > 2 * s else (" ▼" if m < -2 * s else "")
    return f"{100 * m:+.1f}±{100 * s:.1f}{tag}"


above = lambda m, s: m > 2 * s
below = lambda m, s: m < -2 * s

# ---- master table
out += ["## G on the neg group (clean success, points; ▲ > 2 sem above B0, ▼ below)", "",
        "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for arm in CORE:
    out.append(f"| {arm} | " + " | ".join(fmt(*gain(arm, c)[:3]) for c in CONDS) + " |")
out.append(f"\nB0 clean success (neg): {b0.loc[valid, 'neg_success_clean'].mean():.3f}\n")

for g in ("held",):
    out += [f"## G on the {g} group", "", "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for arm in CORE:
        out.append(f"| {arm} | " + " | ".join(fmt(*gain(arm, c, g)[:3]) for c in CONDS) + " |")
    out.append("")

out += ["## good_broken on neg (share of base-good episodes the arm breaks; B0 = 0)", "",
        "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for arm in CORE:
    out.append(f"| {arm} | " + " | ".join(
        f"{df[(df.arm == arm) & (df.cond == c)].neg_good_broken.mean():.3f}" if len(df[(df.arm == arm) & (df.cond == c)]) else "—"
        for c in CONDS) + " |")
out.append("")

# ---- decisions
out += ["## Decisions", "", "| arm | E1 dilution | E2 false labels | label dependence (native − placebo) | verdict |",
        "|---|---|---|---|---|"]
for arm in CORE:
    gp6, gp18, gf1, gf25, gn = (gain(arm, c)[:2] for c in ("pad6", "pad18", "flip0.1", "flip0.25", "native"))
    e1 = "breaks" if below(*gp6) else ("robust" if not below(*gp18) else "partial (pad18 < 0)")
    e2 = "breaks" if below(*gf1) else ("robust" if (not below(*gf25) and gf1[0] >= 0.5 * gn[0]) else "partial")
    ld = diff(arm, "native", arm, "placebo")
    dep = above(ld[0], ld[1])
    real = above(*gn) and dep
    out.append(f"| {arm} | {e1} | {e2} | {fmt(*ld)} → {'caused by labels' if dep else 'NOT label-dependent'} | "
               f"{'gain from labels' if real else ('gain, but not from labels' if above(*gn) else 'no native gain')} |")
out.append("")

out += ["## Controls (native)", ""]
for arm, rnd in (("SDPO_fresh_b25", "RAND_SDPO_fresh_b25"), ("NPO_b100_neg", "RAND_NPO_b100_neg")):
    m, s, n = diff(arm, "native", rnd, "native")
    out.append(f"- `{arm}` − `{rnd}`: {fmt(m, s, n)} → **{'beats the random edit' if above(m, s) else 'does NOT beat the random edit'}**; "
               f"`{rnd}` G = {fmt(*gain(rnd, 'native')[:3])}; edit norm "
               f"{df[(df.arm == arm) & (df.cond == 'native')].edit_norm.mean():.3f}")
for arm in CORE[:-1]:
    m, s, n = diff(arm, "native", "FILTER_rej", "native")
    out.append(f"- `{arm}` − `FILTER_rej` (neg): {fmt(m, s, n)}; held: {fmt(*diff(arm, 'native', 'FILTER_rej', 'native', 'held'))}")
out.append("")

out += ["## Strength (native): G vs behaviour change", "", "| arm | G neg | G held | bad Δ neg | hom_tv neg | edit norm |", "|---|---|---|---|---|---|"]
fam = sorted(df[(df.cond == "native") & df.arm.str.match(r"(SDPO_fresh|NPO|KTO)_b\d+")].arm.unique(),
             key=lambda x: (re.sub(r"_b\d+.*", "", x), int(re.search(r"_b(\d+)", x).group(1))))
for arm in fam:
    dd = df[(df.arm == arm) & (df.cond == "native")]
    out.append(f"| {arm} | {fmt(*gain(arm, 'native')[:3])} | {fmt(*gain(arm, 'native', 'held')[:3])} | "
               f"{fmt(*gain(arm, 'native', 'neg', 'bad')[:3])} | {dd.neg_hom_tv.mean():.3f} | {dd.edit_norm.mean():.3f} |")
out.append("")

out += ["## Coverage and recall (E4): G per group", "", "| arm | cond | neg | noneg | held |", "|---|---|---|---|---|"]
for arm in CORE:
    for c in ("native", "recall0.5", "cover0.25"):
        out.append(f"| {arm} | {c} | " + " | ".join(fmt(*gain(arm, c, g)[:3]) for g in GROUPS) + " |")
out.append("")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
