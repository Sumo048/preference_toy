"""robol round 1: scaling of each arm's gain with N_s, per label and budget rule; the price and floor of
trajectory labels against the headline oracle (approach window). Writes results/robol/RESULTS.md and
results/robol/scaling.png.

G = arm - in-task B0 (same process, same CRN), mean +- sem over seeds.
Price: for an arm whose best oracle gain g* > 0 (2 sem), the N_s at which each label first reaches
0.5 g* (log-linear interpolation between levels); price = N_traj / N_oracle (inf if traj never reaches it).
Floor: G_oracle(512) - G_traj(512).
"""
import glob, json, os, re
import numpy as np, pandas as pd

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ARMS = ["SDPO_fresh_b25", "SDPO_pin_b25", "NPO_b100_neg", "KTO_b100_neg"]
LABELS = ["approach", "causal_pc", "sym", "traj"]
NS = [16, 32, 64, 128, 256, 512]
GROUPS = ["panel", "noneg", "held"]
MET = ["success_clean", "bad", "collision", "jerk", "seg_clean", "good_broken", "mode_tv", "speed_ratio", "retained_disp"]

recs = []
for f in glob.glob(os.path.join(ROBO, "cells", "*", "s*", "*.json")):
    m = re.fullmatch(r"(R[12])_ns(\d+)_(\w+)", os.path.basename(os.path.dirname(os.path.dirname(f))))
    if not m:
        continue
    r = json.load(open(f))
    row = {"budget": m.group(1), "ns": int(m.group(2)), "labels": m.group(3), "arm": r["arm"], "seed": r["seed"],
           "steps": r["steps"], "rows": r["label_info"]["rows"]}
    for g in GROUPS:
        for k in MET:
            x, b = r.get(f"{g}_{k}", np.nan), r["b0"].get(f"{g}_{k}", np.nan)
            row[f"{g}_{k}"] = x
            row[f"d_{g}_{k}"] = x - b if k not in ("good_broken", "mode_tv", "retained_disp") else x
    recs.append(row)
df = pd.DataFrame(recs)
out = [f"# robol round 1: scaling with N_s ({len(df)} arm runs, seeds {sorted(df.seed.unique())})", "",
       "G = change in clean success vs the in-task B0, points; ▲/▼ > 2 sem from 0.", ""]


def agg(d, col):
    g = d.groupby(["budget", "labels", "arm", "ns"])[col]
    return g.mean(), g.sem(), g.count()


def fmt(m, s):
    if np.isnan(m):
        return "—"
    t = " ▲" if m > 2 * s else (" ▼" if m < -2 * s else "")
    return f"{100 * m:+.1f}±{100 * (s if not np.isnan(s) else 0):.1f}{t}"


def cross(ns, g, target):
    """first N_s where g reaches target (log-linear interpolation); inf if never."""
    for i, (n, v) in enumerate(zip(ns, g)):
        if v >= target:
            if i == 0:
                return float(n)
            n0, v0 = ns[i - 1], g[i - 1]
            w = (target - v0) / (v - v0)
            return float(np.exp(np.log(n0) + w * (np.log(n) - np.log(n0))))
    return float("inf")


price_rows = []
for budget in ("R1", "R2"):
    for grp in ("held", "panel", "noneg"):
        col = f"d_{grp}_success_clean"
        mu, se, n = agg(df, col)
        out += [f"## {budget}, {grp} group: G by N_s", "", "| arm | labels | " + " | ".join(map(str, NS)) + " |",
                "|---|---|" + "---|" * len(NS)]
        for arm in ARMS:
            for L in LABELS:
                cells = [fmt(mu.get((budget, L, arm, k), np.nan), se.get((budget, L, arm, k), np.nan)) for k in NS]
                out.append(f"| {arm} | {L} | " + " | ".join(cells) + " |")
            # price vs the headline oracle
            go = np.array([mu.get((budget, "approach", arm, k), np.nan) for k in NS])
            so = np.array([se.get((budget, "approach", arm, k), np.nan) for k in NS])
            gt = np.array([mu.get((budget, "traj", arm, k), np.nan) for k in NS])
            if np.all(np.isnan(go)):
                continue
            i = int(np.nanargmax(go))
            gstar = go[i]
            if gstar > 2 * so[i]:
                no, nt = cross(NS, go, 0.5 * gstar), cross(NS, gt, 0.5 * gstar)
                price = nt / no
            else:
                no = nt = price = float("nan")
            price_rows.append({"budget": budget, "group": grp, "arm": arm, "best_oracle_G": gstar, "at_ns": NS[i],
                               "N_oracle_50": no, "N_traj_50": nt, "price": price, "floor_512": go[-1] - gt[-1]})
        out.append("")

pr = pd.DataFrame(price_rows)
out += ["## Price and floor of trajectory labels (vs the approach-window oracle)", "",
        "price = N_traj / N_oracle to reach half the oracle's best gain (inf: trajectory labels never reach it; "
        "nan: the oracle itself has no gain). floor = G_oracle(512) - G_traj(512).", "",
        "| budget | group | arm | best oracle G (at N_s) | N_oracle@50% | N_traj@50% | price | floor |", "|---|---|---|---|---|---|---|---|"]
for _, r in pr.iterrows():
    out.append(f"| {r.budget} | {r.group} | {r.arm} | {100 * r.best_oracle_G:+.1f} ({r.at_ns}) | {r.N_oracle_50:.0f} | "
               f"{r.N_traj_50:.0f} | {r.price:.2f} | {100 * r.floor_512:+.1f} |")
out.append("")

out += ["## Collateral at N_s = 512 (held group): segment-clean change, good broken, mode TV, speed ratio, behaviour change", "",
        "| budget | arm | labels | G | Δseg_clean | good_broken | mode_tv | speed | retained_disp |", "|---|---|---|---|---|---|---|---|---|"]
d5 = df[df.ns == 512]
for budget in ("R1", "R2"):
    for arm in ARMS:
        for L in LABELS:
            x = d5[(d5.budget == budget) & (d5.arm == arm) & (d5.labels == L)]
            if not len(x):
                continue
            out.append(f"| {budget} | {arm} | {L} | {fmt(x.d_held_success_clean.mean(), x.d_held_success_clean.sem())} | "
                       f"{100 * x.d_held_seg_clean.mean():+.1f} | {x.d_held_good_broken.mean():.3f} | {x.d_held_mode_tv.mean():.3f} | "
                       f"{x.held_speed_ratio.mean():.3f} | {x.d_held_retained_disp.mean():.3f} |")
out.append("")

os.makedirs("results/robol", exist_ok=True)
open("results/robol/RESULTS.md", "w").write("\n".join(out) + "\n")
pr.to_csv("results/robol/price.csv", index=False)
df.to_csv("results/robol/cells.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, len(ARMS), figsize=(4 * len(ARMS), 7), sharex=True)
col = {"approach": "#2a6fdb", "causal_pc": "#6aa84f", "sym": "#9b59b6", "traj": "#d9534f"}
mu, se, _ = agg(df, "d_held_success_clean")
for i, budget in enumerate(("R1", "R2")):
    for j, arm in enumerate(ARMS):
        ax = axes[i, j]
        for L in LABELS:
            y = np.array([mu.get((budget, L, arm, k), np.nan) for k in NS]) * 100
            e = np.array([se.get((budget, L, arm, k), np.nan) for k in NS]) * 100
            ax.errorbar(NS, y, yerr=2 * e, label=L, color=col[L], marker="o", ms=3, capsize=2, lw=1.5)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{arm} ({budget})", fontsize=9)
        if j == 0:
            ax.set_ylabel("G held (pts clean success)")
        if i == 1:
            ax.set_xlabel("N_s setups with negatives")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/robol/scaling.png", dpi=130)
print("\n".join(out))
