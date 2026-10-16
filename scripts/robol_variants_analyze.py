"""Scaling curves for the three SDPO variants vs the plain runs of the k-sweep.

G = held clean success - in-task B0 (points). Writes results/robol/variants_{iter,shift,own}.png and
results/robol/VARIANTS.md (best G per configuration and N_s, next to the plain run's best).
"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
recs = []
for f in glob.glob(os.path.join(ROBO, "cells_v", "*", "*_ns*", "s*", "*.json")):
    r = json.load(open(f))
    b = r["b0"]
    if r["mode"] == "iter":
        cfg = f"round {r['round']}"
    elif r["mode"] in ("shift", "shiftneg"):
        cfg = f"S={r['n_shift']}"
    else:
        cfg = f"K={r['n_win']}"
    arm = r.get("arm", "own")
    recs.append({"mode": r["mode"], "arm": "SDPO_own_b25" if r["mode"] == "own" else arm, "cfg": cfg,
                 "labels": r["labels"], "ns": r["ns"], "k": r["k"], "seed": r["seed"],
                 "G": 100 * (r["held_success_clean"] - b["held_success_clean"]), "steps": r["total_steps"]})
# plain runs from the k-sweep
for f in glob.glob(os.path.join(ROBO, "cells", "K*_ns*_*", "s*", "SDPO_*_b25.json")):
    m = re.fullmatch(r"K([\d.]+)_ns(\d+)_(traj|approach)", os.path.basename(os.path.dirname(os.path.dirname(f))))
    if not m:
        continue
    r = json.load(open(f))
    b = r["b0"]
    recs.append({"mode": "plain", "arm": r["arm"], "cfg": "plain", "labels": m.group(3), "ns": int(m.group(2)),
                 "k": float(m.group(1)), "seed": r["seed"], "G": 100 * (r["held_success_clean"] - b["held_success_clean"]),
                 "steps": r["steps"]})
df = pd.DataFrame(recs)
NS = [16, 32, 64, 128, 256, 512]
out = [f"# SDPO variants ({len(df[df['mode'] != 'plain'])} variant runs + plain k-sweep)", "",
       "Best G over k (points, held clean success vs in-task B0), mean±sem over seeds; plain = the k-sweep run.", ""]


def best(d):
    g = d.groupby("k").G
    mu, se = g.mean(), g.sem()
    if not len(mu):
        return None
    k = mu.idxmax()
    return k, mu[k], se.get(k, np.nan), int(d[d.k == k].steps.median())


def figure(mode, arms, cfgs, fname):
    rows = [(a, L) for a in arms for L in ("traj", "approach")]
    fig, axes = plt.subplots(len(rows), len(NS), figsize=(2.9 * len(NS), 2.5 * len(rows)), sharex=True, squeeze=False)
    cmap = plt.get_cmap("plasma")
    for i, (a, L) in enumerate(rows):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            base_arm = "SDPO_pin_b25" if a.startswith("SDPO_own") else a
            pl = df[(df["mode"] == "plain") & (df.arm == base_arm) & (df.labels == L) & (df.ns == n)]
            g = pl.groupby("k").G
            ax.errorbar(g.mean().index, g.mean(), yerr=2 * g.sem().fillna(0), color="0.55", lw=1.4, marker="o", ms=2,
                        label=f"plain {base_arm.split('_')[1]}")
            for c_i, c in enumerate(cfgs):
                d = df[(df["mode"] == mode) & (df.arm == a) & (df.labels == L) & (df.ns == n) & (df.cfg == c)]
                if not len(d):
                    continue
                g = d.groupby("k").G
                ax.errorbar(g.mean().index, g.mean(), yerr=2 * g.sem().fillna(0), color=cmap(0.15 + 0.7 * c_i / max(len(cfgs) - 1, 1)),
                            lw=1.4, marker="o", ms=2, label=c)
            ax.axhline(0, color="0.7", lw=0.7)
            ax.set_xscale("log", base=2)
            ax.set_ylim(-30, 16)
            if i == 0:
                ax.set_title(f"N_s = {n}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{a.replace('_b25', '')}\n{L}\nΔ clean (pts)", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("k (steps per row, per round)" if mode == "iter" else "k (steps per row)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)


figure("iter", ["SDPO_pin_b25", "SDPO_fresh_b25"], ["round 1", "round 2", "round 3"], "results/robol/variants_iter.png")
figure("shift", ["SDPO_pin_b25", "SDPO_fresh_b25", "SDPO_own_b25_K8"], ["S=2", "S=4", "S=8"], "results/robol/variants_shift.png")
figure("shiftneg", ["SDPO_pin_b25", "SDPO_fresh_b25", "SDPO_own_b25_K8"], ["S=2", "S=4", "S=8"], "results/robol/variants_shiftneg.png")
figure("own", ["SDPO_own_b25"], ["K=1", "K=8"], "results/robol/variants_own.png")

for mode, arms, cfgs in (("iter", ["SDPO_pin_b25", "SDPO_fresh_b25"], ["round 1", "round 2", "round 3"]),
                         ("shift", ["SDPO_pin_b25", "SDPO_fresh_b25", "SDPO_own_b25_K8"], ["S=2", "S=4", "S=8"]),
                         ("shiftneg", ["SDPO_pin_b25", "SDPO_fresh_b25", "SDPO_own_b25_K8"], ["S=2", "S=4", "S=8"]),
                         ("own", ["SDPO_own_b25"], ["K=1", "K=8"])):
    for a in arms:
        base_arm = "SDPO_pin_b25" if a.startswith("SDPO_own") else a
        for L in ("traj", "approach"):
            out += [f"## {mode}: {a}, {L}", "", "| N_s | plain best | " + " | ".join(f"{c} best" for c in cfgs) + " |",
                    "|---|---|" + "---|" * len(cfgs)]
            for n in NS:
                cells = []
                for c, d in [("plain", df[(df["mode"] == "plain") & (df.arm == base_arm) & (df.labels == L) & (df.ns == n)])] + \
                        [(c, df[(df["mode"] == mode) & (df.arm == a) & (df.labels == L) & (df.ns == n) & (df.cfg == c)]) for c in cfgs]:
                    b = best(d)
                    cells.append("—" if b is None else f"{b[1]:+.1f}±{b[2]:.1f} (k={b[0]:g}, {b[3]} steps)")
                out.append(f"| {n} | " + " | ".join(cells) + " |")
            out.append("")
open("results/robol/VARIANTS.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
