"""Segment scaling for SDPO with re-collected negatives ("shiftneg"): gain vs number of segments S at a fixed
number of steps per segment (k_seg x labelled rows per segment).

S = 1 is the plain run (k-sweep, k = k_seg); S = 2..64 are shiftneg runs with k = S x k_seg. Each segment ends
with a new collection of labelled negatives from the current policy, so S also counts labelled collections.
Writes results/robol/segments.png and results/robol/SEGMENTS.md.
"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ARMS = ["SDPO_fresh_b25", "SDPO_pin_b25"]
LABELS = ["traj", "approach"]
NS = [64, 128, 512]
KSEG = [1 / 32, 1 / 16, 1 / 8, 1 / 4]
recs = []
for f in glob.glob(os.path.join(ROBO, "cells_v", "shiftneg", "*_ns*", "s*", "*.json")):
    r = json.load(open(f))
    if r.get("arm") not in ARMS or r["ns"] not in NS:
        continue
    S = r["n_shift"]
    kseg = r["k"] / S
    recs.append({"arm": r["arm"], "labels": r["labels"], "ns": r["ns"], "S": S, "kseg": kseg, "seed": r["seed"],
                 "G": 100 * (r["held_success_clean"] - r["b0"]["held_success_clean"]),
                 "bad": 100 * (r["held_bad"] - r["b0"]["held_bad"]), "steps": r["total_steps"],
                 "rows_last": (r.get("refresh_rows") or [r["rows"]])[-1], "rows0": r["rows"]})
for f in glob.glob(os.path.join(ROBO, "cells", "K*_ns*_*", "s*", "SDPO_*_b25.json")):
    m = re.fullmatch(r"K([\d.]+)_ns(\d+)_(traj|approach)", os.path.basename(os.path.dirname(os.path.dirname(f))))
    if not m or int(m.group(2)) not in NS:
        continue
    r = json.load(open(f))
    recs.append({"arm": r["arm"], "labels": m.group(3), "ns": int(m.group(2)), "S": 1, "kseg": float(m.group(1)),
                 "seed": r["seed"], "G": 100 * (r["held_success_clean"] - r["b0"]["held_success_clean"]),
                 "bad": 100 * (r["held_bad"] - r["b0"]["held_bad"]), "steps": r["steps"], "rows_last": r["label_info"]["rows"],
                 "rows0": r["label_info"]["rows"]})
df = pd.DataFrame(recs)
df = df[df.kseg.apply(lambda x: any(abs(x - k) < 1e-9 for k in KSEG))]
df["kseg"] = df.kseg.apply(lambda x: min(KSEG, key=lambda k: abs(k - x)))

rows = [(a, L) for a in ARMS for L in LABELS]
fig, axes = plt.subplots(len(rows), len(NS), figsize=(4.0 * len(NS), 3.0 * len(rows)), sharex=True, squeeze=False)
cmap = plt.get_cmap("viridis")
for i, (a, L) in enumerate(rows):
    for j, n in enumerate(NS):
        ax = axes[i, j]
        for c, ks in enumerate(KSEG):
            d = df[(df.arm == a) & (df.labels == L) & (df.ns == n) & (df.kseg == ks)]
            g = d.groupby("S").G
            if not len(g.mean()):
                continue
            ax.errorbar(g.mean().index, g.mean(), yerr=2 * g.sem().fillna(0), color=cmap(c / (len(KSEG) - 1)), marker="o",
                        ms=3, lw=1.5, capsize=2, label=f"k_seg = 1/{round(1 / ks)}")
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xscale("log", base=2)
        ax.set_ylim(-25, 22)
        if i == 0:
            ax.set_title(f"N_s = {n}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{a.replace('_b25', '')}, {L}\nΔ clean success (pts)", fontsize=9)
        if i == len(rows) - 1:
            ax.set_xlabel("segments S (= labelled collections)")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/robol/segments.png", dpi=130)

out = [f"# Segment scaling, SDPO with re-collected negatives ({len(df)} runs)", "",
       "G = Δ clean success on held-out courses vs in-task B0 (points), mean±sem over seeds. Each cell: mean G at that S "
       "(steps per segment fixed at k_seg x labelled rows). S = 1 is the plain run.", ""]
SS = sorted(df.S.unique())
for a in ARMS:
    for L in LABELS:
        for n in NS:
            out += [f"## {a}, {L}, N_s = {n}", "", "| k_seg | " + " | ".join(f"S={s}" for s in SS) + " | best |",
                    "|---|" + "---|" * (len(SS) + 1)]
            for ks in KSEG:
                d = df[(df.arm == a) & (df.labels == L) & (df.ns == n) & (df.kseg == ks)]
                g = d.groupby("S").G
                mu, se = g.mean(), g.sem()
                cells = [f"{mu[s]:+.1f}±{se.get(s, 0) if not np.isnan(se.get(s, np.nan)) else 0:.1f}" if s in mu.index else "—" for s in SS]
                b = f"S={mu.idxmax()} ({mu.max():+.1f})" if len(mu) else "—"
                out.append(f"| 1/{round(1 / ks)} | " + " | ".join(cells) + f" | {b} |")
            out.append("")
open("results/robol/SEGMENTS.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
