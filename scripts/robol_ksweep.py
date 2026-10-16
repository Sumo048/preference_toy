"""robol k-sweep: performance vs steps per labelled row (k), one curve per dataset size N_s.

Reads cells/K<k>_ns<N>_<labels>/s*/<arm>.json. G = held clean success - in-task B0 (points).
Writes results/robol/ksweep.png (G vs k), results/robol/ksweep_disp.png (behaviour change vs k),
results/robol/KSWEEP.md (tables, best k per N_s).
"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ARMS = ["SDPO_fresh_b25", "SDPO_pin_b25", "NPO_b100_neg", "KTO_b100_neg"]
LABELS = ["traj", "approach"]
recs = []
for f in glob.glob(os.path.join(ROBO, "cells", "K*_ns*_*", "s*", "*.json")):
    m = re.fullmatch(r"K([\d.]+)_ns(\d+)_(traj|approach)", os.path.basename(os.path.dirname(os.path.dirname(f))))
    if not m:
        continue
    r = json.load(open(f))
    if r["arm"] not in ARMS:
        continue
    b = r["b0"]
    recs.append({"k": float(m.group(1)), "ns": int(m.group(2)), "labels": m.group(3), "arm": r["arm"], "seed": r["seed"],
                 "rows": r["label_info"]["rows"], "steps": r["steps"],
                 "G": 100 * (r["held_success_clean"] - b["held_success_clean"]),
                 "dbad": 100 * (r["held_bad"] - b["held_bad"]),
                 "disp": r["held_retained_disp"], "broken": r["held_good_broken"]})
df = pd.DataFrame(recs)
NS = sorted(df.ns.unique())
KS = sorted(df.k.unique())
cmap = plt.get_cmap("viridis")
col = {n: cmap(i / max(len(NS) - 1, 1)) for i, n in enumerate(NS)}


def panel_plot(metric, ylabel, fname, zero=True):
    fig, axes = plt.subplots(len(LABELS), len(ARMS), figsize=(4.2 * len(ARMS), 3.6 * len(LABELS)), sharex=True)
    for i, L in enumerate(LABELS):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            d = df[(df.arm == arm) & (df.labels == L)]
            for n in NS:
                g = d[d.ns == n].groupby("k")[metric]
                mu, se = g.mean(), g.sem()
                if not len(mu):
                    continue
                rows = int(d[d.ns == n].rows.median())
                ax.errorbar(mu.index, mu.values, yerr=2 * se.fillna(0).values, color=col[n], marker="o", ms=3, lw=1.4,
                            capsize=2, label=f"N_s={n} ({rows} rows)")
            if zero:
                ax.axhline(0, color="grey", lw=0.8)
            ax.axvline(6, color="grey", lw=0.6, ls=":")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{arm} — {L} labels", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(LABELS) - 1:
                ax.set_xlabel("k = steps per labelled row (dotted: R1's k = 6)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


os.makedirs("results/robol", exist_ok=True)
panel_plot("G", "Δ clean success, held (pts)", "results/robol/ksweep.png")
panel_plot("disp", "behaviour change (RMS chunk shift)", "results/robol/ksweep_disp.png", zero=False)

out = [f"# robol k-sweep ({len(df)} arm runs, seeds {sorted(int(x) for x in df.seed.unique())})", "",
       "G = Δ clean success on held-out courses vs in-task B0 (points), mean±sem; ▲/▼ > 2 sem.", ""]
for arm in ARMS:
    for L in LABELS:
        d = df[(df.arm == arm) & (df.labels == L)]
        out += [f"## {arm}, {L}", "", "| N_s (rows) | " + " | ".join(f"k={k:g}" for k in KS) + " | best k |", "|---|" + "---|" * (len(KS) + 1)]
        for n in NS:
            dn = d[d.ns == n]
            cells, best = [], (None, -1e9)
            for k in KS:
                x = dn[dn.k == k].G
                if not len(x):
                    cells.append("—"); continue
                mu, se = x.mean(), x.sem() if len(x) > 1 else 0
                t = " ▲" if mu > 2 * se else (" ▼" if mu < -2 * se else "")
                cells.append(f"{mu:+.1f}±{se:.1f}{t}")
                if mu > best[1]:
                    best = (k, mu)
            out.append(f"| {n} ({int(dn.rows.median()) if len(dn) else 0}) | " + " | ".join(cells) + f" | {best[0]:g} ({best[1]:+.1f}) |")
        out.append("")
open("results/robol/KSWEEP.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---- the same curves against TOTAL steps (k x rows): is the optimum a fixed step count rather than a per-row dose?
fig, axes = plt.subplots(len(LABELS), len(ARMS), figsize=(4.2 * len(ARMS), 3.6 * len(LABELS)), sharex=True)
best_rows = []
for i, L in enumerate(LABELS):
    for j, arm in enumerate(ARMS):
        ax = axes[i, j]
        d = df[(df.arm == arm) & (df.labels == L)]
        for n in NS:
            g = d[d.ns == n].groupby("k").agg(G=("G", "mean"), se=("G", "sem"), steps=("steps", "median"))
            if not len(g):
                continue
            ax.errorbar(g.steps, g.G, yerr=2 * g.se.fillna(0), color=col[n], marker="o", ms=3, lw=1.4, capsize=2,
                        label=f"N_s={n}")
            b = g.G.idxmax()
            best_rows.append({"arm": arm, "labels": L, "ns": n, "best_k": b, "best_steps": int(g.loc[b, "steps"]),
                              "best_G": g.loc[b, "G"], "at_edge": b == min(g.index)})
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log", base=10)
        ax.set_title(f"{arm} — {L} labels", fontsize=9)
        if j == 0:
            ax.set_ylabel("Δ clean success, held (pts)")
        if i == len(LABELS) - 1:
            ax.set_xlabel("total gradient steps (= k x labelled rows)")
        if i == 0 and j == 0:
            ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/robol/ksweep_steps.png", dpi=130)
bt = pd.DataFrame(best_rows)
with open("results/robol/KSWEEP.md", "a") as fh:
    fh.write("\n## Best dose per (arm, labels, N_s): k and total steps (at_edge: best k is the smallest tested)\n\n")
    fh.write(bt.to_string(index=False) + "\n")
print(bt.to_string(index=False))
