"""robol_ksweep.png, robol_segments.png, robol_drift.png from results/robol/sdpo_study.json (5-seed means; derived from the
same cells as KSWEEP.md / SEGMENTS.md / DRIFT.md)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from robol_style import *
import numpy as np

D = json.load(open("/home/soumojib/repos/toypref-report/results/robol/sdpo_study.json"))
NICE = {"NPO_b100_neg": "NPO", "KTO_b100_neg": "KTO", "SDPO_fresh_b25": "SDPO fresh", "SDPO_pin_b25": "SDPO pinned"}
COL = {"NPO_b100_neg": CAT[3], "KTO_b100_neg": CAT[6], "SDPO_fresh_b25": CAT[0], "SDPO_pin_b25": CAT[1]}

# ---------------------------------------------------------------- k-sweep
fig, axs = plt.subplots(2, 2, figsize=(10.8, 6.4), sharex=True, sharey=False)
for r, ns in enumerate(["512", "64"]):
    for c, lab in enumerate(["approach", "traj"]):
        ax = axs[r, c]
        ax.axhline(0, color=SEC, lw=0.6)
        ends = []
        for arm in NICE:
            d = D["ksweep"][arm][lab][ns]
            ks = sorted(d, key=float)
            x = np.array([float(k) for k in ks]); m = np.array([d[k][0] for k in ks]); se = np.array([d[k][1] for k in ks])
            ax.errorbar(x, m, yerr=2 * se, color=COL[arm], lw=1.1, marker="o", ms=3, capsize=1.5, elinewidth=0.7, clip_on=False)
            ends.append((arm, m[-1]))
        ys = spread([max(e[1], -86) for e in ends], 9)
        for (arm, _), y in zip(ends, ys):
            ax.text(18.5, y, NICE[arm], color=COL[arm], va="center", fontsize=7.5, clip_on=False)
        ax.set_xscale("log", base=2)
        ax.set_xlim(2 ** -5.3, 2 ** 4.2); ax.set_ylim(-85, 18)
        ax.set_title(f"{'approach-window (oracle)' if lab == 'approach' else 'trajectory'} labels, N_s = {ns}", loc="left", color=SEC)
        ax.set_ylabel("G: clean success vs B0 (points)")
        if r == 1: ax.set_xlabel("k: gradient steps per labelled row")
        ax.set_xticks([2 ** i for i in range(-5, 5, 1)]); ax.set_xticklabels(["1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16"][:10], fontsize=6.5)
        ax.grid(axis="y", color=GRID, lw=0.5)
fig.subplots_adjust(left=0.07, right=0.9, top=0.96, bottom=0.08, wspace=0.42, hspace=0.12)
fig.savefig(f"{OUTDIR}/robol_ksweep.png"); plt.close(fig)

# ---------------------------------------------------------------- segments
fig, axs = plt.subplots(2, 2, figsize=(10.8, 6.4), sharex=True, sharey=False)
SEGC = {("SDPO_fresh_b25", "1/8"): (CAT[0], "-"), ("SDPO_fresh_b25", "1/32"): (CAT[0], "--"),
        ("SDPO_pin_b25", "1/8"): (CAT[1], "-"), ("SDPO_pin_b25", "1/32"): (CAT[1], "--")}
for r, ns in enumerate(["512", "64"]):
    for c, lab in enumerate(["approach", "traj"]):
        ax = axs[r, c]
        ax.axhline(0, color=SEC, lw=0.6)
        ends = []
        for (arm, kq), (col, ls) in SEGC.items():
            d = D["segments"][arm][lab][ns][kq]
            Ss = sorted(d, key=int)
            x = np.array([int(k) for k in Ss]); m = np.array([d[k][0] for k in Ss]); se = np.array([d[k][1] for k in Ss])
            ax.errorbar(x, m, yerr=2 * se, color=col, ls=ls, lw=1.1, marker="o", ms=3, capsize=1.5, elinewidth=0.7)
            ends.append((arm, kq, m[-1]))
        ys = spread([e[2] for e in ends], 9)
        for (arm, kq, _), y in zip(ends, ys):
            ax.text(76, y, f"{'pinned' if 'pin' in arm else 'fresh'}, {kq}", color=SEGC[(arm, kq)][0], va="center", fontsize=7, clip_on=False)
        ax.set_xscale("log", base=2); ax.set_xlim(0.85, 70); ax.set_ylim(-85, 20)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        ax.set_title(f"{'approach-window (oracle)' if lab == 'approach' else 'trajectory'} labels, N_s = {ns}", loc="left", color=SEC)
        ax.set_ylabel("G: clean success vs B0 (points)")
        if r == 1: ax.set_xlabel("S: segments (negatives re-collected S-1 times)")
        ax.grid(axis="y", color=GRID, lw=0.5)
fig.subplots_adjust(left=0.07, right=0.9, top=0.96, bottom=0.1, wspace=0.42, hspace=0.12)
fig.text(0.07, 0.015, "line label = steps per labelled row per segment, k_seg (1/8 solid, 1/32 dashed); S = 1 is the plain run", fontsize=7, color=SEC)
fig.savefig(f"{OUTDIR}/robol_segments.png"); plt.close(fig)

# ---------------------------------------------------------------- drift
VAR = [("SDPO_fresh_b25", "fresh", CAT[0], "-"), ("SDPO_pin_b25", "pinned", CAT[1], "-"),
       ("SDPO_fresh_b25_W", "fresh, winner only", CAT[0], ":"), ("SDPO_pin_b25_W", "pinned, winner only", CAT[1], ":"),
       ("SDPO_fresh_b25_L", "fresh, loser only", CAT[0], "--"), ("SDPO_pin_b25_L", "pinned, loser only", CAT[1], "--")]
import glob, collections
RAW = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("/grogu/datasets/rollout/soumojib/toypref/runs/robol/drift/traj_ns*/s*/*.json"):
    r = json.load(open(f))
    RAW[(r["arm"], str(r["ns"]))][r["steps"]].append((100 * (r["held_success_clean"] - r["b0"]["held_success_clean"]), r["held_energy"]))
def raw(arm, ns, key):
    d = RAW[(arm, ns)]; st = sorted(d)
    v = [np.array([x[0 if key == "G" else 1] for x in d[k]]) for k in st]
    return np.array(st), np.array([a.mean() for a in v]), np.array([a.std(ddof=1) / np.sqrt(len(a)) for a in v]), min(len(a) for a in v)
fig, axs = plt.subplots(2, 2, figsize=(10.8, 6.4), sharex=True)
for c, ns in enumerate(["64", "512"]):
    for r, (key, yl) in enumerate([("energy", "energy distance to base, held-out states (0 = none)"), ("G", "G: clean success vs B0 (points)")]):
        ax = axs[r, c]
        ends = []
        for arm, nm, col, ls in VAR:
            x, m, se, nmin = raw(arm, ns, key)
            ax.errorbar(x, m, yerr=2 * se, color=col, ls=ls, lw=1.1, marker="o", ms=2.5, capsize=1.2, elinewidth=0.6)
            ends.append((nm, col, m[-1]))
        if key == "energy":
            ax.set_yscale("symlog", linthresh=0.003); ax.set_ylim(-0.02, 3); ax.axhline(0, color=SEC, lw=0.5)
        else:
            ax.axhline(0, color=SEC, lw=0.6); ax.set_ylim(-85, 20)
        ax.set_xscale("log"); ax.set_xlim(40, 6500)
        ax.grid(axis="y", color=GRID, lw=0.5)
        if c == 0: ax.set_ylabel(yl)
        if r == 0: ax.set_title(f"trajectory labels, N_s = {ns}", loc="left", color=SEC)
        if r == 1: ax.set_xlabel("gradient steps")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=col, ls=ls, lw=1.3) for _, _, col, ls in VAR], [nm for _, nm, _, _ in VAR],
           loc="lower center", ncol=len(VAR), fontsize=7.5, handlelength=3)
fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.14, wspace=0.18, hspace=0.12)
fig.savefig(f"{OUTDIR}/robol_drift.png"); plt.close(fig)
print("ok")
