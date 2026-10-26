"""robo_bars.png: per-arm gain G in clean success (points, neg group) vs the same-seed B0, valid seeds only.
Source: results/robo/cells_full_st5000_m64.csv (branch robo-toy; read from the toypref-robo worktree, not modified).
Valid seeds = frozen base clean success >= 0.5 on the neg group (seed 3 diverged)."""
import csv, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SRC = os.path.expanduser("~/repos/toypref-robo/results/robo/cells_full_st5000_m64.csv")
OUT = os.path.expanduser("~/repos/toypref-report/artifact/sdpo_toys/fig/robo_bars.png")
rows = list(csv.DictReader(open(SRC)))
b0 = {r["seed"]: float(r["neg_success_clean"]) for r in rows if r["arm"] == "B0_frozen"}
valid = sorted(s for s, v in b0.items() if v >= 0.5)
LAB = {"SDPO_fresh_b25": "SDPO fresh (beta 25)", "SDPO_pin_b25": "SDPO pin (beta 25)", "SDPO_hop_pin_b25": "SDPO hop-pin (beta 25)",
       "SDPO_fresh_b1": "SDPO fresh (beta 1)", "SDPO_pin_b1": "SDPO pin (beta 1)", "SDPO_hop_pin_b1": "SDPO hop-pin (beta 1)",
       "NPO_b10_neg": "NPO (beta 10)", "NPO_b100_neg": "NPO (beta 100)", "KTO_b10_neg": "KTO (beta 10)", "KTO_b100_neg": "KTO (beta 100)",
       "R_ours_neg": "reroute (ours)", "R_hop_bsup_neg": "hop reroute", "R_hop_bsup_neg_AL": "hop reroute + anchor"}
KEEP = ["SDPO_fresh_b25", "SDPO_pin_b25", "SDPO_hop_pin_b25", "NPO_b100_neg", "KTO_b100_neg", "R_hop_bsup_neg_AL", "R_hop_bsup_neg", "R_ours_neg"]  # native-dose arms (beta 25/100); beta 1/10 arms collapse (-60..-85) and are left out
G = {}
for r in rows:
    if r["arm"] in KEEP and r["seed"] in valid:
        G.setdefault(r["arm"], []).append(100 * (float(r["neg_success_clean"]) - b0[r["seed"]]))
arms = sorted(G, key=lambda a: np.mean(G[a]))
BLUE, TX, T2, BG = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=BG)
ax.set_facecolor(BG)
for i, a in enumerate(arms):
    v = np.array(G[a]); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    ax.scatter(v, np.full(len(v), i), s=12, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
    ax.errorbar(m, i, xerr=se, fmt="o", color=BLUE, ms=6, capsize=3, lw=1.2, zorder=3)
ax.axvline(0, color=TX, lw=0.8, zorder=1)
ax.set_yticks(range(len(arms))); ax.set_yticklabels([LAB[a] for a in arms], fontsize=9, color=TX)
ax.set_xlabel(f"gain in clean success on negative setups vs B0 (points)\nseeds {','.join(valid)}; dot = mean +/- s.e.m., faint dots = single seeds", fontsize=8.5, color=T2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_color(T2)
ax.tick_params(length=2.5, width=0.6, colors=T2)
fig.tight_layout(); fig.savefig(OUT, dpi=200, facecolor=BG)
for a in arms: print(a, np.round(G[a], 2), round(float(np.mean(G[a])), 2))
print("valid", valid)
