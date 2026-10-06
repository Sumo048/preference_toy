"""Figures from results/summary.csv and the saved sample files.

figures/heatmap_<metric>.png   arms x (n_forget, n_neg), seed-mean
figures/curves_<metric>.png    metric vs n_neg, one panel per n_forget, key arms
figures/tradeoff.png           forgetting-strength vs collateral damage per cell
figures/samples_<cell>.png     decoded samples for key arms, coloured by role
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toypref.arms import ARMS
from toypref.config import DEFAULT, ROOT, RUNS
from toypref.editset import forget_modes
from toypref.mixture import OFF, assign_modes

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RETAIN_C, FORGET_C, OFF_C = SERIES[0], SERIES[1], "#a3a29c"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})

KEY_ARMS = [  # one representative per family; <= 8 so hues are fixed, never cycled
    "B1_filtered", "B3_npo_b10", "B5_kto_b10", "DPO_cpl_b25",
    "R_ours_both", "R_spread_nhood_both", "R_spread_nhood_neg", "RM_bon8",
]
HEAT_METRICS = {
    "neg_mass_ratio": ("forget-mode mass / pi_ref's (lower = more forgetting)", "Blues_r"),
    "retain_tv": ("TV between retained-mode distributions, arm vs pi_ref", "Blues"),
    "retain_noise_flip": ("share of pi_ref-retain noises that change mode", "Blues"),
    "mass_off": ("off-mode mass", "Blues"),
    "disp_retain_mean": ("mean action displacement on retain noises", "Blues"),
    "logp_delta_forget_mean": ("delta log p, held-out forget-mode samples", "RdBu"),
    "logp_delta_retain_mean": ("delta log p, held-out retain-mode samples", "RdBu"),
    "tv_to_ideal": ("TV to the ideal (retained modes, uniform)", "Blues"),
}


def heatmap(summ, metric, title, cmap, out):
    summ = summ.copy()
    summ["col"] = summ.n_forget.astype(str) + "|" + summ.n_neg.astype(str)
    piv = summ.pivot(index="arm", columns="col", values=f"{metric}_mean")
    cols = sorted(piv.columns, key=lambda c: tuple(int(x) for x in c.split("|")))
    arms = [a for a in ARMS if a in piv.index]
    piv = piv.loc[arms, cols]
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 3.2, 0.2 * len(arms) + 1.6))
    v = piv.values.astype(float)
    if cmap == "RdBu":
        lim = np.nanpercentile(np.abs(v), 95) or 1.0
        im = ax.imshow(v, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(v, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(arms)), arms, fontsize=7)
    ax.set_xticks(range(len(cols)), [c.split("|")[1] for c in cols], fontsize=7)
    ks = [c.split("|")[0] for c in cols]
    for i in range(1, len(cols)):
        if ks[i] != ks[i - 1]:
            ax.axvline(i - 0.5, color=SURF, lw=2)
    for k in sorted(set(ks), key=int):
        idx = [i for i, kk in enumerate(ks) if kk == k]
        ax.text(np.mean(idx), -1.2, f"{k} forgotten", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("logged negatives per forgotten mode")
    ax.grid(False)
    ax.set_title(title, fontsize=10, pad=18, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def curves(summ, metric, ylabel, out):
    ks = sorted(summ.n_forget.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        for i, arm in enumerate(KEY_ARMS):
            d = summ[(summ.arm == arm) & (summ.n_forget == k)].sort_values("n_neg")
            if d.empty:
                continue
            ax.errorbar(d.n_neg, d[f"{metric}_mean"], yerr=d[f"{metric}_sem"], color=SERIES[i],
                        marker="o", ms=4, capsize=0, label=arm, lw=1.6)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{k} forgotten of 8", fontsize=9, loc="left")
        ax.set_xlabel("negatives per forgotten mode")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def tradeoff(summ, out, x="retain_tv", y="neg_mass_ratio"):
    ks = sorted(summ.n_forget.unique())
    ns = [n for n in (1, 8, 32) if n in set(summ.n_neg)]
    fig, axes = plt.subplots(len(ns), len(ks), figsize=(3.0 * len(ks), 2.8 * len(ns)), squeeze=False)
    fam_c = {"reference": INK2, "baseline": SERIES[1]}
    for r, n in enumerate(ns):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            d = summ[(summ.n_forget == k) & (summ.n_neg == n)]
            for _, row in d.iterrows():
                fam = row.family
                col = fam_c.get(fam, SERIES[0])
                ax.scatter(row[f"{x}_mean"], row[f"{y}_mean"], s=14, color=col, edgecolor=SURF, lw=0.6, zorder=3)
            for arm in ("B0_frozen", "B1_filtered", "ORACLE_retain_bc", "R_ours_both", "R_spread_nhood_neg", "B3_npo_b10"):
                q = d[d.arm == arm]
                if len(q):
                    ax.annotate(arm.replace("_", " "), (q[f"{x}_mean"].iloc[0], q[f"{y}_mean"].iloc[0]),
                                fontsize=6, color=INK2, xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"{k} forgotten, {n} neg/mode", fontsize=8, loc="left")
            if r == len(ns) - 1:
                ax.set_xlabel("retain TV vs pi_ref (collateral)")
            if c == 0:
                ax.set_ylabel("forget mass ratio")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in
         (("reference", INK2), ("baseline", SERIES[1]), ("rerouting", SERIES[0]))]
    axes[0, -1].legend(handles=h, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def trajectories(probes, out, k=2, ns=(1, 8, 32)):
    """Seed-mean probe path of each key arm: collateral (x) vs forgetting (y) over training."""
    ns = [n for n in ns if n in set(probes.n_neg)]
    if not ns:
        return
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.2), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns):
        d = probes[(probes.n_forget == k) & (probes.n_neg == n) & (probes["round"] == 1)]
        for i, arm in enumerate(KEY_ARMS):
            q = d[d.arm == arm].groupby("step")[["retain_tv", "neg_mass_ratio"]].mean().reset_index()
            if q.empty:
                continue
            ax.plot(q.retain_tv, q.neg_mass_ratio, color=SERIES[i], lw=1.4, marker="o", ms=2.5, label=arm)
            ax.scatter(q.retain_tv.iloc[-1:], q.neg_mass_ratio.iloc[-1:], s=30, color=SERIES[i], edgecolor=SURF, zorder=4)
        ax.set_title(f"{k} forgotten, {n} neg/mode", fontsize=9, loc="left")
        ax.set_xlabel("retain TV vs pi_ref")
    axes[0, 0].set_ylabel("forget mass ratio")
    axes[0, -1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("training paths (dot = final step)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def samples(cell_dir, out, arms):
    name = os.path.basename(cell_dir)
    s, k, n = (int(p[1:]) for p in name.split("_"))
    mix = DEFAULT.mixture
    F = forget_modes(mix, s, k)
    arms = [a for a in arms if os.path.exists(os.path.join(cell_dir, f"{a}_samples.npy"))]
    if not arms:
        return
    cols = 4
    rows = int(np.ceil(len(arms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), squeeze=False)
    for ax, arm in zip(axes.flat, arms):
        x = np.load(os.path.join(cell_dir, f"{arm}_samples.npy")).astype(np.float32)
        x = x[np.isfinite(x).all(1)]
        lab = assign_modes(mix, x)
        role = np.where(lab == OFF, 2, np.where(np.isin(lab, F), 1, 0))
        for rr, c in ((0, RETAIN_C), (1, FORGET_C), (2, OFF_C)):
            ax.scatter(x[role == rr, 0], x[role == rr, 1], s=1.5, color=c, alpha=0.6, lw=0)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.set_title(arm, fontsize=7, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(arms):]:
        ax.axis("off")
    fig.suptitle(f"{name}: retained (blue), forgotten (orange), off-mode (gray)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cells", default=os.path.join(RUNS, "cells"))
    ap.add_argument("--sample_cells", default="s0_k2_n1,s0_k2_n8,s0_k2_n32,s0_k4_n8")
    args = ap.parse_args()
    figs = os.path.join(args.results, "figures")
    os.makedirs(figs, exist_ok=True)
    summ = pd.read_csv(os.path.join(args.results, "summary.csv"))
    for m, (t, cmap) in HEAT_METRICS.items():
        if f"{m}_mean" in summ:
            heatmap(summ, m, t, cmap, os.path.join(figs, f"heatmap_{m}.png"))
    for m, yl in (("neg_mass_ratio", "forget mass ratio"), ("retain_tv", "retain TV vs pi_ref"),
                  ("retain_noise_flip", "retain-noise flip rate"), ("mass_off", "off-mode mass")):
        curves(summ, m, yl, os.path.join(figs, f"curves_{m}.png"))
    tradeoff(summ, os.path.join(figs, "tradeoff.png"))
    pp = os.path.join(args.results, "probes.csv")
    if os.path.exists(pp):
        probes = pd.read_csv(pp)
        for k in sorted(probes.n_forget.unique()):
            trajectories(probes, os.path.join(figs, f"trajectories_k{k}.png"), k=k)
    show = ["B0_frozen", "ORACLE_retain_bc", "B1_filtered", "B3_npo_b10", "B3_npo_b100_negonly", "B5_kto_b10",
            "DPO_cpl_b25", "RM_bon8", "R_ours_both", "R_spread_nhood_both", "R_spread_nhood_neg",
            "R_guide_nhood_both", "R_spread_inv_r1.5_both", "R_spread_reach_near_both",
            "R_spread_ninv_t0.4_fresh_both", "R_online2_spread_nhood_neg"]
    for c in args.sample_cells.split(","):
        d = os.path.join(args.cells, c)
        if os.path.isdir(d):
            samples(d, os.path.join(figs, f"samples_{c}.png"), show)
    print("figures ->", figs)


if __name__ == "__main__":
    main()
