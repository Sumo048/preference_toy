"""TRKNG rendered outputs and report charts.

  python explore/TRKNG/render.py cell alt 5 4    # rebuild B0, B1a, B6, TC for one cell; plot sampled actions
  python explore/TRKNG/render.py report          # D-per-setting chart and label-free-check scatter

`cell` retrains B1a, B6 and the best no-grid builder (TC) through trkng.run_train, the frozen code path,
unchanged. It points trkng.OUT at a scratch dir (holding a copy of the cell's committed TC masks) so the
committed results are not touched, and captures each trained policy at its E.compute call. The retrained
D is checked against the committed JSON (render_check_<cell>.json). The oracle (MIX.assign, mode centres)
is used for colouring and metrics only.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DOCS = os.path.join(ROOT, "docs", "TRKNG_report")
BANDS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
N_DEC, RENDER_PURPOSE = 1500, 97_001
MAX_BYTES = 600 * 1024
B6SPEC = "a=track60k-w=1-st=20000"


def save_small(fig, path, dpi=110):
    while True:
        fig.savefig(path, dpi=dpi)
        if os.path.getsize(path) < MAX_BYTES or dpi <= 40:
            break
        dpi -= 10
    plt.close(fig)
    return dpi, os.path.getsize(path)


def mode_colours(forget, retain, n_modes):
    reds = plt.get_cmap("Reds")(np.linspace(0.5, 0.95, max(len(forget), 1)))
    cool = list(plt.get_cmap("Blues")(np.linspace(0.55, 0.9, (len(retain) + 1) // 2)))
    warm = list(plt.get_cmap("Greens")(np.linspace(0.55, 0.9, len(retain) // 2 + 1)))
    col = {}
    for i, m in enumerate(sorted(forget)):
        col[m] = reds[i]
    for i, m in enumerate(sorted(retain)):
        col[m] = cool[i // 2] if i % 2 == 0 else warm[i // 2]
    col[-1] = (0.6, 0.6, 0.6, 1.0)
    return col


def render_cell(setting, seed, npm):
    import torch
    import trkng as T
    import e3
    from toypref.cond import experiment as E

    torch.set_num_threads(1)
    fz = json.load(open(os.path.join(HERE, "frozen.json")))
    best = fz["best"]
    best_name = fz["names"][best]
    src = T.cell_dir(setting, seed, npm)
    tmp = tempfile.mkdtemp(prefix="trkng_render_")
    T.OUT = tmp  # run_train writes its JSON here, and reads the masks copied here
    dst = T.cell_dir(setting, seed, npm)
    os.makedirs(dst)
    for ext in (".json", ".npz"):
        shutil.copy(os.path.join(src, f"masks_{best}_N{fz['N']}{ext}"), dst)

    captured = []
    orig = E.compute

    def spy(bank, pol):
        met = orig(bank, pol)
        captured.append((pol, met))
        return met

    E.compute = spy
    rows = []
    cell = T.Cell(setting, seed, npm)
    rows.append(("B0 (pi_ref)", cell.ref, orig(cell.bank, cell.ref), None))
    committed = {"B1a": os.path.join(ROOT, "explore", "E4", "results", setting, f"s{seed}_n{npm}", "B1a.json"),
                 "B6": os.path.join(ROOT, "explore", "E4", "results", setting, f"s{seed}_n{npm}", B6SPEC + ".json"),
                 best_name: os.path.join(src, best_name + ".json")}
    for arm, label in (("B1a", "B1a"), ("B6", "B6"), (best_name, f"{best} (best no-grid)")):
        captured.clear()
        T.run_train(setting, seed, npm, arm)
        pol, met = captured[-1]
        rows.append((label, pol, met, committed[arm]))
    E.compute = orig

    check = {}
    for label, _, met, cf in rows:
        if cf:
            d_ref = json.load(open(cf))["D"]
            check[label] = {"D_rerun": met["tv_to_ideal"], "D_committed": d_ref, "source": os.path.relpath(cf, ROOT),
                            "abs_diff": abs(met["tv_to_ideal"] - d_ref)}
    os.makedirs(DOCS, exist_ok=True)
    json.dump(check, open(os.path.join(DOCS, f"render_check_{setting}_s{seed}_n{npm}.json"), "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)

    MIX = E.MIX
    forget, retain = list(cell.ed.forget), list(cell.ed.retain)
    col = mode_colours(forget, retain, MIX.n_modes)
    g = E.gen(seed, RENDER_PURPOSE)
    u = torch.rand(len(BANDS), N_DEC, generator=g)
    xi = torch.randn(len(BANDS), N_DEC, 2, generator=g)
    cs = [lo + (hi - lo) * u[b] for b, (lo, hi) in enumerate(BANDS)]

    fig, axes = plt.subplots(len(rows), len(BANDS) + 1, figsize=(17, 3.3 * len(rows)),
                             gridspec_kw={"width_ratios": [1] * len(BANDS) + [1.25]})
    lim = MIX.r0 + MIX.r_slope + 4 * MIX.mode_std
    for r, (label, pol, met, _) in enumerate(rows):
        for b, (lo, hi) in enumerate(BANDS):
            ax = axes[r, b]
            with torch.no_grad():
                a = pol.sample(xi[b], cs[b]).numpy()
            lab = MIX.assign(a, cs[b])
            ax.scatter(a[:, 0], a[:, 1], s=3, c=[col[int(m)] for m in lab], linewidths=0, rasterized=True)
            track = MIX.centers(np.linspace(lo, hi, 25))
            mid = MIX.centers(np.array([(lo + hi) / 2]))[0]
            for m in range(MIX.n_modes):
                ax.plot(track[:, m, 0], track[:, m, 1], color="k", lw=0.6, alpha=0.5)
                ax.scatter(*mid[m], s=46, facecolors="none", edgecolors="k", linewidths=1.0)
                if m in forget:
                    ax.scatter(*mid[m], s=46, marker="x", color="k", linewidths=1.3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"c in [{lo:.2f}, {hi:.2f})", fontsize=11)
        D, rem, colat = met["tv_to_ideal"], 1 - met["neg_mass_ratio"], met["retain_tv"]
        axes[r, 0].set_ylabel(f"{label}\nD {D:.3f}  Removed {rem:.2f}\nCollateral {colat:.3f}", fontsize=10)
        ax = axes[r, -1]
        mass = [met[f"mass_m{m}"] for m in range(MIX.n_modes)] + [met["mass_off"]]
        ax.bar(range(MIX.n_modes + 1), mass, color=[col[m] for m in range(MIX.n_modes)] + [col[-1]])
        ax.axhline(1 / len(retain), color="k", ls="--", lw=0.8)
        ax.set_xticks(range(MIX.n_modes + 1))
        ax.set_xticklabels([f"{m}{'x' if m in forget else ''}" for m in range(MIX.n_modes)] + ["off"], fontsize=8)
        ax.set_ylim(0, max(0.45, 1.15 * max(mass)))
        if r == 0:
            ax.set_title("mass per mode (eval bank, 20k)\ndashed = ideal 1/|retain|", fontsize=10)
    fig.suptitle(f"{setting} s{seed} n{npm}: sampled actions, {N_DEC} decodes per band from the same noises in "
                 f"every row; reds = forgotten modes (x), blues/greens = retained, grey = between modes",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(DOCS, f"render_{setting}_s{seed}_n{npm}.png")
    dpi, size = save_small(fig, out)
    print(json.dumps({"png": os.path.relpath(out, ROOT), "dpi": dpi, "bytes": size, "check": check}), flush=True)


def report():
    import pandas as pd
    from scipy.stats import spearmanr

    df = pd.read_csv(os.path.join(HERE, "cells_confirm.csv"))
    settings = ["k1", "k2", "k4", "contig", "alt", "pairs", "b3p1"]
    arms = ["B1a", "B6", "TA", "TB", "TC"]
    colours = {"B1a": "#9e9e9e", "B6": "#424242", "TA": "#e6a23c", "TB": "#6a9fd4", "TC": "#2e7d32"}
    os.makedirs(DOCS, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.16
    for i, a in enumerate(arms):
        m, se = [], []
        for s in settings:
            d = df[(df.setting == s) & (df.arm == a)].D.dropna()
            m.append(d.mean()); se.append(d.std(ddof=1) / np.sqrt(len(d)))
        ax.bar(np.arange(len(settings)) + (i - 2) * w, m, w, yerr=se, capsize=2, label=a, color=colours[a])
    ax.set_xticks(range(len(settings))); ax.set_xticklabels(settings)
    ax.set_ylabel("D (TV to ideal), mean ± SE over cells")
    ax.legend(ncol=5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    print("d_by_setting", save_small(fig, os.path.join(DOCS, "d_by_setting.png")))

    b = df[df.arm.isin(["TA", "TB", "TC"])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, ch in zip(axes, ["stab_ari", "stab_jacc", "balance", "agree"]):
        for a in ["TA", "TB", "TC"]:
            d = b[b.arm == a]
            ax.scatter(d[ch], d.prec, s=10, alpha=0.7, color=colours[a], label=a, linewidths=0)
        dd = b[[ch, "prec"]].dropna()
        rho = spearmanr(dd[ch], dd.prec).correlation
        ax.set_title(f"{ch}: pooled ρ {rho:+.2f} (n {len(dd)})", fontsize=10)
        ax.set_xlabel(ch)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("oracle mask precision")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    print("checks_scatter", save_small(fig, os.path.join(DOCS, "checks_scatter.png")))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "explore", "E3"))
    if sys.argv[1] == "cell":
        render_cell(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        report()
