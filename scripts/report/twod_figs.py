"""Draw the report figures from the npz files written by twod_unc_run.py / twod_cond_run.py.
usage: twod_figs.py [unc|cond|all]   (TOYPREF_RUNS = report_2d dir)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

RUNS = os.environ.get("TOYPREF_RUNS", "/grogu/datasets/rollout/soumojib/toypref/report_2d")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "artifact", "sdpo_toys", "fig")
os.makedirs(FIG, exist_ok=True)
BG, TX, TX2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.family": "sans-serif",
                     "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"], "text.color": TX,
                     "axes.labelcolor": TX2, "xtick.color": TX2, "ytick.color": TX2, "axes.edgecolor": TX2,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8})


def colors_for(lab, forget):
    c = np.full(len(lab), BLUE, dtype=object)
    c[np.isin(lab, forget)] = ORANGE
    c[lab == -1] = AQUA
    return c


def clean(ax, lim=4.2, grid=False):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if grid:
        ax.grid(color="#ecebe6", lw=0.4)
    ax.set_axisbelow(True)


def rings(ax, centers, r=0.75, forget=()):
    for j, (x, y) in enumerate(centers):
        ax.add_patch(plt.Circle((x, y), r, fill=False, ec=GREY, lw=0.8, zorder=1))


def legend_handles():
    return [Line2D([], [], marker="o", ls="", ms=4, color=BLUE, label="retained mode"),
            Line2D([], [], marker="o", ls="", ms=4, color=ORANGE, label="forgotten mode"),
            Line2D([], [], marker="o", ls="", ms=4, color=AQUA, label="off-mode"),
            Line2D([], [], marker="x", ls="", ms=5, color="k", mew=1.2, label="logged negative")]


# --------------------------------------------------------------------------- unconditional
UNC_ARMS = [("B0_frozen", "B0 (frozen)"), ("B1_filtered", "B1 (BC on positives)"),
            ("B3_npo_b100_negonly", "NPO b100 (neg-only)"), ("B5_kto_b100_negonly", "KTO b100 (neg-only)"),
            ("R_ours_neg", "R_ours_neg"), ("R_near_nf_nhood_neg", "R_near_nf_nhood_neg"),
            ("SDPO_fresh_b1", "SDPO fresh b1"), ("SDPO_fresh_b25", "SDPO fresh b25"),
            ("SDPO_pin_b1", "SDPO pin b1"), ("SDPO_pin_b25", "SDPO pin b25")]


def load_unc():
    d = os.path.join(RUNS, "report_unc")
    com = np.load(os.path.join(d, "common.npz"))
    arms = {}
    for k, _ in UNC_ARMS:
        z = np.load(os.path.join(d, f"{k}.npz"), allow_pickle=True)
        arms[k] = dict(a=z["a_fix"], grid=z["grid_lab"], fin=json.loads(str(z["final"])), hist=json.loads(str(z["hist"])))
    return com, arms


def unc_figs():
    from toypref.mixture import MixtureConfig, assign_modes
    mix = MixtureConfig()
    com, arms = load_unc()
    F = list(com["forget"]); C = com["centers"]
    # 1 setup
    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=200)
    rings(ax, C)
    ax.scatter(*com["ref_a"].T, s=2.5, c=colors_for(com["lab_ref"], F), lw=0, alpha=0.8, zorder=2)
    ax.scatter(*com["a_neg"].T, marker="x", s=26, c="k", lw=1.1, zorder=4)
    for j, (x, y) in enumerate(C):
        r = 4.0
        ax.text(x * r / 3, y * r / 3, str(j), ha="center", va="center", fontsize=7,
                color=ORANGE if j in F else TX2, fontweight="bold" if j in F else "normal")
    clean(ax, 4.5, grid=False)
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=7, handlelength=1, columnspacing=1.2)
    fig.savefig(os.path.join(FIG, "unc_setup.png"), bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    # 3 samples
    fig, axs = plt.subplots(2, 5, figsize=(11, 5.3), dpi=200)
    for ax, (k, nm) in zip(axs.ravel(), UNC_ARMS):
        A = arms[k]; lab = assign_modes(mix, A["a"])
        rings(ax, C)
        ax.scatter(*A["a"].T, s=2, c=colors_for(lab, F), lw=0, alpha=0.8, zorder=2)
        ax.scatter(*com["a_neg"].T, marker="x", s=14, c="k", lw=0.8, zorder=4)
        clean(ax, 4.2)
        ax.set_title(f"{nm}\nforgotten mass {A['fin']['neg_mass_ratio']:.2f}   retain TV {A['fin']['retain_tv']:.2f}",
                     fontsize=7.5, loc="left", color=TX, pad=3)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2)
    fig.savefig(os.path.join(FIG, "unc_samples.png"), bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    # 4 noise map
    NM = [("B0_frozen", "B0 (frozen)"), ("B3_npo_b100_negonly", "NPO b100"), ("SDPO_fresh_b25", "SDPO fresh b25"),
          ("SDPO_pin_b25", "SDPO pin b25"), ("SDPO_pin_b1", "SDPO pin b1")]
    cmap = ListedColormap([AQUA, BLUE, ORANGE])
    fig, axs = plt.subplots(1, 5, figsize=(12.5, 2.9), dpi=200)
    for ax, (k, nm) in zip(axs, NM):
        g = arms[k]["grid"]                      # [iy? ] meshgrid(g1,g1) -> shape (ny, nx), first idx = y
        v = np.where(g == -1, 0, np.where(np.isin(g, F), 2, 1))
        ax.imshow(v, origin="lower", extent=(-3, 3, -3, 3), cmap=cmap, vmin=0, vmax=2, alpha=0.55, interpolation="nearest")
        ax.scatter(*com["xi_neg"].T, marker="x", s=16, c="k", lw=0.9, zorder=3)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
        for sp in ax.spines.values(): sp.set_linewidth(0.5)
        f = arms[k]["fin"]
        ax.set_title(f"{nm}\nforgotten mass {f['neg_mass_ratio']:.2f}   off {f['mass_off']:.2f}", fontsize=7.5, loc="left", pad=3)
    fig.legend(handles=[Line2D([], [], marker="s", ls="", ms=6, color=BLUE, alpha=0.55, label="noise decodes to a retained mode"),
                        Line2D([], [], marker="s", ls="", ms=6, color=ORANGE, alpha=0.55, label="decodes to a forgotten mode"),
                        Line2D([], [], marker="s", ls="", ms=6, color=AQUA, alpha=0.55, label="decodes off-mode"),
                        Line2D([], [], marker="x", ls="", ms=5, color="k", mew=1.2, label="logged negative noise ξ")],
               loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(FIG, "unc_noisemap.png"), bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    # 6 paths
    PA = [k for k, _ in UNC_ARMS if k != "B0_frozen"]
    nm = dict(UNC_ARMS)
    cols = {k: (CAT[i] if i < 8 else "#8b5e34") for i, k in enumerate(PA)}
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=200)
    ends = []
    for k in PA:
        h = arms[k]["hist"]
        x = [1.0] + [r["probe_neg_mass_ratio"] for r in h if "probe_neg_mass_ratio" in r]
        y = [0.0] + [r["probe_retain_tv"] for r in h if "probe_retain_tv" in r]
        ls = "--" if "pin" in k else "-"
        ax.plot(x, y, ls, color=cols[k], lw=1.4, marker="o", ms=2.2, zorder=3)
        ax.plot(x[-1], y[-1], "o", color=cols[k], ms=5, zorder=4)
        ends.append((k, x[-1], y[-1]))
    ax.plot(1, 0, "s", color=TX2, ms=5, zorder=5); ax.annotate("B0 / start", (1, 0), (1.0, 0.03), fontsize=7, color=TX2, ha="right", va="bottom")
    ax.plot(0, 0, "*", color=TX, ms=11, zorder=5); ax.annotate("ideal", (0, 0), (0.02, 0.025), fontsize=7.5, color=TX)
    ax.set_xlabel("forgotten-mode mass relative to π_ref (lower = more forgotten)")
    ax.set_ylabel("retain TV (lower = less collateral)")
    ax.set_ylim(-0.045, None)
    ax.grid(color="#ecebe6", lw=0.4)
    # direct labels next to each line end (manual offsets in data units for this fixed cell)
    off = {"SDPO_pin_b1": (0.04, 0.0, "left"), "R_ours_neg": (0.05, 0.0, "left"), "R_near_nf_nhood_neg": (0.05, 0.0, "left"),
           "B5_kto_b100_negonly": (0.05, 0.012, "left"), "B1_filtered": (0.12, 0.035, "left"),
           "SDPO_pin_b25": (0.06, 0.065, "left"), "SDPO_fresh_b1": (0.05, -0.04, "left"),
           "B3_npo_b100_negonly": (-0.24, -0.04, "left"), "SDPO_fresh_b25": (-0.10, -0.03, "left")}
    for k, x, y in ends:
        dx, dy, ha = off[k]
        ax.annotate(nm[k], (x, y), (x + dx, y + dy), fontsize=7.5, color=cols[k], ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=cols[k], shrinkA=1, shrinkB=3))
    ax.set_xlim(-0.04, 1.03)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(FIG, "unc_paths.png"), bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return {k: arms[k]["fin"] for k, _ in UNC_ARMS}


# --------------------------------------------------------------------------- conditional
COND_ARMS = [("B0_frozen", "B0 (frozen)"), ("NPO_b100_neg", "NPO b100"), ("KTO_b10_neg", "KTO b10"),
             ("R_hop_bsup_neg", "R_hop_bsup_neg"), ("R_hopB_bsup_neg_AMdist", "R_hopB_bsup_AMdist"),
             ("SDPO_fresh_b25", "SDPO fresh b25"), ("SDPO_pin_b25", "SDPO pin b25"), ("SDPO_hop_pin_b25", "SDPO hop pin b25")]


def cond_figs():
    from toypref.cond.experiment import MIX
    d = os.path.join(RUNS, "report_cond")
    com = np.load(os.path.join(d, "common.npz"))
    CS = com["CS"]; F = list(com["forget"]); nc, na = com["neg_c"], com["neg_a"]
    NZ = com["ref_a"].shape[1]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3), dpi=200)
    for i, (ax, c) in enumerate(zip(axs, CS)):
        lab = MIX.assign(com["ref_a"][i], np.full(NZ, c))
        cen = MIX.centers([c])[0]
        rings(ax, cen, r=3 * MIX.mode_std)
        ax.scatter(*com["ref_a"][i].T, s=2, c=colors_for(lab, F), lw=0, alpha=0.8, zorder=2)
        m = np.abs(nc - c) < 0.1
        ax.scatter(*na[m].T, marker="x", s=22, c="k", lw=1.0, zorder=4)
        clean(ax, 4.2)
        ax.set_title(f"c = {c:.2f}   ({m.sum()} logged negatives with |c−c₀|<0.1)", fontsize=7.5, loc="left", pad=3)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(FIG, "cond_setup.png"), bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    # samples: 2 conditions with most negatives nearby
    cnt = [(np.abs(nc - c) < 0.1).sum() for c in CS]
    sel = sorted(np.argsort(cnt)[::-1][:2])
    res = {}
    fig, axs = plt.subplots(2, len(COND_ARMS), figsize=(len(COND_ARMS) * 2.0, 4.7), dpi=200)
    for j, (k, nmx) in enumerate(COND_ARMS):
        z = np.load(os.path.join(d, f"{k}.npz"), allow_pickle=True); fin = json.loads(str(z["final"])); res[k] = fin
        for r, i in enumerate(sel):
            ax = axs[r, j]; c = CS[i]; a = z["a"][i]; lab = z["lab"][i]
            rings(ax, MIX.centers([c])[0], r=3 * MIX.mode_std)
            ax.scatter(*a.T, s=1.6, c=colors_for(lab, F), lw=0, alpha=0.8, zorder=2)
            m = np.abs(nc - c) < 0.1
            ax.scatter(*na[m].T, marker="x", s=12, c="k", lw=0.8, zorder=4)
            clean(ax, 4.2); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{nmx}\nforgotten {fin['neg_mass_ratio']:.2f} · retain TV {fin['retain_tv']:.2f}", fontsize=6.8, loc="left", pad=3)
            if j == 0:
                ax.set_ylabel(f"c = {c:.2f}", fontsize=8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8, w_pad=0.4)
    fig.savefig(os.path.join(FIG, "cond_samples.png"), bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return res, [float(CS[i]) for i in sel], cnt


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if w in ("unc", "all"):
        out["unc"] = unc_figs()
    if w in ("cond", "all"):
        out["cond"] = cond_figs()
    print(json.dumps(out, indent=1, default=float))
