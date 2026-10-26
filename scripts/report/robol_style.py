import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, TXT, SEC = "#fcfcfb", "#0b0b0b", "#52514e"
CLEAN, NEG, TO, OBST = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRID = "#e6e5df"
OUTDIR = "/home/soumojib/repos/toypref-report/artifact/sdpo_toys/fig"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "text.color": TXT, "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "axes.edgecolor": SEC, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "font.size": 8, "axes.titlesize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200, "figure.dpi": 100,
})


def spread(ys, gap):
    """Greedy repel of label y positions (sorted bottom-up) to keep >= gap between neighbours."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out
