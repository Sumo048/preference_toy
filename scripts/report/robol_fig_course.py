"""robol_setup.png, robol_rollouts.png, robol_chunkfan.png (needs setup_data.pkl / policy_data.pkl)."""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
from robol_style import *
import numpy as np
from matplotlib.patches import Circle, Rectangle
D = "/grogu/datasets/rollout/soumojib/toypref/report_robol"
OCC = {0: CLEAN, 1: NEG, 2: TO}
ZO = {0: 3, 2: 4, 1: 5}


def tf(c):
    """world (x, y) -> display (u, v): course rotated so start -> goal runs left to right."""
    d = c["wps"][-1] - c["wps"][0]
    th = np.arctan2(d[1], d[0]); cs, sn = np.cos(th), np.sin(th)
    o = c["wps"][0]
    return lambda P: np.stack([(P[..., 0] - o[0]) * cs + (P[..., 1] - o[1]) * sn, -(P[..., 0] - o[0]) * sn + (P[..., 1] - o[1]) * cs], -1)


def draw_course(ax, c, pad=0.35, lw=0.7, alpha=0.7):
    T = tf(c)
    for ctr, r in zip(c["centre"], c["radius"]):
        u = T(np.asarray(ctr)); ax.add_patch(Circle((u[0], u[1]), r, fc=OBST, ec="none", zorder=1))
    w = T(c["wps"])
    ax.plot(w[:, 0], w[:, 1], color="#0b0b0b", lw=0.4, ls=(0, (2, 2)), zorder=2, alpha=0.6)
    ax.scatter(w[1:-1, 0], w[1:-1, 1], s=7, c="#0b0b0b", zorder=6)
    ax.scatter(w[0, 0], w[0, 1], s=16, c="#0b0b0b", marker="s", zorder=6)
    ax.scatter(w[-1, 0], w[-1, 1], s=26, c="#0b0b0b", marker="D", zorder=6)
    ax.set_aspect("equal"); ax.axis("off")


def draw_paths(ax, paths, ocs, T, lw=0.7, alpha=0.7):
    for k in (0, 2, 1):
        for P, o in zip(paths, ocs):
            if o == k:
                Q = T(P); ax.plot(Q[:, 0], Q[:, 1], color=OCC[o], lw=lw, alpha=alpha, zorder=ZO[o], solid_capstyle="round")


def limits(c, paths_list, pad=0.3):
    T = tf(c)
    pts = [T(c["wps"])] + [T(P) for ps in paths_list for P in ps]
    xs = np.concatenate([q[:, 0] for q in pts]); ys = np.concatenate([q[:, 1] for q in pts])
    return (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)


def legend_outcomes(fig, x, y):
    for k, (lab, col) in enumerate([("clean success", CLEAN), ("negative (contact or jerk)", NEG), ("timeout", TO)]):
        pass


def outcome_handles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=CLEAN, lw=1.6, label="clean success"), Line2D([0], [0], color=NEG, lw=1.6, label="negative (contact or jerk)"),
            Line2D([0], [0], color=TO, lw=1.6, label="timeout"), Line2D([0], [0], marker="s", color="w", markerfacecolor="#0b0b0b", ms=4, label="start"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b0b0b", ms=3.5, label="waypoint"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#0b0b0b", ms=4.5, label="goal"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=OBST, ms=7, label="obstacle")]


# ================================================================= setup
if os.path.exists(f"{D}/setup_data.pkl") and "setup" in sys.argv:
    sd = pickle.load(open(f"{D}/setup_data.pkl", "rb"))
    fig = plt.figure(figsize=(12, 6.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1, 1, 1], hspace=0.28, wspace=0.05, left=0.01, right=0.99, top=0.95, bottom=0.07)
    for j, p in enumerate(sd["panels"]):
        ax = fig.add_subplot(gs[j, 0])
        draw_course(ax, p["course"]); draw_paths(ax, p["pos"], p["oc"], tf(p["course"]))
        x0, x1, y0, y1 = limits(p["course"], [p["pos"]]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        o = p["oc"]
        ax.set_title(f"held-out course {j + 1}, 16 base rollouts\n{int((o == 0).sum())} clean, {int((o == 1).sum())} negative, {int((o == 2).sum())} timeout",
                     fontsize=7.5, color=SEC, loc="left")
    ex = sd["example"]
    ax = fig.add_subplot(gs[0:2, 1])
    draw_course(ax, ex["course"])
    Tx = tf(ex["course"]); P = Tx(ex["pos"]); H = 8; hi = ex["hi"]
    ax.plot(P[:, 0], P[:, 1], color=NEG, lw=0.9, alpha=0.5, zorder=3)
    lo = max(hi - 2, 0)
    ax.plot(P[lo * H:(hi + 1) * H + 1, 0], P[lo * H:(hi + 1) * H + 1, 1], color=NEG, lw=2.4, zorder=5)
    ax.plot(P[hi * H:(hi + 1) * H + 1, 0], P[hi * H:(hi + 1) * H + 1, 1], color="#0b0b0b", lw=1.2, zorder=6)
    bnd = np.arange(0, len(P), H)
    ax.scatter(P[bnd, 0], P[bnd, 1], s=9, c="#fcfcfb", edgecolors=SEC, linewidths=0.5, zorder=7)
    for n_, k in enumerate(bnd[:-1] // H):
        a_, b_ = P[max(k * H - 2, 0)], P[min(k * H + 2, len(P) - 1)]
        tg = (b_ - a_) / (np.linalg.norm(b_ - a_) + 1e-9); nm_ = np.array([-tg[1], tg[0]]) * (0.2 if k % 2 == 0 else -0.2)
        ax.text(P[k * H, 0] + nm_[0], P[k * H, 1] + nm_[1], str(k), fontsize=5.5, color=TXT if lo <= k <= hi else SEC, ha="center", va="center", zorder=8)
    ax.set_title("one negative episode: dots = chunk boundaries (numbered)\nthick orange = approach window, black = symptom chunk", fontsize=7.5, color=SEC, loc="left")
    pts = np.concatenate([P, Tx(ex["course"]["wps"])]); ax.set_xlim(pts[:, 0].min() - 0.4, pts[:, 0].max() + 0.4); ax.set_ylim(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5)
    # strips
    ax2 = fig.add_axes([0.63, 0.085, 0.355, 0.27])
    n = ex["n_ran"]
    rows = [("trajectory label\n(whole episode)", range(0, n)), ("approach window\n(oracle: symptom + 2 before)", range(max(hi - 2, 0), hi + 1)), ("symptom chunk only\n(oracle)", [hi])]
    for r, (lab, ks) in enumerate(rows):
        y = 2 - r
        for k in range(18):
            ran = k < n
            on = k in ks
            ax2.add_patch(Rectangle((k, y - 0.34), 0.88, 0.68, fc=(NEG if on else "#ecebe4") if ran else SURF, ec=SEC if not ran else "none", lw=0.4, ls=":" if not ran else "-"))
        ax2.text(-0.4, y, lab, ha="right", va="center", fontsize=7, color=TXT)
    for k in range(18):
        ax2.text(k + 0.44, 2.62, str(k), ha="center", fontsize=6, color=SEC)
    ax2.text(8.8, 3.0, "chunk index", ha="center", fontsize=7, color=SEC)
    ax2.set_xlim(-0.2, 18); ax2.set_ylim(-0.5, 3.2); ax2.axis("off")
    ax2.text(0, -0.55, f"filled = labelled negative.  This episode ran {n} chunks; its contact is in chunk {hi}.", fontsize=6.5, color=SEC)
    fig.legend(handles=outcome_handles(), loc="lower left", bbox_to_anchor=(0.02, -0.005), ncol=7, fontsize=7, handlelength=1.5, columnspacing=1.2)
    fig.savefig(f"{OUTDIR}/robol_setup.png"); plt.close(fig)
    print("setup ok")

# ================================================================= rollouts + fan
if os.path.exists(f"{D}/policy_data.pkl") and "policy" in sys.argv:
    pdt = pickle.load(open(f"{D}/policy_data.pkl", "rb"))
    order = pdt["order"]; res = pdt["res"]
    short = {"B0": "B0 (base policy)", "NPO": "NPO", "KTO": "KTO", "SDPO fresh": "SDPO fresh", "SDPO pinned": "SDPO pinned",
             "SDPO pinned, 5000 steps": "SDPO pinned, overdose (5000 steps)",
             "SDPO fresh + re-collected negatives, S=8": "SDPO fresh, re-collected negatives, S = 8",
             "SDPO fresh + re-collected negatives, S=64": "SDPO fresh + re-collected\nnegatives, S = 64 (past optimum)"}
    sub = {"NPO": "k = 1/32, 475 steps", "KTO": "k = 1/32, 475 steps", "SDPO fresh": "k = 1/8, 1,901 steps", "SDPO pinned": "k = 1/32, 475 steps",
           "SDPO pinned, 5000 steps": "5,000 steps", "SDPO fresh + re-collected negatives, S=8": "k = 1, 15,205 steps",
           "SDPO fresh + re-collected negatives, S=64": "k = 2, 30,410 steps", "B0": "no edit"}
    nr = len(order)
    fig, axs = plt.subplots(nr, 3, figsize=(10.4, 1.25 * nr + 0.6), gridspec_kw=dict(left=0.27, right=0.99, top=0.975, bottom=0.05, hspace=0.12, wspace=0.04))
    for r, nm in enumerate(order):
        for c in range(3):
            ax = axs[r, c]; g = pdt["courses_geo"][c]; pn = res[nm]["panels"][c]
            draw_course(ax, g); draw_paths(ax, pn["pos"], pn["oc"], tf(g), lw=0.65, alpha=0.7)
            x0, x1, y0, y1 = limits(g, [res[o]["panels"][c]["pos"] for o in order], pad=0.2); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ra = res[nm]["rates"]
        axs[r, 0].text(-0.02, 0.5, f"{short[nm]}\n{sub[nm]}\nclean {ra['clean']:.3f}  negative {ra['neg']:.3f}  timeout {ra['timeout']:.3f}",
                       transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=7, color=TXT, linespacing=1.35)
    fig.legend(handles=outcome_handles(), loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.6, -0.002), columnspacing=1.2)
    fig.savefig(f"{OUTDIR}/robol_rollouts.png"); plt.close(fig)
    print("rollouts ok")

    FD = pdt["fan_train"]
    fan = FD["fan"]; fo = [a for a in ["B0", "NPO", "SDPO fresh", "SDPO pinned", "SDPO pinned, 5000 steps", "SDPO fresh + re-collected negatives, S=8"] if a in fan]
    g = FD["course"]; agg = FD["agg"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.3), gridspec_kw=dict(left=0.01, right=0.99, top=0.96, bottom=0.07, hspace=0.12, wspace=0.03))
    axs = axs.ravel()
    pre = FD["prefix"]; lg_ = FD["logged"]; T = tf(g); Pre = T(pre); p0 = Pre[-1]
    Lg = T(lg_); mid = 0.5 * (Lg[0] + Lg[-1])
    for ax, nm in zip(axs, fo):
        draw_course(ax, g)
        ax.plot(Pre[:, 0], Pre[:, 1], color="#0b0b0b", lw=0.8, alpha=0.5, zorder=2)
        draw_paths(ax, list(fan[nm]["seg"]), fan[nm]["oc"], T, lw=1.0, alpha=0.7)
        ax.plot(Lg[:, 0], Lg[:, 1], color="#0b0b0b", lw=2.2, zorder=8)
        ax.scatter(Lg[-1, 0], Lg[-1, 1], s=14, c="#0b0b0b", zorder=9)
        ax.set_xlim(mid[0] - 0.75, mid[0] + 0.75); ax.set_ylim(mid[1] - 0.75, mid[1] + 0.95)
        sh = fan[nm]["shares"]
        txt = f"{short.get(nm, nm)}\nthis state: clean {sh[0] * 100:.0f}%  neg. {sh[1] * 100:.0f}%  t/o {sh[2] * 100:.0f}%"
        if nm in agg:
            a_ = agg[nm]; txt += f"\nmean of {FD['nstates']} states: clean {a_[0] * 100:.0f}%  neg. {a_[1] * 100:.0f}%  t/o {a_[2] * 100:.0f}%"
        ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", fontsize=6.5, color=TXT, linespacing=1.4, zorder=20)
    for ax in axs[len(fo):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    hs = outcome_handles()[:3] + [Line2D([0], [0], color="#0b0b0b", lw=2, label="logged bad chunk")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(f"{OUTDIR}/robol_chunkfan.png"); plt.close(fig)
    print("fan ok")
