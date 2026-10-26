"""Figures robo_setup.png, robo_chunkfan.png, robo_rollouts.png for the SDPO report (robo2d seed 0).

Needs: base (--base, robo_base.py output) and arm policies (--arms, robo_train.py output). CPU only.
All rollouts use the repo's DT.rollout / E.Episodes; noises come from the CRN eval stream of the seed.
"""
import argparse, copy, json, os, sys
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from toypref.robo import data as DT, env as E, evaluate as EV, methods as MT
from toypref.robo.policy import ChunkFlow, frozen, load

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--arms", required=True)
ap.add_argument("--fig", default=os.path.expanduser("~/repos/toypref-report/artifact/sdpo_toys/fig"))
ap.add_argument("--facts", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only", default="setup,fan,rollouts")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
BLUE, ORANGE, AQUA, GREY, TX, T2, BG = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": TX, "axes.labelcolor": T2,
                     "xtick.color": T2, "ytick.color": T2, "axes.edgecolor": T2, "axes.linewidth": 0.6, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})
FACTS = {}

# ------------------------------------------------------------------ load
bd = os.path.join(args.base, f"s{s}")
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
log = DT.LogResult(lg["rows"], lg["diag"], lg["episodes"], lg["per_setup"])
rows, diag = log.rows, log.diag
train, held = DT.experiment_setups(s)
ev = EV.EvalSet(ref, s, train, held, log, M=DT.M_EVAL)
st = ev.st
base = ev.base
NAME = {"B0_frozen": "B0 (base)", "NPO_b100_neg": "NPO (β=100)", "KTO_b100_neg": "KTO (β=100)", "R_hop_bsup_neg_AL": "hop reroute + anchor",
        "FILTER_rej": "FILTER (reject)", "SDPO_fresh_b25": "SDPO fresh (β=25)", "SDPO_pin_b25": "SDPO pin (β=25)"}
ORDER = list(NAME)
pols = {}
for n in ORDER:
    if n == "B0_frozen": pols[n] = ref
    elif n == "FILTER_rej": pols[n] = MT.RejectFilter(ref, s, rows)
    elif not os.path.exists(os.path.join(args.arms, f"{n}.pt")):
        pols[n] = None
    else:
        ck = torch.load(os.path.join(args.arms, f"{n}.pt"), map_location="cpu", weights_only=False)
        p = ChunkFlow(**ck["kw"]); p.load_state_dict(ck["state"]); pols[n] = frozen(p)
fail_rate = np.array([base.negative[ev.idx == k].mean() for k in range(len(st))]) if False else None
sel_all = np.unique(ev.idx)
base_neg_rate = {int(k): float(base.negative[ev.idx == k].mean()) for k in sel_all}
base_fail16 = {int(k): float(base.negative[np.flatnonzero(ev.idx == k)[:16]].mean()) for k in sel_all}


def to_canon(setup):
    """similarity map into the canonical frame: start left, goal right, centred."""
    a, g = st.start[setup], st.goal[setup]
    u = (g - a) / np.linalg.norm(g - a); mid = (a + g) / 2
    R = np.array([[u[0], u[1]], [-u[1], u[0]]])
    return lambda P: (np.asarray(P) - mid) @ R.T


def outcome_color(neg, succ):
    return np.where(neg, ORANGE, np.where(succ, BLUE, AQUA))


def style(ax):
    for sp in ax.spines.values(): sp.set_linewidth(0.5)
    ax.set_xticks([]); ax.set_yticks([])


def draw_setup(ax, setup, T, xlim=None, ylim=None, goal=True):
    c, r = st.centre[setup], st.radius[setup]
    for k in range(st.n_obs[setup]):
        cc = T(c[k][None])[0]
        ax.add_patch(Circle(cc, r[k], fc=GREY, ec="none", zorder=1))
    a, g = T(st.start[setup][None])[0], T(st.goal[setup][None])[0]
    ax.plot(*a, "o", color=TX, ms=4.5, zorder=6, mew=0)
    if goal: ax.plot(*g, "*", color=TX, ms=9, zorder=6, mew=0)
    ax.set_aspect("equal")


def draw_trajs(ax, pos, neg, succ, T, lw=1.0, alpha=0.8):
    col = outcome_color(neg, succ)
    order = np.argsort([c == ORANGE for c in col], kind="stable")      # negatives on top
    for i in order:
        P = pos[i][~np.isnan(pos[i][:, 0])]
        Q = T(P)
        ax.plot(Q[:, 0], Q[:, 1], color=col[i], lw=lw, alpha=alpha, zorder=3, solid_capstyle="round")


def rolls(pol, setup, n=16):
    m = np.flatnonzero(ev.idx == setup)[:n]
    ro = DT.rollout(pol, st, ev.idx[m], ev.noise[m])
    return ro, m


# ------------------------------------------------------------------ fig 1: setups
def fig_setup():
    cand = sorted(sel_all, key=lambda k: base_neg_rate[int(k)])
    n = len(cand)
    picks = []
    for q in (0.2, 0.5, 0.8):
        k = cand[int(q * n)]
        picks.append(int(k))
    picks.append(int(cand[-1]))                    # the setup where the base fails most often
    # prefer diversity of obstacle counts for the first three
    FACTS["setup_picks"] = [{"setup": k, "n_obs": int(st.n_obs[k]), "base_neg_rate_64": base_neg_rate[k], "base_neg_16": base_fail16[k],
                             "group": "held" if k >= len(train) else "train"} for k in picks]
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 5.0))
    EXT = []
    for ax, k in zip(axs.ravel(), picks):
        T = to_canon(k)
        ro, m = rolls(ref, k, 16)
        draw_setup(ax, k, T)
        draw_trajs(ax, ro.pos, ro.negative, ro.success, T, lw=1.1, alpha=0.75)
        if k == picks[0]:     # chunk boundaries on one clean trajectory
            ok = np.flatnonzero(~ro.negative & ro.success)
            i = ok[0]
            P = T(ro.pos[i][~np.isnan(ro.pos[i][:, 0])])
            idx = np.arange(E.H, len(P), E.H)
            ax.plot(P[idx, 0], P[idx, 1], "o", ms=4.2, mfc=BG, mec=TX, mew=0.9, zorder=7)
        nneg = int(ro.negative.sum())
        ax.set_xlabel(f"{st.n_obs[k]} obstacle{'s' if st.n_obs[k] > 1 else ''}; {nneg}/16 negative here ({100*base_neg_rate[k]:.0f}% over 64 noises)", fontsize=8, labelpad=3)
        style(ax)
        ext = np.concatenate([T(st.start[k][None]), T(st.goal[k][None])] + [T(st.centre[k][j][None]) + np.array([[0, st.radius[k][j]], [0, -st.radius[k][j]]]) for j in range(st.n_obs[k])] +
                             [T(ro.pos[i][~np.isnan(ro.pos[i][:, 0])]) for i in range(len(ro.pos))])
        EXT.append((ext.min(0), ext.max(0)))
    lo = np.min([e[0] for e in EXT], 0); hi = np.max([e[1] for e in EXT], 0)
    for ax in axs.ravel():
        ax.set_xlim(lo[0] - 0.05, hi[0] + 0.05); ax.set_ylim(lo[1] - 0.05, hi[1] + 0.05)
    handles = [Line2D([], [], color=BLUE, lw=2, label="clean success"), Line2D([], [], color=ORANGE, lw=2, label="negative (collision or jerk)"),
               Line2D([], [], color=AQUA, lw=2, label="timeout (goal not reached)"),
               Line2D([], [], marker="o", color=TX, ls="", ms=4.5, label="start"), Line2D([], [], marker="*", color=TX, ls="", ms=9, label="goal"),
               Line2D([], [], marker="o", mfc=BG, mec=TX, ls="", ms=4.5, label="chunk boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.015, "one chunk = 8 velocity steps, decoded from noise at the current state, run open loop, then the policy replans (6 chunks per episode)", ha="center", fontsize=7.5, color=T2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.14, hspace=0.18, wspace=0.04)
    fig.savefig(os.path.join(args.fig, "robo_setup.png"), dpi=200); plt.close(fig)


# ------------------------------------------------------------------ fig 2: chunk fan
def tile(ep1, K):
    ep = E.Episodes(st, [int(ep1.idx[0])] * K)
    for a in ("p", "v", "t", "pos", "alive", "collided", "success", "col_chunk", "n_chunks", "clipped"):
        x = getattr(ep1, a)
        setattr(ep, a, np.repeat(x.copy(), K, 0))
    return ep


@torch.no_grad()
def prefix(setup, noise_seq, c):
    ep = E.Episodes(st, [setup])
    for k in range(c):
        o = torch.from_numpy(ep.obs())
        a = ref.sample(torch.from_numpy(noise_seq[k:k + 1]).float(), o)
        ep.execute(a.numpy(), k)
    return ep


@torch.no_grad()
def fan(pol, ep1, c, Z, cont):
    K = Z.shape[0]
    ep = tile(ep1, K)
    a1 = pol.sample(Z, torch.from_numpy(ep.obs()))
    ep.execute(a1.numpy(), c)
    seg = ep.pos[:, E.H * c: E.H * c + E.H + 1].copy()
    for k in range(c + 1, E.N_CHUNKS):
        live = np.flatnonzero(ep.alive)
        if not len(live): break
        a = np.zeros((K, E.ACT_DIM), np.float32)
        a[live] = pol.sample(cont[live, k], torch.from_numpy(ep.obs()[live])).numpy()
        ep.execute(a, k)
    neg, col, jerky, js = E.label(ep)
    return seg, neg, ep.success.copy(), ep.pos.copy()


def fig_fan():
    G = torch.Generator().manual_seed(2026)
    K = 64
    Z = torch.randn(K, E.ACT_DIM, generator=G)
    cont = torch.randn(K, E.N_CHUNKS, E.ACT_DIM, generator=G)
    bad_rows = np.flatnonzero(diag.bad_chunk)
    epn = log.episodes["noise"]; eps = log.episodes["setup"]
    # replay sanity
    rp = DT.rollout(ref, train, eps, torch.from_numpy(epn))
    FACTS["replay_logged_neg_still_bad"] = float(rp.negative.mean())
    pre = {}
    fr = {}
    cand = []
    for r in bad_rows:
        c = int(rows.chunk[r]); e = int(rows.episode[r])
        if c < 1 or not diag.ep_collided[r]: continue
        ep1 = prefix(int(eps[e]), epn[e], c)
        if not ep1.alive[0]: continue
        pre[r] = ep1
        _, neg, _, _ = fan(ref, ep1, c, Z, cont)
        fr[r] = float(neg.mean()); cand.append(r)
    fv = np.array([fr[r] for r in cand])
    ok = [r for r in cand if 0.25 <= fr[r] <= 0.75]
    med = np.median([fr[r] for r in ok])
    r0 = min(ok, key=lambda r: (abs(fr[r] - med), r))
    c0, e0 = int(rows.chunk[r0]), int(rows.episode[r0]); k0 = int(eps[e0])
    FACTS["fan"] = {"row": int(r0), "setup": k0, "chunk": c0, "n_candidate_collision_rows": len(cand), "base_fail_frac_at_state": fr[r0],
                    "median_base_fail_frac_all_cand": float(np.median(fv)), "n_noises": K}
    arms5 = ["B0_frozen", "SDPO_fresh_b25", "SDPO_pin_b25", "NPO_b100_neg", "KTO_b100_neg"]
    # aggregate P(fail | state) over ALL bad-chunk rows (collision + jerk), 16 noises each
    Z16, cont16 = Z[:16], cont[:16]
    agg = {n: [] for n in arms5}
    for r in bad_rows:
        c = int(rows.chunk[r]); e = int(rows.episode[r])
        ep1 = pre.get(r)
        if ep1 is None:
            ep1 = prefix(int(eps[e]), epn[e], c)
            if not ep1.alive[0]: continue
        for n in arms5:
            _, neg, _, _ = fan(pols[n], ep1, c, Z16, cont16)
            agg[n].append(float(neg.mean()))
    FACTS["fan"]["mean_fail_prob_over_bad_chunk_states"] = {n: float(np.mean(v)) for n, v in agg.items()}
    FACTS["fan"]["n_bad_chunk_states"] = len(agg["B0_frozen"])
    ep1 = pre[r0]
    p0 = ep1.p[0].copy()
    T = lambda P: np.asarray(P)
    fig, axs = plt.subplots(1, 5, figsize=(13.0, 3.5))
    dirg = st.goal[k0] - p0; dirg /= np.linalg.norm(dirg)
    ctr = p0 + 0.28 * dirg
    half = 0.5
    logged_a = rows.a[r0:r0 + 1].numpy()
    lep = tile(ep1, 1); lep.execute(logged_a, c0)
    lseg = lep.pos[0, E.H * c0: E.H * c0 + E.H + 1]
    for ax, n in zip(axs, arms5):
        seg, neg, succ, pos = fan(pols[n], ep1, c0, Z, cont)
        c, r = st.centre[k0], st.radius[k0]
        for j in range(st.n_obs[k0]):
            ax.add_patch(Circle(c[j], r[j], fc=GREY, ec="none", zorder=1))
        col = outcome_color(neg, succ)
        P0 = ep1.pos[0][:E.H * c0 + 1]
        ax.plot(P0[:, 0], P0[:, 1], color=T2, lw=1.0, zorder=2)
        for i in np.argsort([x == ORANGE for x in col], kind="stable"):
            full = pos[i][~np.isnan(pos[i][:, 0])]
            ax.plot(full[E.H * c0:, 0], full[E.H * c0:, 1], color=col[i], lw=0.6, alpha=0.18, zorder=2)
            q = seg[i][~np.isnan(seg[i][:, 0])]
            ax.plot(q[:, 0], q[:, 1], color=col[i], lw=1.3, alpha=0.8, zorder=3)
            ax.plot(q[-1, 0], q[-1, 1], "o", color=col[i], ms=2.2, alpha=0.9, zorder=3, mew=0)
        q = lseg[~np.isnan(lseg[:, 0])]
        ax.plot(q[:, 0], q[:, 1], color=TX, lw=2.0, zorder=5)
        ax.plot(q[-1, 0], q[-1, 1], "o", color=TX, ms=4, zorder=5)
        ax.plot(*p0, "o", mfc=BG, mec=TX, ms=6, mew=1.2, zorder=6)
        if np.abs(st.goal[k0] - ctr).max() < half: ax.plot(*st.goal[k0], "*", color=TX, ms=10, zorder=6)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_aspect("equal")
        style(ax)
        ax.set_xlabel(f"{NAME[n]}\nfail {int(neg.sum())}/{K} at this state\nall {len(agg[n])} bad-chunk states: {100*np.mean(agg[n]):.0f}% fail", fontsize=8, labelpad=4)
        FACTS["fan"].setdefault("fail_here", {})[n] = int(neg.sum())
    handles = [Line2D([], [], color=BLUE, lw=2, label="chunk then clean success"), Line2D([], [], color=ORANGE, lw=2, label="chunk then negative"),
               Line2D([], [], color=AQUA, lw=2, label="chunk then timeout"), Line2D([], [], color=TX, lw=2.2, label="logged bad chunk"),
               Line2D([], [], marker="o", mfc=BG, mec=TX, ls="", ms=6, label="state (chunk start)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, columnspacing=1.6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.30, wspace=0.03)
    fig.savefig(os.path.join(args.fig, "robo_chunkfan.png"), dpi=200); plt.close(fig)


# ------------------------------------------------------------------ fig 3: rollout grid
def fig_rollouts():
    hid = [k for k in sel_all if k >= len(train)]
    hs = sorted(hid, key=lambda k: base_neg_rate[int(k)])
    n = len(hs)
    picks = [int(hs[-1]), int(hs[int(0.7 * n)]), int(hs[int(0.4 * n)])]
    FACTS["roll_setups"] = [{"setup": k, "n_obs": int(st.n_obs[k]), "base_neg_rate_64": base_neg_rate[k]} for k in picks]
    nums = {}
    for a in ORDER:
        met = ev.compute(pols[a])
        nums[a] = {k: met[k] for k in ("held_success_clean", "held_bad", "held_timeout", "neg_success_clean", "neg_bad") if k in met}
    FACTS["roll_numbers_seed0"] = nums
    fig, axs = plt.subplots(len(ORDER), 3, figsize=(9.0, 1.3 * len(ORDER) + 0.5))
    R = {}
    ext = []
    for a in ORDER:
        for j, k in enumerate(picks):
            ro, m = rolls(pols[a], k, 16); R[a, j] = ro
            T = to_canon(k)
            P = np.concatenate([T(ro.pos[i][~np.isnan(ro.pos[i][:, 0])]) for i in range(len(ro.pos))] + [T(st.start[k][None]), T(st.goal[k][None])] +
                               [T(st.centre[k][q][None]) + np.array([[0, st.radius[k][q]], [0, -st.radius[k][q]]]) for q in range(st.n_obs[k])])
            ext.append((P.min(0), P.max(0)))
    lo = np.min([e[0] for e in ext], 0) - 0.05; hi = np.max([e[1] for e in ext], 0) + 0.05
    for i, a in enumerate(ORDER):
        for j, k in enumerate(picks):
            ax = axs[i, j]; T = to_canon(k); ro = R[a, j]
            draw_setup(ax, k, T)
            draw_trajs(ax, ro.pos, ro.negative, ro.success, T, lw=0.9, alpha=0.7)
            style(ax)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.text(0.015, 0.97, f"{int(ro.negative.sum())}/16 neg", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=T2,
                    bbox=dict(fc=BG, ec="none", alpha=0.85, pad=1.0), zorder=10)
            if j == 0:
                ax.set_ylabel(f"{NAME[a]}\nclean {100*nums[a]['held_success_clean']:.1f}%\nneg {100*nums[a]['held_bad']:.1f}%", rotation=0, ha="right", va="center", labelpad=6, fontsize=8, color=TX)
    handles = [Line2D([], [], color=BLUE, lw=2, label="clean success"), Line2D([], [], color=ORANGE, lw=2, label="negative (collision or jerk)"),
               Line2D([], [], color=AQUA, lw=2, label="timeout")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.19, right=0.995, top=0.995, bottom=0.05, hspace=0.12, wspace=0.03)
    fig.savefig(os.path.join(args.fig, "robo_rollouts.png"), dpi=200); plt.close(fig)


for w in args.only.split(","):
    {"setup": fig_setup, "fan": fig_fan, "rollouts": fig_rollouts}[w]()
    json.dump(FACTS, open(args.facts, "w"), indent=1, default=float)
    print("done", w, flush=True)
