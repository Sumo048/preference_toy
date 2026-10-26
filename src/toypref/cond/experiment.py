"""Conditional-toy experiment: base policy, edit sets, arms and metrics.

Fairness rules are those of the unconditional toy: every stream is keyed by (seed, purpose[,
negative uid]); all arms of a seed share the base, the eval set, the minibatch stream and each
negative's target/support draws; edit sets are nested in n_neg and n_forget.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .. import objectives as O
from ..flow import EMA, FlowPolicy
from ..train import cosine
from . import targets as TG
from .task import OFF, CondFlow, CondMixture, frozen

MIX = CondMixture()
P_BASE, P_EVAL, P_POOL, P_PERM, P_TRAIN, P_PU, P_ANCHOR = 0, 10_000, 20_000, 30_000, 50_000, 60_000, 70_000


def gen(seed, purpose):
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + purpose)


def nprng(seed, purpose):
    return np.random.default_rng(int(seed) * 1_000_003 + purpose)


# ------------------------------------------------------------------ base
def train_base(seed, steps=40_000, batch=1024, lr=1e-3, log=print):
    torch.manual_seed(seed)
    pol = CondFlow()
    ema = EMA(pol, 0.999)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
    g, rng = gen(seed, P_BASE), nprng(seed, P_BASE)
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine(step, steps, 1000, lr, 1e-5)
        c = rng.random(batch).astype(np.float32)
        x, _ = MIX.sample(c, rng)
        a, c = torch.from_numpy(x), torch.from_numpy(c)
        noise = torch.randn(a.shape, generator=g)
        t = FlowPolicy.sample_time(batch, g)
        loss = pol.flow_loss(a, noise, t, c).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        ema.update(pol)
        if step % 5000 == 0 or step == steps - 1:
            log(f"[cond base s{seed}] step {step} loss {float(loss.detach()):.4f}")
    pol.load_state_dict(ema.shadow)
    return frozen(pol)


# ------------------------------------------------------------------ data
@dataclass
class Pool:
    c: torch.Tensor
    xi: torch.Tensor
    a: torch.Tensor
    mode: np.ndarray


def make_pool(ref, seed, n, purpose=P_POOL) -> Pool:
    g = gen(seed, purpose)
    c = torch.rand(n, generator=g)
    xi = torch.randn(n, 2, generator=g)
    a = ref.sample(xi, c)
    return Pool(c=c, xi=xi, a=a, mode=MIX.assign(a, c))


LAYOUTS = {"contig": (0, 1, 2, 3), "alt": (0, 2, 4, 6), "pairs": (0, 1, 4, 5), "b3p1": (0, 1, 2, 5)}


def forget_modes(seed, k, layout="random"):
    """random: first k of a per-seed permutation (rounds 1-3). Otherwise a fixed ring pattern
    rotated by a per-seed random offset (n_forget = 4 only)."""
    if layout == "random":
        return sorted(int(m) for m in nprng(seed, P_PERM).permutation(MIX.n_modes)[:k])
    pat = LAYOUTS[layout]
    assert k == len(pat), (layout, k)
    o = int(nprng(seed, P_PERM + 1).integers(MIX.n_modes))
    return sorted((o + d) % MIX.n_modes for d in pat)


@dataclass
class Edit:
    forget: List[int]
    retain: List[int]
    uid: np.ndarray
    c: torch.Tensor
    xi: torch.Tensor
    a: torch.Tensor


def editset(pool: Pool, seed, k, n_neg, layout="random") -> Edit:
    F_ = forget_modes(seed, k, layout)
    idx = np.concatenate([np.flatnonzero(pool.mode == m)[:n_neg] for m in F_])
    return Edit(forget=F_, retain=[m for m in range(MIX.n_modes) if m not in F_], uid=idx,
                c=pool.c[idx], xi=pool.xi[idx], a=pool.a[idx])


# ------------------------------------------------------------------ PU classifier
class BadNet(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.c_freqs = 8
        self.net = torch.nn.Sequential(torch.nn.Linear(2 + 16, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, 1))

    def forward(self, c, a):
        k = torch.arange(1, self.c_freqs + 1, dtype=torch.float32)
        ang = math.pi * c.reshape(-1, 1) * k
        return self.net(torch.cat([a / 3.0, torch.sin(ang), torch.cos(ang)], -1)).squeeze(-1)


def train_pu(ed: Edit, pool: Pool, seed, prior: float, steps=3000, batch=256):
    """nnPU (Kiryo et al. 2017): labelled = logged negatives (the 'bad' class), unlabelled =
    pi_ref rollouts at random conditions; class prior = share of bad behaviour (assumed known)."""
    torch.manual_seed(seed)
    net = BadNet()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = gen(seed, P_PU)
    nU = pool.c.shape[0]
    sp = lambda z: F.softplus(-z)        # loss for label +1 (bad)
    sn = lambda z: F.softplus(z)         # loss for label -1
    for _ in range(steps):
        i = torch.randint(0, ed.c.shape[0], (batch,), generator=g)
        j = torch.randint(0, nU, (batch,), generator=g)
        zp, zu = net(ed.c[i], ed.a[i]), net(pool.c[j], pool.a[j])
        neg_risk = sn(zu).mean() - prior * sn(zp).mean()
        loss = prior * sp(zp).mean() + torch.clamp(neg_risk, min=0.0)
        if neg_risk < 0:
            loss = loss - neg_risk          # nnPU gradient step on the violating term
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    return net


# ------------------------------------------------------------------ arms
@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                 # frozen | npo | kto | reroute
    beta: float = 0.0
    rule: str = ""
    support: str = "nhood"
    transplant: bool = False
    anchor: bool = False      # NEG+U: unlabelled pi_ref pairs at random conditions
    amask: str = ""           # round 3: masked anchor on samples NOT voted bad ("map" | "dist")
    mine: bool = False        # round 3: reroute vote-flagged unlabelled samples too
    track: str = ""           # round 4: knn | sortc | m16 | robust | joint | atyp | reach
    sdpo_noise: str = ""      # self-paired DPO (user idea): "fresh" | "pin"
    sdpo_pos: str = "rand"    # positive: "rand" (8 fresh pi_ref samples at c_i) | "hop" (the hop target)
    practical: bool = True    # False = uses information a real system lacks (oracle geometry)


ARMS: Dict[str, Arm] = {}
for a in [Arm("B0_frozen", "frozen"),
          Arm("NPO_b10_neg", "npo", beta=10.0), Arm("NPO_b100_neg", "npo", beta=100.0),
          Arm("KTO_b10_neg", "kto", beta=10.0), Arm("KTO_b100_neg", "kto", beta=100.0),
          Arm("R_ours_neg", "reroute", rule="ours", support="log"),
          Arm("R_own_nhood_neg", "reroute", rule="own", practical=False),
          Arm("R_basin_nhood_neg", "reroute", rule="basin"),
          Arm("R_basin_bsup_neg", "reroute", rule="basin", support="bsup"),
          Arm("R_hop_nhood_neg", "reroute", rule="hop"),
          Arm("R_hop_bsup_neg", "reroute", rule="hop", support="bsup"),
          Arm("R_pu_nhood_neg", "reroute", rule="pu"),
          Arm("R_basin_nhood_tx_neg", "reroute", rule="basin", transplant=True),
          Arm("R_basin_bsup_tx_neg", "reroute", rule="basin", support="bsup", transplant=True),
          Arm("R_hop_nhood_tx_neg", "reroute", rule="hop", transplant=True),
          # round 2: noise veto (docs/COND_REROUTE_PREREG.md)
          Arm("R_hopV_bsup_neg", "reroute", rule="hopV", support="bsup"),
          Arm("R_hopB_bsup_neg", "reroute", rule="hopB", support="bsup"),
          Arm("R_hopVB_bsup_neg", "reroute", rule="hopVB", support="bsup"),
          # round 3 (n_forget 1-2): masked anchor and mined pseudo-negatives
          Arm("R_hopB_bsup_neg_AMmap", "reroute", rule="hopB", support="bsup", amask="map"),
          Arm("R_hopB_bsup_neg_AMdist", "reroute", rule="hopB", support="bsup", amask="dist"),
          Arm("R_hopB_bsup_mine_neg_AMmap", "reroute", rule="hopB", support="bsup", amask="map", mine=True),
          Arm("R_hopB_bsup_mine_neg", "reroute", rule="hopB", support="bsup", mine=True),
          Arm("NPO_b10_neg_AMmap", "npo", beta=10.0, amask="map"),
          Arm("NPO_b100_neg_AMmap", "npo", beta=100.0, amask="map"),
          Arm("NPO_b10_neg_AMdist", "npo", beta=10.0, amask="dist"),
          Arm("KTO_b10_neg_AMmap", "kto", beta=10.0, amask="map"),
          Arm("KTO_b100_neg_AMmap", "kto", beta=100.0, amask="map"),
          # user's local anchor (round 3 amendment)
          Arm("R_hop_bsup_neg_AL", "reroute", rule="hop", support="bsup", amask="local"),
          Arm("R_hopB_bsup_neg_AL", "reroute", rule="hopB", support="bsup", amask="local"),
          Arm("R_hopB_bsup_neg_AML", "reroute", rule="hopB", support="bsup", amask="local+map"),
          Arm("NPO_b10_neg_AL", "npo", beta=10.0, amask="local"),
          Arm("NPO_b100_neg_AL", "npo", beta=100.0, amask="local"),
          Arm("KTO_b10_neg_AL", "kto", beta=10.0, amask="local"),
          Arm("PSEUDO_vote_map", "pseudo", amask="map"),
          Arm("PSEUDO_vote_dist", "pseudo", amask="dist"),
          # round 4 (docs/NOVEL_PREREG.md): behaviour tracking and friends
          Arm("TRACK_bc", "maskbc", track="knn"),
          Arm("TRACK_bc_sortc", "maskbc", track="sortc"),
          Arm("TRACK_bc_M16", "maskbc", track="m16"),
          Arm("TRACK_bc_M256", "maskbc", track="m256"),     # reporting only (spec #1 asks for M = 256)
          Arm("RTRACK_bc", "maskbc", track="robust"),
          Arm("JOINT_bc", "maskbc", track="joint"),
          Arm("ATYP_dist", "maskbc", track="atyp"),
          Arm("PRA_trk", "pra", track="knn"),
          Arm("REACH_recipe", "reroute", rule="hopR", support="bsup", amask="reach", track="reach"),
          # self-paired DPO (user idea, 2026-10-09): loser = logged negative, winner = pi_ref sample at its state
          Arm("SDPO_fresh_b1", "sdpo", beta=1.0, sdpo_noise="fresh"),
          Arm("SDPO_fresh_b25", "sdpo", beta=25.0, sdpo_noise="fresh"),
          Arm("SDPO_pin_b1", "sdpo", beta=1.0, sdpo_noise="pin"),
          Arm("SDPO_pin_b25", "sdpo", beta=25.0, sdpo_noise="pin"),
          Arm("SDPO_hop_pin_b1", "sdpo", beta=1.0, sdpo_noise="pin", sdpo_pos="hop"),
          Arm("SDPO_hop_pin_b25", "sdpo", beta=25.0, sdpo_noise="pin", sdpo_pos="hop"),
          # skeptic's controls (novelty workflow wf_2078e6c7-444)
          Arm("SDPO_fresh_b10", "sdpo", beta=10.0, sdpo_noise="fresh"),
          Arm("SDPO_fresh_b100", "sdpo", beta=100.0, sdpo_noise="fresh"),
          Arm("SDPO_hop_pin_bH1", "sdpo", beta=1.0, sdpo_noise="pin", sdpo_pos="hop", track="betaH"),
          Arm("SDPO_hop_pin_bH3", "sdpo", beta=3.0, sdpo_noise="pin", sdpo_pos="hop", track="betaH"),
          Arm("SDPO_hop_pin_bH10", "sdpo", beta=10.0, sdpo_noise="pin", sdpo_pos="hop", track="betaH"),
          Arm("CTRL_hop_GA", "sdpo", beta=0.0, sdpo_noise="pin", sdpo_pos="hop", track="ungated"),
          Arm("CTRL_synB1_hop", "sdpo", beta=0.0, sdpo_noise="fresh", sdpo_pos="hop", track="synbc"),
          # NEG+U: same methods with an unlabelled self-distillation anchor
          Arm("NPO_b10_neg_A", "npo", beta=10.0, anchor=True),
          Arm("KTO_b10_neg_A", "kto", beta=10.0, anchor=True),
          Arm("R_basin_nhood_neg_A", "reroute", rule="basin", anchor=True),
          Arm("R_basin_bsup_tx_neg_A", "reroute", rule="basin", support="bsup", transplant=True, anchor=True),
          Arm("R_hop_nhood_neg_A", "reroute", rule="hop", anchor=True)]:
    ARMS[a.name] = a


@dataclass
class Ctx:
    ref: CondFlow
    seed: int
    ed: Edit
    pool: Pool
    anchor: Pool
    steps: int = 5000
    batch: int = 256
    lr: float = 1e-4
    pu: Optional[object] = None


def _pick(n, k, g):
    return torch.randint(0, n, (k,), generator=g)


N_VOTE_POOL = 4096
N_MINE = 256


def get_vote(ctx: "Ctx"):
    """(bank, unflagged mask over the first N_VOTE_POOL anchor samples, diagnostics); cached."""
    if getattr(ctx, "_vote", None) is None:
        bank = TG.NegBank(ctx.ref, ctx.seed, ctx.ed.uid, ctx.ed.c, ctx.ed.xi, ctx.ed.a)
        P = ctx.anchor
        v = bank.rate(P.xi[:N_VOTE_POOL])
        flag = (v >= bank.theta).numpy()
        bad = np.isin(P.mode[:N_VOTE_POOL], ctx.ed.forget)
        diag = {"vote_theta": bank.theta, "vote_self_rate_median": bank.self_rate_median,
                "vote_flag_frac": float(flag.mean()),
                "vote_precision": float(bad[flag].mean()) if flag.any() else float("nan"),
                "vote_recall": float(flag[bad].mean()) if bad.any() else float("nan")}
        ctx._vote = (bank, torch.from_numpy(~flag), diag)
    return ctx._vote


def get_local(ctx: "Ctx"):
    if getattr(ctx, "_local", None) is None:
        ctx._local = TG.local_anchor_triplets(ctx.ref, ctx.seed, ctx.ed.uid, ctx.ed.c, ctx.ed.a)
    return ctx._local


N_NODES = 4096


def _nodes(ctx):
    n = getattr(ctx, "n_nodes", N_NODES)
    P = ctx.anchor
    node_c = np.concatenate([P.c[:n].numpy(), ctx.ed.c.numpy()]).astype(np.float64)
    neg_node = np.arange(n, n + len(ctx.ed.c))
    return n, node_c, neg_node


def get_tracker(ctx, variant):
    """(tracker, neg_comp) cached per variant key (knn/sortc/m16/bank2)."""
    cache = ctx.__dict__.setdefault("_trk", {})
    if variant not in cache:
        from . import track as TK
        n, node_c, neg_node = _nodes(ctx)
        kw = {"knn": {}, "sortc": {"graph": "sortc"}, "m16": {"M": 16}, "m256": {"M": 256}, "bank2": {"bank": TK.P_BANK2}}[variant]
        cache[variant] = TK.build_tracker(ctx.ref, ctx.seed, node_c, neg_node, ctx.ed.a.numpy(), **kw)
    return cache[variant]


def get_mask(ctx, variant):
    """good mask over the first n anchor-pool samples + diagnostics (oracle used ONLY for diagnostics)."""
    cache = ctx.__dict__.setdefault("_masks", {})
    if variant in cache:
        return cache[variant]
    from . import track as TK
    n, node_c, neg_node = _nodes(ctx)
    P = ctx.anchor
    pa, pnode = P.a[:n].numpy(), np.arange(n)
    diag = {}
    if variant in ("knn", "sortc", "m16", "m256", "atyp"):
        tr, nc = get_tracker(ctx, "knn" if variant == "atyp" else variant)
        flagged = TK.flag_or(nc) if variant != "atyp" else set()
        bad, assigned = TK.pool_bad_mask(tr, flagged, pa, pnode)
        diag.update({f"trk_{k}": v for k, v in tr.diag.items()})
        diag["trk_flagged"] = len(flagged)
    elif variant == "robust":
        bads, asg = [], []
        for bk in ("knn", "bank2"):
            tr, nc = get_tracker(ctx, bk)
            flagged, alarms = TK.flag_enrichment(tr, nc, neg_node)
            b, a = TK.pool_bad_mask(tr, flagged, pa, pnode)
            bads.append(b)
            asg.append(a)
            diag[f"rtrk_{bk}_flagged"] = len(flagged)
            diag[f"rtrk_{bk}_alarms"] = alarms
            diag[f"rtrk_{bk}_refusals"] = tr.diag["refusals"]
        bad = bads[0] & bads[1]
        assigned = asg[0] | asg[1]
    elif variant == "joint":
        bad, assigned, jd = TK.joint_labeller(ctx.ref, ctx.seed, node_c[:n], ctx.ed.c.numpy(), ctx.ed.a.numpy(),
                                              P.c[:n].numpy(), pa)
        diag.update(jd)
    else:
        raise ValueError(variant)
    good = assigned & ~bad
    diag["mask_empty"] = bool(not good.any())
    truth_bad = np.isin(P.mode[:n], ctx.ed.forget)
    flagged_bad = bad & assigned
    diag.update({"mask_good_frac": float(good.mean()), "mask_excluded_frac": float((~assigned).mean()),
                 "mask_bad_precision": float(truth_bad[flagged_bad].mean()) if flagged_bad.any() else float("nan"),
                 "mask_bad_recall": float(flagged_bad[truth_bad].mean()) if truth_bad.any() else float("nan"),
                 "mask_good_purity": float((~truth_bad)[good].mean()) if good.any() else float("nan")})
    cache[variant] = (torch.from_numpy(good), diag)
    return cache[variant]


def mask_bc_loss(ctx, good):
    P = ctx.anchor
    n = good.shape[0]
    c, a = P.c[:n][good], P.a[:n][good]
    B = ctx.batch
    if c.shape[0] == 0:          # review fix 4: nothing to clone -> no edit (policy stays pi_ref)
        return lambda pol, g: sum((p * 0).sum() for p in pol.parameters())

    def loss(pol, g):
        j = _pick(c.shape[0], B, g)
        t = FlowPolicy.sample_time(B, g)
        return pol.flow_loss(a[j], torch.randn(a[j].shape, generator=g), t, c[j]).mean()
    return loss


def get_reach(ctx):
    """REACH vote (spec #5) on top of the knn tracker's delta; None when fewer than 2 negatives."""
    if "_reach" in ctx.__dict__:
        return ctx._reach
    from . import track as TK
    N = len(ctx.ed.c)
    if N < 2:
        ctx._reach = None
        return None
    tr, _ = get_tracker(ctx, "knn")
    n, node_c, neg_node = _nodes(ctx)
    bank = TG.NegBank(ctx.ref, ctx.seed, ctx.ed.uid, ctx.ed.c, ctx.ed.xi, ctx.ed.a)
    dnn = tr.delta[np.ix_(neg_node, neg_node)].astype(np.float64)
    rv = TK.ReachVote(ctx.ref, bank.basins, ctx.ed.c.numpy(), ctx.ed.xi, dnn)
    P = ctx.anchor
    d_pool = tr.delta[np.ix_(np.arange(n), neg_node)].astype(np.float64)
    v, ws = rv.vote(P.xi[:n], d_pool)
    abst = ws < np.quantile(ws, 0.1)
    keep = (v < rv.theta) & ~abst
    ctx._reach = (rv, tr, neg_node, torch.from_numpy(keep),
                  {"reach_theta": rv.theta, "reach_bw": rv.bw, "reach_keep_frac": float(keep.mean()),
                   "reach_abstain_frac": float(abst.mean())})
    return ctx._reach


def masked_anchor_loss(ctx: "Ctx", kind: str):
    """kind: map | dist (vote-masked pool), local (user's local anchor, pinned),
    local+map (both pools, pinned)."""
    B = ctx.batch
    parts = []
    if kind in ("map", "dist", "local+map"):
        _, keep, _ = get_vote(ctx)
        P = ctx.anchor
        parts.append((P.c[:N_VOTE_POOL][keep], P.xi[:N_VOTE_POOL][keep], P.a[:N_VOTE_POOL][keep]))
    if kind in ("local", "local+map"):
        parts.append(get_local(ctx))
    if kind == "reach":
        r = get_reach(ctx)
        if r is None:                      # one negative: no pairs, reduces to the recipe's vote mask
            _, keep, _ = get_vote(ctx)
        else:
            keep = r[3]
        P = ctx.anchor
        n = keep.shape[0]
        parts.append((P.c[:n][keep], P.xi[:n][keep], P.a[:n][keep]))
        kind = "dist"
    c = torch.cat([p[0] for p in parts])
    xi = torch.cat([p[1] for p in parts])
    a = torch.cat([p[2] for p in parts])
    kind = "dist" if kind == "dist" else "map"

    def loss(pol, g):
        j = _pick(c.shape[0], B, g)
        t = FlowPolicy.sample_time(B, g)
        noise = xi[j] if kind == "map" else torch.randn(a[j].shape, generator=g)
        return pol.flow_loss(a[j], noise, t, c[j]).mean()
    return loss


K_SDPO = 8


@torch.no_grad()
def sdpo_positives(arm: Arm, ctx: "Ctx"):
    """(N, K, 2) winners per negative: K fresh pi_ref samples at c_i (own stream), or the hop target (K=1)."""
    ed, ref = ctx.ed, ctx.ref
    if arm.sdpo_pos == "hop":
        out, stats = [], {}
        for i in range(len(ed.uid)):
            g = TG.row_gen(ctx.seed, ed.uid[i], 1)            # same stream as R_hop's target
            out.append(TG.target_one(ref, "hop", float(ed.c[i]), ed.xi[i], ed.a[i], g, stats=stats).reshape(1, 2))
        return torch.stack(out), stats
    pos = []
    for i in range(len(ed.uid)):
        z = torch.randn(K_SDPO, 2, generator=TG.row_gen(ctx.seed, ed.uid[i], 5))
        pos.append(ref.sample(z, torch.full((K_SDPO,), float(ed.c[i]))))
    return torch.stack(pos), {}


def sdpo_step_fn(arm: Arm, ctx: "Ctx", pos):
    """L = -log sigmoid(beta (r(a_w) - r(a_l))), r = l_ref - l_theta at a shared t; noise independent
    ("fresh") or the logged noise for both ("pin"). B/2 pairs per step (= B rows)."""
    ed, ref = ctx.ed, ctx.ref
    P = ctx.batch // 2
    N, K = pos.shape[0], pos.shape[1]

    beta_i = None
    if arm.track == "betaH":
        # beta per negative = (beta H-scale) / H_i,  H_i = E_t l_ref(a_w; xi_l, t)  (skeptic's calibration)
        with torch.no_grad():
            gH = gen(ctx.seed, 77_000)
            H = []
            for i in range(N):
                tt = FlowPolicy.sample_time(64, gH)
                H.append(float(ref.flow_loss(pos[i, 0].expand(64, -1), ed.xi[i].expand(64, -1), tt,
                                             ed.c[i].expand(64)).mean()))
        beta_i = arm.beta / torch.tensor(H).clamp_min(1e-6)

    def step(pol, g):
        i = _pick(N, P, g)
        k = _pick(K, P, g)
        aw, al, c = pos[i, k], ed.a[i], ed.c[i]
        t = FlowPolicy.sample_time(P, g)
        if arm.sdpo_noise == "pin":
            nw = nl = ed.xi[i]
        else:
            nw = torch.randn(aw.shape, generator=g)
            nl = torch.randn(al.shape, generator=g)
        if arm.track == "synbc":                   # control: BC on the winners only (fresh noise)
            return pol.flow_loss(aw, nw, t, c).mean()
        lw, ll = pol.flow_loss(aw, nw, t, c), pol.flow_loss(al, nl, t, c)
        if arm.track == "ungated":                 # control: rerouting + constant push, no gate
            return (lw - ll).mean()
        with torch.no_grad():
            rw0, rl0 = ref.flow_loss(aw, nw, t, c), ref.flow_loss(al, nl, t, c)
        margin = (rw0 - lw) - (rl0 - ll)
        b = beta_i[i] if beta_i is not None else arm.beta
        return -torch.nn.functional.logsigmoid(b * margin).mean()
    return step


@torch.no_grad()
def sdpo_diagnostics(arm: Arm, ctx: "Ctx", pol, pos, n_pairs=2048):
    """Post-training: where the margin comes from, gate weight on contaminated vs clean pairs (oracle used
    for diagnostics only), and where the logged noise now decodes."""
    ed, ref = ctx.ed, ctx.ref
    g = gen(ctx.seed, 78_000)
    N, K = pos.shape[0], pos.shape[1]
    i = _pick(N, n_pairs, g)
    k = _pick(K, n_pairs, g)
    aw, al, c = pos[i, k], ed.a[i], ed.c[i]
    t = FlowPolicy.sample_time(n_pairs, g)
    nw = nl = ed.xi[i]
    if arm.sdpo_noise != "pin":
        nw, nl = torch.randn(aw.shape, generator=g), torch.randn(al.shape, generator=g)
    rw = ref.flow_loss(aw, nw, t, c) - pol.flow_loss(aw, nw, t, c)
    rl = ref.flow_loss(al, nl, t, c) - pol.flow_loss(al, nl, t, c)
    contam = torch.from_numpy(np.isin(MIX.assign(aw, c), ed.forget))
    gate = torch.sigmoid(-(max(arm.beta, 1e-9)) * (rw - rl))
    a_end = pol.sample(ed.xi, ed.c)
    lab_end = MIX.assign(a_end, ed.c)
    d = {"sdpo_rw_mean": float(rw.mean()), "sdpo_rl_mean": float(rl.mean()),
         "sdpo_frac_rw_neg": float((rw < 0).float().mean()),
         "sdpo_gate_contam": float(gate[contam].mean()) if contam.any() else float("nan"),
         "sdpo_gate_clean": float(gate[~contam].mean()) if (~contam).any() else float("nan"),
         "sdpo_logged_end_in_forget": float(np.isin(lab_end, ed.forget).mean()),
         "sdpo_logged_end_off": float((lab_end == OFF).mean()),
         "sdpo_logged_end_dist_to_winner": float(torch.linalg.norm(a_end - pos[:, 0], dim=1).mean()),
         "sdpo_logged_end_dist_to_loser": float(torch.linalg.norm(a_end - ed.a, dim=1).mean())}
    return d


def step_fn(arm: Arm, ctx: Ctx, rows=None):
    B, ref, ed = ctx.batch, ctx.ref, ctx.ed
    if arm.kind == "sdpo":
        return sdpo_step_fn(arm, ctx, rows)
    if arm.kind == "pseudo":
        return masked_anchor_loss(ctx, arm.amask)
    if arm.kind == "maskbc":
        return mask_bc_loss(ctx, get_mask(ctx, arm.track)[0])

    def method(pol, g):
        if arm.kind in ("reroute", "pra"):
            i = _pick(rows.c.shape[0], B, g)
            t = FlowPolicy.sample_time(B, g)
            return pol.flow_loss(rows.target[i], rows.noise[i], t, rows.c[i]).mean()
        i = _pick(ed.c.shape[0], B, g)
        a, c = ed.a[i], ed.c[i]
        noise = torch.randn(a.shape, generator=g)
        t = FlowPolicy.sample_time(B, g)
        l_th = pol.flow_loss(a, noise, t, c)
        with torch.no_grad():
            l_ref = ref.flow_loss(a, noise, t, c)
        r = l_ref - l_th
        if arm.kind == "npo":
            return O.npo_term(r, arm.beta).mean()
        k = min(8, B)          # KTO undesirable-only; mismatched z0 = rolled rows (other conditions' actions)
        with torch.no_grad():
            ar, nr = torch.roll(a, 1, 0)[:k], torch.roll(noise, 1, 0)[:k]
            z0 = torch.clamp((ref.flow_loss(ar, nr, t[:k], c[:k]) - pol.flow_loss(ar, nr, t[:k], c[:k])).mean(), min=0.0)
        return (1.0 - torch.sigmoid(arm.beta * (z0 - r))).mean()

    if arm.amask:
        al = masked_anchor_loss(ctx, arm.amask)

        def masked(pol, g):
            return 0.5 * (method(pol, g) + al(pol, g))
        return masked
    if not arm.anchor:
        return method
    P = ctx.anchor

    def anchored(pol, g):
        lm = method(pol, g)
        j = _pick(P.c.shape[0], B, g)
        t = FlowPolicy.sample_time(B, g)
        la = pol.flow_loss(P.a[j], P.xi[j], t, P.c[j]).mean()
        return 0.5 * (lm + la)
    return anchored


def train_arm(arm: Arm, ctx: Ctx, probe=None):
    info: Dict = {}
    if arm.kind == "frozen":
        return ctx.ref, info
    rows = None
    if arm.kind == "pra":
        from . import track as TK
        tr, nc = get_tracker(ctx, arm.track)
        n, node_c, neg_node = _nodes(ctx)
        train_nodes = np.concatenate([neg_node, np.arange(0, n, 8)])
        c_, z_, t_, st = TK.pra_rows(ctx.ref, ctx.seed, tr, TK.flag_or(nc), train_nodes)
        rows = TG.Rows(c=c_, noise=z_, target=t_, diag=st)
        info.update({f"rows_{k}": v for k, v in st.items()})
        info.update(get_mask(ctx, arm.track)[1])
    elif arm.kind == "reroute":
        vetoes_ext = None
        if arm.rule == "hopR":
            r = get_reach(ctx)
            if r is None:
                rule = "hopB"
            else:
                rv, tr, neg_node, _, rd = r
                info.update(rd)
                vetoes_ext = []
                for i in range(len(ctx.ed.c)):
                    d_i = tr.delta[neg_node[i], neg_node].astype(np.float64)
                    vetoes_ext.append(lambda z, d_i=d_i: torch.from_numpy(
                        rv.vote(z, np.repeat(d_i[None], z.shape[0], 0))[0] >= rv.theta))
                rule = "hop"
        else:
            rule = arm.rule
        rows = TG.build(ctx.ref, MIX, ctx.seed, ctx.ed.uid, ctx.ed.c, ctx.ed.xi, ctx.ed.a, rule,
                        arm.support, transplant=arm.transplant, tx_conditions=ctx.pool.c[:4096],
                        d0=4 * MIX.mode_std, pu=ctx.pu, forget=ctx.ed.forget, vetoes_ext=vetoes_ext)
        info.update({f"rows_{k}": v for k, v in rows.diag.items()})
        if arm.mine:
            bank, keep, _ = get_vote(ctx)
            P = ctx.anchor
            flagged = torch.nonzero(~keep).flatten()[:N_MINE]
            mrows = TG.build_mined(ctx.ref, MIX, ctx.seed, bank, P.c[flagged], P.xi[flagged], P.a[flagged],
                                   uid_offset=50_000_000, forget=ctx.ed.forget)
            info.update({f"mined_{k}": v for k, v in mrows.diag.items()})
            if mrows.c.shape[0]:
                rows = TG.Rows(c=torch.cat([rows.c, mrows.c]), noise=torch.cat([rows.noise, mrows.noise]),
                               target=torch.cat([rows.target, mrows.target]), diag=rows.diag)
    if arm.kind == "sdpo":
        rows, st = sdpo_positives(arm, ctx)
        info.update({f"sdpo_{k}": v for k, v in st.items()})
        lab = MIX.assign(rows.reshape(-1, 2), np.repeat(ctx.ed.c.numpy(), rows.shape[1]))
        info["sdpo_pos_frac_in_forget"] = float(np.isin(lab, ctx.ed.forget).mean())     # diagnostic only
        info["sdpo_pos_frac_off"] = float((lab == OFF).mean())
    if arm.kind == "maskbc":
        info.update(get_mask(ctx, arm.track)[1])
    if arm.mine or arm.amask in ("map", "dist", "local+map") or (arm.amask == "reach" and get_reach(ctx) is None):
        info.update(get_vote(ctx)[2])
    if "local" in arm.amask:
        lc, _, la = get_local(ctx)
        info["local_anchor_n"] = int(lc.shape[0])
        info["local_anchor_frac_in_forget"] = float(np.isin(MIX.assign(la, lc), ctx.ed.forget).mean()) if lc.shape[0] else float("nan")
    fn = step_fn(arm, ctx, rows)
    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref)
    pol.train()
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(pol.parameters(), lr=ctx.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    g = gen(ctx.seed, P_TRAIN)
    hist = []
    for step in range(ctx.steps):
        lr = cosine(step, ctx.steps, min(500, ctx.steps // 10), ctx.lr, 0.1 * ctx.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        if probe is not None and (step % 1000 == 0 or step == ctx.steps - 1):
            pol.eval()
            hist.append({"step": step, "loss": float(loss.detach()), **probe(pol)})
            pol.train()
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    if arm.kind == "sdpo":
        info.update(sdpo_diagnostics(arm, ctx, pol, rows))
    info["train_history"] = hist
    return pol, info


# ------------------------------------------------------------------ metrics
class Bank:
    def __init__(self, ref, seed, ed: Edit, n=20_000):
        g = gen(seed, P_EVAL)
        self.c = torch.rand(n, generator=g)
        self.xi = torch.randn(n, 2, generator=g)
        self.ed = ed
        self.lab_ref = MIX.assign(ref.sample(self.xi, self.c), self.c)
        self.a_ref = ref.sample(self.xi, self.c)
        # condition distance from each eval point to the nearest logged negative OF THE SAME MODE
        # that pi_ref sends this eval noise to (only meaningful for eval points in forget modes)
        dc = torch.cdist(self.c.reshape(-1, 1), ed.c.reshape(-1, 1)).numpy()
        self.dc_any = dc.min(1)


def masses(lab):
    M = MIX.n_modes
    return np.array([np.mean(lab == j) for j in range(M)] + [np.mean(lab == OFF)])


def compute(bank: Bank, pol) -> Dict:
    ed = bank.ed
    F_, R = ed.forget, ed.retain
    a = pol.sample(bank.xi, bank.c)
    lab = MIX.assign(a, bank.c)
    m, mr = masses(lab), masses(bank.lab_ref)
    M = MIX.n_modes
    out = {f"mass_m{j}": float(m[j]) for j in range(M)}
    out["mass_off"] = float(m[M])
    out["neg_mass_ratio"] = float(m[F_].sum() / max(mr[F_].sum(), 1e-12))
    pR, qR = m[R] / max(m[R].sum(), 1e-12), mr[R] / max(mr[R].sum(), 1e-12)
    out["retain_tv"] = 0.5 * float(np.abs(pR - qR).sum())
    ideal = np.zeros(M + 1)
    ideal[R] = 1 / len(R)
    out["tv_to_ideal"] = 0.5 * float(np.abs(m - ideal).sum())
    out["tv_to_ideal_ref"] = 0.5 * float(np.abs(mr - ideal).sum())
    ref_neg, ref_ret = np.isin(bank.lab_ref, F_), np.isin(bank.lab_ref, R)
    same = lab == bank.lab_ref
    out["retain_noise_flip"] = float((~same)[ref_ret].mean())
    out["neg_noise_to_retain"] = float(np.isin(lab, R)[ref_neg].mean())
    out["neg_noise_still_forget"] = float(np.isin(lab, F_)[ref_neg].mean())
    out["disp_retain_mean"] = float(torch.linalg.norm(a - bank.a_ref, dim=1).numpy()[ref_ret].mean())
    # cross-condition generalisation: forgetting on eval points near vs far (in c) from any negative
    for nm, sel in (("near", bank.dc_any < 0.01), ("far", bank.dc_any > 0.05)):
        s = ref_neg & sel
        out[f"neg_still_forget_{nm}"] = float(np.isin(lab, F_)[s].mean()) if s.any() else float("nan")
        out[f"n_eval_neg_{nm}"] = int(s.sum())
    return out


def make_probe(bank: Bank, n=4000):
    sub = Bank.__new__(Bank)
    sub.c, sub.xi, sub.ed = bank.c[:n], bank.xi[:n], bank.ed
    sub.lab_ref, sub.a_ref, sub.dc_any = bank.lab_ref[:n], bank.a_ref[:n], bank.dc_any[:n]

    def probe(pol):
        r = compute(sub, pol)
        return {k: r[k] for k in ("neg_mass_ratio", "retain_tv", "mass_off", "tv_to_ideal")}
    return probe
