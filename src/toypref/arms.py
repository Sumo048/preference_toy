"""The arm registry and how each arm is trained.

Batch composition (one deliberate deviation, applied to every arm alike): the
preference repo samples rows i.i.d. from one index with per-row weights chosen so
the two groups carry equal summed weight at lambda = 1. With 1-32 negatives
against 512 positives that makes a 256-row batch carry 0-8 hugely weighted
negative rows; here every two-group arm instead draws HALF its batch from each
group and combines the two group means with the same lambda. The expectation is
identical; the variance is not.

Mixes: ``both`` = edit rows + T_p, ``neg`` = edit rows only (the negatives-only
regime). Every negatives-only arm is flagged ``negonly``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from . import objectives as O
from . import seeds
from .flow import FlowPolicy
from .reroute import build_rows
from .train import finetune


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str                 # frozen | bc | reroute | npo | kto | dpo | ga | rerank | online
    family: str
    note: str = ""
    target: str = ""          # reroute: resample4 | spread | guide
    support: str = "log"
    mix: str = "both"         # both | neg
    lambda_kl: float = 0.0
    beta: float = 0.0
    negonly: bool = False
    negative_control: bool = False
    #: uses more labelled negatives than the cell's n_neg (online loop)
    extra_label_budget: bool = False
    #: uses data no other arm has (oracle reference)
    privileged: bool = False
    #: unlabelled anchor (round "where does rerouting win"): "" | self_map | self_dist |
    #: nf_map | nf_dist. self = pi_ref's own samples (unlabelled); nf = those farther
    #: than d0 from every logged negative. map = pinned (xi, pi_ref(xi)) pairs
    #: (distils the noise->action map); dist = fresh noise (distils the density).
    anchor: str = ""
    lambda_anchor: float = 1.0
    #: run only by an explicit arm list (kept out of the default "all" sweep)
    round_: str = ""
    #: self-paired DPO (user idea): noise "fresh" | "pin"; positive "rand" | "nearnf"
    sdpo_noise: str = ""
    sdpo_pos: str = "rand"


ARMS: Dict[str, ArmSpec] = {}


def _add(s: ArmSpec):
    assert s.name not in ARMS, s.name
    ARMS[s.name] = s


# --- references ------------------------------------------------------------
_add(ArmSpec("B0_frozen", "frozen", "reference", "pi_ref unedited"))
_add(ArmSpec("ORACLE_retain_bc", "bc", "reference",
             "BC on 20k pi_ref samples in retained modes; the target a perfect edit approaches",
             privileged=True))

# --- baselines -------------------------------------------------------------
_add(ArmSpec("B1_filtered", "bc", "baseline", "BC on T_p only"))
for b in (0.1, 1.0, 10.0, 100.0):
    _add(ArmSpec(f"B3_npo_b{b:g}", "npo", "baseline", "NPO on T_n + lambda_+=1 BC on T_p", beta=b))
    _add(ArmSpec(f"B5_kto_b{b:g}", "kto", "baseline",
                 "KTO, lambda_D set by the 1.01 class-ratio rule, lambda_U=1", beta=b))
_add(ArmSpec("B3_npo_b100_negonly", "npo", "baseline", "pure NPO, no anchor", beta=100.0,
             mix="neg", negonly=True))
_add(ArmSpec("B5_kto_b100_negonly", "kto", "baseline", "KTO undesirable-only", beta=100.0,
             mix="neg", negonly=True))
for b in (1.0, 25.0):
    _add(ArmSpec(f"DPO_cpl_b{b:g}", "dpo", "baseline", "Flow-DPO on (T_p, T_n) pairs + BC on T_p", beta=b))
_add(ArmSpec("RM_bon8", "rerank", "baseline", "Bradley-Terry RM on (T_p,T_n) pairs, best-of-8 from pi_ref"))
_add(ArmSpec("B4_naive_ga", "ga", "baseline", "gradient ascent on T_n + BC on T_p",
             negative_control=True))

# --- rerouting: Ours and the 2x2x2 grid -----------------------------------
for mix in ("both", "neg"):
    _add(ArmSpec(f"R_ours_{mix}", "reroute", "reroute",
                 "ours_v2_r2: 4 fresh pi_ref resamples per negative, S-log",
                 target="resample4", support="log", mix=mix, negonly=mix == "neg"))
    for cand in ("spread", "guide"):
        for sup in ("log", "nhood"):
            _add(ArmSpec(f"R_{cand}_{sup}_{mix}", "reroute", "reroute_grid",
                         f"{cand} farthest-of-8 target, S-{sup}",
                         target=cand, support=sup, mix=mix, negonly=mix == "neg"))

# --- KL-to-reference (patch 0018) on the R_spread_nhood_neg recipe ---------
for lk in (0.1, 0.3, 1.0):
    _add(ArmSpec(f"R_spread_nhood_neg_kl{lk:g}", "reroute", "reroute_kl",
                 "R_spread_nhood_neg + lambda_kl (v - v_ref)^2 on a retain loader (T_p)",
                 target="spread", support="nhood", mix="neg", lambda_kl=lk, negonly=False))
    # negonly=False: the KL retain loader is T_p (labelled positives), so these arms
    # see positive data (corrected 2026-10-07; first reports listed them as neg-only)

# --- spherical nhood and S-select (inv-nhood branch) -----------------------
for s in (0.485, 0.867):
    _add(ArmSpec(f"R_spread_nhood_sph{s:g}_neg", "reroute", "reroute_support",
                 "spherical jitter sqrt(1-s^2) xi + s eps", target="spread",
                 support=f"nhood_sph_{s}", mix="neg", negonly=True))
_add(ArmSpec("R_spread_select0.867_neg", "reroute", "reroute_support",
             "4 of 16 spherical jitters whose decode is nearest a_neg", target="spread",
             support="select_0.867", mix="neg", negonly=True))

# --- inverse family (reroute-inverse, inv-nhood) ---------------------------
for rho in (0.5, 1.5, 3.0):
    for mix in ("both", "neg"):
        _add(ArmSpec(f"R_spread_inv_r{rho:g}_{mix}", "reroute", "reroute_inverse",
                     "Euler inversion of 4 actions in the L2 ball of radius rho about a_neg",
                     target="spread", support=f"inv_r{rho}", mix=mix, negonly=mix == "neg"))
for band, (lo, hi) in (("near", (0, 4)), ("mid", (8, 12)), ("far", (16, 20))):
    _add(ArmSpec(f"R_spread_reach_{band}_both", "reroute", "reroute_inverse",
                 f"S-reach: prior noises of decode-distance ranks {lo}:{hi} of 32",
                 target="spread", support=f"reach_{lo}_{hi}", mix="both"))
for t0 in (0.4, 0.6, 0.8):
    for mix in ("both", "neg"):
        _add(ArmSpec(f"R_spread_ninv_t{t0:g}_fresh_{mix}", "reroute", "reroute_inverse",
                     "noised partial inversion, fresh e, reversed t0 -> 1 (FRS)",
                     target="spread", support=f"ninv_{t0}_fresh", mix=mix, negonly=mix == "neg"))
for mix in ("both", "neg"):
    _add(ArmSpec(f"R_spread_ninv_t0.4_mix_{mix}", "reroute", "reroute_inverse",
                 "noised partial inversion, e = sqrt(.75) xi + .5 eps", target="spread",
                 support="ninv_0.4_mix", mix=mix, negonly=mix == "neg"))

# --- label-filtered targets (next step 2, launched 2026-10-07) -------------
# Same arms, but a candidate is eligible as a target only if the mode classifier
# puts its decode in a RETAINED mode. That label on candidate actions is
# information the repo's method does not have, hence extra_label_budget.
for mix in ("both", "neg"):
    _add(ArmSpec(f"R_ours_filt_{mix}", "reroute", "reroute_filtered",
                 "Ours with label-filtered resamples (rejection until retained)",
                 target="resample4_filt", support="log", mix=mix, negonly=mix == "neg",
                 extra_label_budget=True))
    for sup in ("log", "nhood"):
        _add(ArmSpec(f"R_spread_filt_{sup}_{mix}", "reroute", "reroute_filtered",
                     f"spread farthest-of-8 among RETAINED candidates, S-{sup}",
                     target="spread_filt", support=sup, mix=mix, negonly=mix == "neg",
                     extra_label_budget=True))
_add(ArmSpec("R_spread_filt_nhood_neg_kl1", "reroute", "reroute_filtered",
             "R_spread_filt_nhood_neg + lambda_kl = 1 on the T_p retain loader",
             target="spread_filt", support="nhood", mix="neg", lambda_kl=1.0, negonly=False,
             extra_label_budget=True))

FILTERED_ARMS = [n for n, s in ARMS.items() if s.family == "reroute_filtered"]


def depends_on_tp(spec: "ArmSpec") -> bool:
    """Arms whose training touches T_p (BC/anchor rows, pairs, RM, KL retain loader)."""
    if spec.kind == "frozen":
        return True          # kept for pairing, cheap
    if spec.privileged:
        return False         # ORACLE uses its own 20k pi_ref retain samples
    if spec.kind in ("bc", "dpo", "rerank", "ga"):
        return True
    if spec.kind in ("npo", "kto"):
        return not spec.negonly
    if spec.kind in ("reroute", "online"):
        return spec.mix == "both" or spec.lambda_kl > 0
    return False


# --- online loop (scripts/m3/online_round.py) ------------------------------
_add(ArmSpec("R_online2_spread_nhood_neg", "online", "reroute_online",
             "2 rounds: roll out pi_1, add its n_neg/mode new negatives, retrain from pi_ref "
             "on the accumulated set; targets from frozen pi_ref", target="spread",
             support="nhood", mix="neg", negonly=True, extra_label_budget=True))


# --- round "where does rerouting win" (docs/REROUTE_WINS_PREREG.md) -------
RW = "rw"
ANCHORS = ("self_map", "self_dist", "nf_map", "nf_dist")
_RW_NEG_METHODS = {  # name stem -> ArmSpec kwargs (all negatives-only)
    "B3_npo_b10_negonly": dict(kind="npo", family="baseline", beta=10.0),
    "B3_npo_b100_negonly": dict(kind="npo", family="baseline", beta=100.0),
    "B5_kto_b10_negonly": dict(kind="kto", family="baseline", beta=10.0),
    "B5_kto_b100_negonly": dict(kind="kto", family="baseline", beta=100.0),
    "R_ours_neg": dict(kind="reroute", family="reroute", target="resample4", support="log"),
    "R_spread_log_neg": dict(kind="reroute", family="reroute_grid", target="spread", support="log"),
    "R_spread_nhood_neg": dict(kind="reroute", family="reroute_grid", target="spread", support="nhood"),
    "R_ours_nf_neg": dict(kind="reroute", family="reroute_rw", target="resample4_nf", support="log"),
    "R_spread_nf_log_neg": dict(kind="reroute", family="reroute_rw", target="spread_nf", support="log"),
    "R_near_nf_log_neg": dict(kind="reroute", family="reroute_rw", target="near_nf", support="log"),
    "R_near_nf_nhood_neg": dict(kind="reroute", family="reroute_rw", target="near_nf", support="nhood"),
}
for stem, kw in _RW_NEG_METHODS.items():
    kw = dict(kw)
    kind, fam = kw.pop("kind"), kw.pop("family")
    mix = "neg"
    for anc in ANCHORS:
        _add(ArmSpec(f"{stem}_A{anc}", kind, fam, f"{stem} + unlabelled anchor {anc}",
                     mix=mix, negonly=True, anchor=anc, round_=RW, **kw))
# un-anchored versions that do not exist yet
for stem in ("B3_npo_b10_negonly", "B5_kto_b10_negonly", "R_ours_nf_neg", "R_spread_nf_log_neg",
             "R_near_nf_log_neg", "R_near_nf_nhood_neg"):
    kw = dict(_RW_NEG_METHODS[stem])
    kind, fam = kw.pop("kind"), kw.pop("family")
    _add(ArmSpec(stem, kind, fam, f"{stem}, no anchor", mix="neg", negonly=True, round_=RW, **kw))
# the anchors alone: nf = BC on pseudo-positives (pi_ref samples far from every negative)
for anc in ("nf_map", "nf_dist"):
    _add(ArmSpec(f"PSEUDO_{anc}", "anchor_only", "baseline",
                 "label-free baseline: flow-matching on pi_ref samples farther than d0 from every "
                 "logged negative", negonly=True, anchor=anc, round_=RW))
# wider / denser neighbourhoods for the label-free near target
for sup in ("nhoodx_0.6_16", "nhoodx_1.0_16"):
    for anc in ("self_map", "nf_map"):
        _add(ArmSpec(f"R_near_nf_{sup}_neg_A{anc}", "reroute", "reroute_rw",
                     "near_nf target, wide jitter support", target="near_nf", support=sup,
                     mix="neg", negonly=True, anchor=anc, round_=RW))
# positives + negatives: new target rules with the T_p anchor
for stem, tgt, sup, extra in (("R_near_nf_log_both", "near_nf", "log", False),
                              ("R_near_nf_nhood_both", "near_nf", "nhood", False),
                              ("R_spread_nf_log_both", "spread_nf", "log", False),
                              ("R_ours_nf_both", "resample4_nf", "log", False),
                              ("R_near_filt_log_both", "near_filt", "log", True)):
    _add(ArmSpec(stem, "reroute", "reroute_rw", f"{tgt} target, S-{sup}, mix both",
                 target=tgt, support=sup, mix="both", extra_label_budget=extra, round_=RW))
_add(ArmSpec("R_near_filt_log_neg_Anf_map", "reroute", "reroute_rw",
             "label-filtered near target + nf_map anchor", target="near_filt", support="log",
             mix="neg", negonly=True, anchor="nf_map", extra_label_budget=True, round_=RW))

RW_ARMS = [n for n, s_ in ARMS.items() if s_.round_ == RW]

# --- self-paired DPO (user idea, 2026-10-09) ---------------------------------
for _b in (1.0, 25.0):
    for _nz in ("fresh", "pin"):
        _add(ArmSpec(f"SDPO_{_nz}_b{_b:g}", "sdpo", "baseline",
                     "self-paired DPO: loser = logged negative, winner = fresh pi_ref sample",
                     beta=_b, mix="neg", negonly=True, sdpo_noise=_nz, round_="sdpo"))
_add(ArmSpec("SDPO_nearnf_pin_b1", "sdpo", "baseline",
             "self-paired DPO with the near_nf target as winner (logged noise shared)",
             beta=1.0, mix="neg", negonly=True, sdpo_noise="pin", sdpo_pos="nearnf", round_="sdpo"))
SDPO_ARMS = [n for n, s_ in ARMS.items() if s_.round_ == "sdpo"]


# =========================================================================
@dataclass
class ArmContext:
    ref: FlowPolicy
    seed: int
    xi_neg: torch.Tensor
    a_neg: torch.Tensor
    neg_pool_idx: np.ndarray
    a_pos: torch.Tensor
    forget: List[int]
    mix_cfg: object
    tcfg: object
    oracle_a: Optional[torch.Tensor] = None
    online_pool_fn: Optional[object] = None   # policy -> RolloutPool
    n_neg_per_mode: int = 0
    #: unlabelled pi_ref (xi, a) pairs for the anchors, and which are farther than
    #: d0 from every logged negative ("nf")
    anchor_xi: Optional[torch.Tensor] = None
    anchor_a: Optional[torch.Tensor] = None
    anchor_nf: Optional[torch.Tensor] = None
    d0: Optional[float] = None
    typical: Optional[object] = None


def _f(x):
    return float(x.detach())


def _pick(n_rows: int, k: int, g):
    return torch.randint(0, n_rows, (k,), generator=g)


def _bc_loss(policy, a_all, k, g):
    a = a_all[_pick(a_all.shape[0], k, g)]
    noise = torch.randn(a.shape, generator=g)
    t = FlowPolicy.sample_time(k, g)
    return policy.flow_loss(a, noise, t).mean()


def reroute_step_fn(ctx: ArmContext, rows, spec: ArmSpec):
    B = ctx.tcfg.batch
    n_e = B if spec.mix == "neg" else B // 2

    def step(policy, g):
        idx = _pick(rows.noise.shape[0], n_e, g)
        t = FlowPolicy.sample_time(n_e, g)
        l_e = policy.flow_loss(rows.target[idx], rows.noise[idx], t).mean()
        stats = {"l_edit": _f(l_e)}
        if spec.mix == "both":
            l_p = _bc_loss(policy, ctx.a_pos, B - n_e, g)
            loss = 0.5 * (l_e + l_p)        # lambda_- = 1
            stats["l_pos"] = _f(l_p)
        else:
            loss = l_e
        if spec.lambda_kl > 0:
            a = ctx.a_pos[_pick(ctx.a_pos.shape[0], B, g)]
            noise = torch.randn(a.shape, generator=g)
            tk = FlowPolicy.sample_time(B, g)
            kl = O.kl_velocity(policy, ctx.ref, a, noise, tk).mean()
            loss = loss + spec.lambda_kl * kl
            stats["kl"] = _f(kl)
        return loss, stats
    return step


def npo_step_fn(ctx, spec):
    B = ctx.tcfg.batch
    n_n = B if spec.negonly else B // 2

    def step(policy, g):
        a = ctx.a_neg[_pick(ctx.a_neg.shape[0], n_n, g)]
        noise = torch.randn(a.shape, generator=g)
        t = FlowPolicy.sample_time(n_n, g)
        r, _ = O.surrogate_r(policy, ctx.ref, a, noise, t)
        npo = O.npo_term(r, spec.beta).mean()
        stats = {"r_neg": _f(r.mean()), "W_neg": _f((2 * torch.sigmoid(spec.beta * r)).mean())}
        if spec.negonly:
            return npo, stats
        bc = _bc_loss(policy, ctx.a_pos, B - n_n, g)
        stats["bc"] = _f(bc)
        return 0.5 * (npo + bc), stats          # lambda_+ = 1, normalised by 1 + lambda_+
    return step


def kto_step_fn(ctx, spec):
    B = ctx.tcfg.batch
    n_n = B if spec.negonly else B // 2
    lambda_u = 1.0
    # class-ratio rule lambda_D n_D / (lambda_U n_U) = 1.01 with n_D = n_U per batch
    lambda_d = 1.01 * lambda_u

    def step(policy, g):
        a_n = ctx.a_neg[_pick(ctx.a_neg.shape[0], n_n, g)]
        parts = [a_n]
        is_neg = [torch.ones(n_n, dtype=torch.bool)]
        if not spec.negonly:
            parts.append(ctx.a_pos[_pick(ctx.a_pos.shape[0], B - n_n, g)])
            is_neg.append(torch.zeros(B - n_n, dtype=torch.bool))
        a = torch.cat(parts)
        neg = torch.cat(is_neg)
        perm = torch.randperm(a.shape[0], generator=g)
        a, neg = a[perm], neg[perm]
        noise = torch.randn(a.shape, generator=g)
        t = FlowPolicy.sample_time(a.shape[0], g)
        r, _ = O.surrogate_r(policy, ctx.ref, a, noise, t)
        z0 = O.kto_z0(policy, ctx.ref, a, noise, t)
        loss = O.kto_values(r, z0, neg, spec.beta, lambda_d, lambda_u).mean()
        stats = {"z0": _f(z0), "r_neg": _f(r[neg].mean())}
        if (~neg).any():
            stats["r_pos"] = _f(r[~neg].mean())
        return loss, stats
    return step


def dpo_step_fn(ctx, spec):
    B = ctx.tcfg.batch
    n_pairs = B // 4

    def step(policy, g):
        bc = _bc_loss(policy, ctx.a_pos, B // 2, g)
        a_w = ctx.a_pos[_pick(ctx.a_pos.shape[0], n_pairs, g)]
        a_l = ctx.a_neg[_pick(ctx.a_neg.shape[0], n_pairs, g)]
        t = FlowPolicy.sample_time(n_pairs, g)           # shared within a pair (patch 0014)
        r_w, _ = O.surrogate_r(policy, ctx.ref, a_w, torch.randn(a_w.shape, generator=g), t)
        r_l, _ = O.surrogate_r(policy, ctx.ref, a_l, torch.randn(a_l.shape, generator=g), t)
        pair = O.dpo_pair_term(r_w, r_l, spec.beta).mean()
        return bc + pair, {"bc": _f(bc), "pair": _f(pair), "margin": _f((r_w - r_l).mean())}
    return step


K_SDPO = 8


@torch.no_grad()
def sdpo_positives(ctx, spec):
    """(N, K, 2) winners per negative: K fresh pi_ref samples (own per-row stream), or the near_nf target."""
    if spec.sdpo_pos == "nearnf":
        rows = build_rows(ctx.ref, ctx.xi_neg, ctx.a_neg, ctx.neg_pool_idx, "near_nf", "log", ctx.seed,
                          ctx.mix_cfg, ctx.forget, d0=ctx.d0, typical=ctx.typical)
        return rows.target.reshape(-1, 1, rows.target.shape[1])
    out = []
    for i in range(ctx.xi_neg.shape[0]):
        g = seeds.row_gen(ctx.seed, int(ctx.neg_pool_idx[i]), 5)
        out.append(ctx.ref.sample(torch.randn(K_SDPO, ctx.xi_neg.shape[1], generator=g)))
    return torch.stack(out)


def sdpo_step_fn(ctx, spec, pos):
    P = ctx.tcfg.batch // 2
    N, K = pos.shape[0], pos.shape[1]

    def step(policy, g):
        i = _pick(N, P, g)
        k = _pick(K, P, g)
        aw, al = pos[i, k], ctx.a_neg[i]
        t = FlowPolicy.sample_time(P, g)
        if spec.sdpo_noise == "pin":
            nw = nl = ctx.xi_neg[i]
        else:
            nw = torch.randn(aw.shape, generator=g)
            nl = torch.randn(al.shape, generator=g)
        rw, _ = O.surrogate_r(policy, ctx.ref, aw, nw, t)
        rl, _ = O.surrogate_r(policy, ctx.ref, al, nl, t)
        loss = O.dpo_pair_term(rw, rl, spec.beta).mean()
        return loss, {"margin": _f((rw - rl).mean())}
    return step


def ga_step_fn(ctx, spec):
    B = ctx.tcfg.batch

    def step(policy, g):
        neg = _bc_loss(policy, ctx.a_neg, B // 2, g)
        pos = _bc_loss(policy, ctx.a_pos, B // 2, g)
        return -neg + pos, {"neg_bc": _f(neg), "pos_bc": _f(pos)}
    return step


def anchor_loss_fn(ctx: ArmContext, anchor: str):
    """Flow-matching on unlabelled pi_ref samples. ``*_map`` keeps each sample's own
    noise (distils the map), ``*_dist`` draws fresh noise (distils the density)."""
    src, kind = anchor.split("_")
    if src == "self":
        xi, a = ctx.anchor_xi, ctx.anchor_a
    elif src == "nf":
        xi, a = ctx.anchor_xi[ctx.anchor_nf], ctx.anchor_a[ctx.anchor_nf]
    else:
        raise ValueError(anchor)
    if a.shape[0] == 0:
        raise ValueError("anchor pool empty")
    B = ctx.tcfg.batch

    def loss(policy, g):
        idx = _pick(a.shape[0], B, g)
        noise = xi[idx] if kind == "map" else torch.randn(a[idx].shape, generator=g)
        if kind not in ("map", "dist"):
            raise ValueError(anchor)
        t = FlowPolicy.sample_time(B, g)
        return policy.flow_loss(a[idx], noise, t).mean()
    return loss


def with_anchor(fn, ctx: ArmContext, spec: ArmSpec):
    """(l_method + lambda_a l_anchor) / (1 + lambda_a): same normalisation as the
    lambda_- = 1 balance of the `both` mix."""
    al = anchor_loss_fn(ctx, spec.anchor)
    lam = spec.lambda_anchor

    def step(policy, g):
        l_m, stats = fn(policy, g)
        l_a = al(policy, g)
        stats = dict(stats)
        stats["l_anchor"] = _f(l_a)
        return (l_m + lam * l_a) / (1 + lam), stats
    return step


# =========================================================================
class RewardModel(torch.nn.Module):
    """BTRewardHead with the state features removed: standardised action -> 256 ReLU -> 1.

    The repo's head LayerNorms its 2048-d prefix features; that LayerNorm is NOT
    carried over to the action. On a 2-d input a LayerNorm maps every point to
    roughly sign(x1 - x2) and the RM cannot separate the modes (first sweep: mean
    training-pair accuracy 0.70, min 0.51; results discarded and rerun)."""

    def __init__(self, dim=2, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))

    def forward(self, a):
        return self.net((a - self.mu) / self.sd).squeeze(-1)


def train_reward_model(ctx: ArmContext, steps=3000, batch=256, lr=1e-3, wd=1e-4):
    torch.manual_seed(ctx.seed)
    rm = RewardModel(ctx.a_pos.shape[1])
    allx = torch.cat([ctx.a_pos, ctx.a_neg])
    rm.mu.copy_(allx.mean(0))
    rm.sd.copy_(allx.std(0).clamp_min(1e-6))
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=wd)
    g = seeds.gen(ctx.seed, seeds.RM_TRAIN)
    for _ in range(steps):
        w = ctx.a_pos[_pick(ctx.a_pos.shape[0], batch, g)]
        l = ctx.a_neg[_pick(ctx.a_neg.shape[0], batch, g)]
        loss = -torch.nn.functional.logsigmoid(rm(w) - rm(l)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    rm.eval()
    with torch.no_grad():
        acc = float((rm(ctx.a_pos)[:, None] > rm(ctx.a_neg)[None, :]).float().mean())
    return rm, {"rm_train_pair_acc": acc, "rm_final_loss": _f(loss)}


class BoNSampler:
    """Best-of-N from frozen pi_ref. Candidate 0 is the eval noise itself, so the
    per-noise transition metrics stay paired with B0."""

    def __init__(self, ref, rm, seed, n=8):
        self.ref, self.rm, self.seed, self.n = ref, rm, seed, n

    @torch.no_grad()
    def decode(self, Z):
        g = seeds.gen(self.seed, seeds.BON)
        N, D = Z.shape
        extra = torch.randn(N * (self.n - 1), D, generator=g)
        cand = torch.cat([Z[:, None, :], self.ref.sample(extra).reshape(N, self.n - 1, D)], 1)
        cand[:, 0] = self.ref.sample(Z)
        score = self.rm(cand.reshape(-1, D)).reshape(N, self.n)
        return cand[torch.arange(N), score.argmax(1)]


class PolicySampler:
    def __init__(self, policy):
        self.policy = policy

    def decode(self, Z):
        return self.policy.sample(Z)


# =========================================================================
def run_arm(spec: ArmSpec, ctx: ArmContext, log=None, probe=None):
    """-> (sampler, policy or None, info dict)."""
    info: Dict = {"arm": spec.name}
    if spec.kind == "frozen":
        return PolicySampler(ctx.ref), ctx.ref, info
    if spec.kind == "rerank":
        rm, d = train_reward_model(ctx)
        info.update(d)
        return BoNSampler(ctx.ref, rm, ctx.seed), None, info

    if spec.kind == "bc":
        data = ctx.oracle_a if spec.privileged else ctx.a_pos
        B = ctx.tcfg.batch
        fn = lambda pol, g: (_bc_loss(pol, data, B, g), {})
    elif spec.kind == "anchor_only":
        fn = lambda pol, g: (anchor_loss_fn(ctx, spec.anchor)(pol, g), {})
    elif spec.kind in ("reroute", "online"):
        rows = build_rows(ctx.ref, ctx.xi_neg, ctx.a_neg, ctx.neg_pool_idx, spec.target,
                          spec.support, ctx.seed, ctx.mix_cfg, ctx.forget, d0=ctx.d0,
                          typical=ctx.typical)
        info.update({f"rows_{k}": v for k, v in rows.diag.items()})
        fn = reroute_step_fn(ctx, rows, spec)
    elif spec.kind == "npo":
        fn = npo_step_fn(ctx, spec)
    elif spec.kind == "kto":
        fn = kto_step_fn(ctx, spec)
    elif spec.kind == "dpo":
        fn = dpo_step_fn(ctx, spec)
    elif spec.kind == "sdpo":
        pos = sdpo_positives(ctx, spec)
        from .mixture import assign_modes
        lab = assign_modes(ctx.mix_cfg, pos.reshape(-1, pos.shape[-1]))
        info["sdpo_pos_frac_in_forget"] = float(np.isin(lab, ctx.forget).mean())   # diagnostic only
        info["sdpo_pos_frac_off"] = float((lab == -1).mean())
        fn = sdpo_step_fn(ctx, spec, pos)
    elif spec.kind == "ga":
        fn = ga_step_fn(ctx, spec)
    else:
        raise ValueError(spec.kind)
    if spec.anchor and spec.kind != "anchor_only":
        fn = with_anchor(fn, ctx, spec)
    if spec.anchor:
        info["anchor_pool_size"] = int(ctx.anchor_a.shape[0] if spec.anchor.startswith("self")
                                       else int(ctx.anchor_nf.sum()))

    pol, hist = finetune(ctx.ref, fn, ctx.tcfg, ctx.seed, log=log, probe=probe)
    info["train_history"] = hist
    info["diverged"] = any(h.get("diverged") for h in hist)

    if spec.kind == "online":
        pol, extra = _online_round2(spec, ctx, pol, rows, log, probe)
        info.update(extra)
    return PolicySampler(pol), pol, info


def _online_round2(spec, ctx, pi1, rows1, log, probe=None):
    """Round 2 of the online loop: roll out pi_1, keep the first n_neg rollouts per
    forget mode that pi_1 still sends there, reroute them against FROZEN pi_ref,
    and retrain from pi_ref on rounds 1+2 together."""
    pool = ctx.online_pool_fn(pi1)
    new = []
    found = {}
    for m in ctx.forget:
        idx = np.flatnonzero(pool.mode == m)[:ctx.n_neg_per_mode]
        found[int(m)] = int(len(idx))
        new.append(idx)
    new = np.concatenate(new) if new else np.zeros(0, dtype=int)
    extra = {"online_r2_new_negatives": int(len(new)), "online_r2_found_per_mode": found}
    if len(new) == 0:
        extra["online_r2_skipped"] = True
        return pi1, extra
    rows2 = build_rows(ctx.ref, pool.xi[new], pool.a[new], new, spec.target, spec.support,
                       ctx.seed, ctx.mix_cfg, ctx.forget, round_=1, d0=ctx.d0, typical=ctx.typical)
    from .reroute import RerouteRows
    acc = RerouteRows(noise=torch.cat([rows1.noise, rows2.noise]),
                      target=torch.cat([rows1.target, rows2.target]),
                      neg_row=torch.cat([rows1.neg_row, rows2.neg_row + int(rows1.neg_row.max()) + 1]),
                      slot=torch.cat([rows1.slot, rows2.slot]))
    pi2, hist2 = finetune(ctx.ref, reroute_step_fn(ctx, acc, spec), ctx.tcfg, ctx.seed, log=log, probe=probe)
    extra["train_history_r2"] = hist2
    return pi2, extra
