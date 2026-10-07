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


def forget_modes(seed, k):
    return sorted(int(m) for m in nprng(seed, P_PERM).permutation(MIX.n_modes)[:k])


@dataclass
class Edit:
    forget: List[int]
    retain: List[int]
    uid: np.ndarray
    c: torch.Tensor
    xi: torch.Tensor
    a: torch.Tensor


def editset(pool: Pool, seed, k, n_neg) -> Edit:
    F_ = forget_modes(seed, k)
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


def step_fn(arm: Arm, ctx: Ctx, rows=None):
    B, ref, ed = ctx.batch, ctx.ref, ctx.ed

    def method(pol, g):
        if arm.kind == "reroute":
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
    if arm.kind == "reroute":
        rows = TG.build(ctx.ref, MIX, ctx.seed, ctx.ed.uid, ctx.ed.c, ctx.ed.xi, ctx.ed.a, arm.rule,
                        arm.support, transplant=arm.transplant, tx_conditions=ctx.pool.c[:4096],
                        d0=4 * MIX.mode_std, pu=ctx.pu, forget=ctx.ed.forget)
        info.update({f"rows_{k}": v for k, v in rows.diag.items()})
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
