"""Negatives-only arms for robo2d and the shared fine-tune loop.

Inputs of every arm: the frozen base ``ref``, the experiment seed and the logged negative rows
(``data.NegRows``: observation, logged noise, logged chunk, ids). Nothing else -- no env, no
privileged per-row facts. Every row of a negative episode is a negative (credit assignment left in).

Arms
  B0_frozen            the base
  NPO_b{10,100}_neg    (2/beta) softplus(beta r), r = l_ref - l_theta at shared fresh (xi, t)
  KTO_b{10,100}_neg    undesirable-only KTO, z0 from rolled (mismatched) rows
  R_ours_neg           reroute: 4 fresh base decodes at the row's observation, trained from the logged noise
  R_hop_bsup_neg       reroute: hop target, basin support
  R_hop_bsup_neg_AL    + local anchor (64 base samples at each row's observation outside the bad basin,
                       pinned to their noise), loss = 1/2 reroute + 1/2 anchor
  SDPO_*               the user's self-paired DPO: loser = the logged negative chunk, winner = a base
                       sample at the SAME observation; -log sigmoid(beta (r_w - r_l)), shared t in a pair
                       fresh  independent fresh noise for winner and loser (8 fixed positives per row)
                       pin    both evaluated on the row's LOGGED noise (8 fixed positives per row)
                       hop    winner = the R_hop_bsup target, pinned
  (stress round)
  SDPO_fresh_b{5,10,50,100}, NPO/KTO_b{30,300}   strength sweep
  RAND_<arm>           control: the base plus a random edit with the SAME per-tensor norm as <arm>'s
                       trained edit (prices "any small edit helps")
  FILTER_rej           control, inference time: K = 4 candidate chunks per decision (noise xi Q_k,
                       Q_0 = I); a candidate is rejected when it lies within tau of a logged negative
                       chunk whose observation is within r_o; the first unrejected candidate is used
                       (all rejected: the farthest). No training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .. import objectives as O
from ..flow import FlowPolicy
from ..train import cosine
from . import seeds as S
from . import targets as TG
from .data import NegRows
from .policy import ChunkFlow


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                 # frozen | npo | kto | reroute | sdpo
    beta: float = 0.0
    rule: str = ""            # reroute: ours | hop
    support: str = "log"      # reroute: log | bsup
    local: bool = False       # + local anchor
    pos: str = ""             # sdpo: fresh | hop
    noise: str = ""           # sdpo: fresh | pin
    of: str = ""              # rand: the arm whose edit norm is matched
    n_win: int = 1            # sdpo noise="own": winners averaged per row (r_w = mean over winners)
    half: str = ""            # sdpo ablation: "w" winner term only, "l" loser term only


ARMS: Dict[str, Arm] = {a.name: a for a in [
    Arm("B0_frozen", "frozen"),
    Arm("NPO_b10_neg", "npo", beta=10.0), Arm("NPO_b100_neg", "npo", beta=100.0),
    Arm("KTO_b10_neg", "kto", beta=10.0), Arm("KTO_b100_neg", "kto", beta=100.0),
    Arm("R_ours_neg", "reroute", rule="ours", support="log"),
    Arm("R_hop_bsup_neg", "reroute", rule="hop", support="bsup"),
    Arm("R_hop_bsup_neg_AL", "reroute", rule="hop", support="bsup", local=True),
    Arm("SDPO_fresh_b1", "sdpo", beta=1.0, pos="fresh", noise="fresh"),
    Arm("SDPO_fresh_b25", "sdpo", beta=25.0, pos="fresh", noise="fresh"),
    Arm("SDPO_pin_b1", "sdpo", beta=1.0, pos="fresh", noise="pin"),
    Arm("SDPO_pin_b25", "sdpo", beta=25.0, pos="fresh", noise="pin"),
    Arm("SDPO_hop_pin_b1", "sdpo", beta=1.0, pos="hop", noise="pin"),
    Arm("SDPO_hop_pin_b25", "sdpo", beta=25.0, pos="hop", noise="pin"),   # beta 25 was the better gate in the cond toy
    # stress round: strength sweep and controls
    *[Arm(f"SDPO_fresh_b{b}", "sdpo", beta=float(b), pos="fresh", noise="fresh") for b in (5, 10, 50, 100)],
    *[Arm(f"NPO_b{b}_neg", "npo", beta=float(b)) for b in (30, 300)],
    *[Arm(f"KTO_b{b}_neg", "kto", beta=float(b)) for b in (30, 300)],
    Arm("RAND_SDPO_fresh_b25", "rand", of="SDPO_fresh_b25"),
    Arm("RAND_NPO_b100_neg", "rand", of="NPO_b100_neg"),
    Arm("RAND_SDPO_pin_b25", "rand", of="SDPO_pin_b25"),
    Arm("RAND_KTO_b100_neg", "rand", of="KTO_b100_neg"),
    Arm("FILTER_rej", "filter"),
    # own-noise pinned SDPO (Soumojit, 2026-10-16): loser at its logged noise, each winner at the noise that
    # generated it; r_w = mean over n_win winners
    Arm("SDPO_own_b25_K1", "sdpo", beta=25.0, pos="fresh", noise="own", n_win=1),
    Arm("SDPO_own_b25_K8", "sdpo", beta=25.0, pos="fresh", noise="own", n_win=8),
    # drift experiment D: which half of the SDPO loss moves the policy
    Arm("SDPO_pin_b25_W", "sdpo", beta=25.0, pos="fresh", noise="pin", half="w"),
    Arm("SDPO_pin_b25_L", "sdpo", beta=25.0, pos="fresh", noise="pin", half="l"),
    Arm("SDPO_fresh_b25_W", "sdpo", beta=25.0, pos="fresh", noise="fresh", half="w"),
    Arm("SDPO_fresh_b25_L", "sdpo", beta=25.0, pos="fresh", noise="fresh", half="l"),

]}

P_RAND = 90_000
P_FILTER = 91_000
FILTER_K = 4
FILTER_TAU = 0.5          # x median pairwise base-decode distance at the negative's observation
FILTER_RO = 1.0           # x median 1-NN distance between logged negative observations


class RejectFilter:
    """Inference-time rejection of chunks near logged negatives. Duck-types ChunkFlow.sample."""

    def __init__(self, ref: ChunkFlow, seed: int, rows: NegRows):
        self.ref = ref
        g = S.gen(seed, P_FILTER)
        d = rows.a.shape[1]
        Q = [torch.eye(d)] + [torch.linalg.qr(torch.randn(d, d, generator=g))[0] for _ in range(FILTER_K - 1)]
        self.Q = torch.stack(Q)                                                   # (K, d, d), orthogonal
        self.no, self.na = rows.obs.float(), rows.a.float()
        P = TG.fresh_decodes(ref, seed, rows, TG.N_SDPO_POS, S.ROW_SDPO)          # (N, 8, d)
        pd_ = torch.cdist(P, P)
        iu = torch.triu_indices(P.shape[1], P.shape[1], 1)
        self.tau = FILTER_TAU * pd_[:, iu[0], iu[1]].median(1).values             # (N,)
        dn = torch.cdist(self.no, self.no)
        dn.fill_diagonal_(float("inf"))
        self.r_o = float(FILTER_RO * dn.min(1).values.median())
        self.stats = {"filter_tau_median": float(self.tau.median()), "filter_r_o": self.r_o}

    @torch.no_grad()
    def sample(self, xi, o, batch=8192):
        out = []
        for s in range(0, xi.shape[0], batch):
            x, oo = xi[s:s + batch].float(), o[s:s + batch].float()
            n = x.shape[0]
            cand = self.ref.sample(torch.einsum("nd,kde->knd", x, self.Q).reshape(-1, x.shape[1]),
                                   oo.repeat(FILTER_K, 1)).reshape(FILTER_K, n, -1)
            near = torch.cdist(oo, self.no) <= self.r_o                           # (n, N)
            dmin = torch.full((FILTER_K, n), float("inf"))
            hit = near.any(1)
            if hit.any():
                ii = torch.nonzero(hit).squeeze(1)
                da = torch.cdist(cand[:, ii], self.na.unsqueeze(0).expand(FILTER_K, -1, -1))   # (K, h, N)
                rel = da / self.tau.view(1, 1, -1)
                rel = torch.where(near[ii].unsqueeze(0), rel, torch.full_like(rel, float("inf")))
                dmin[:, ii] = rel.min(2).values                                   # in units of tau
            ok = dmin >= 1.0
            first = torch.where(ok.any(0), ok.float().argmax(0), dmin.argmax(0))
            out.append(cand[first, torch.arange(n)])
        return torch.cat(out) if out else xi.new_zeros((0, xi.shape[1]))


@dataclass
class Ctx:
    ref: ChunkFlow
    seed: int
    rows: NegRows
    steps: int = 5000
    batch: int = 256
    lr: float = 1e-4
    cache: Optional[TG.HopCache] = None


def get_cache(ctx: Ctx) -> TG.HopCache:
    if ctx.cache is None:
        ctx.cache = TG.build_hop_cache(ctx.ref, ctx.seed, ctx.rows)
    assert np.array_equal(ctx.cache.uid, ctx.rows.uid), "hop cache built for different rows"
    return ctx.cache


def _pick(n, k, g):
    return torch.randint(0, n, (k,), generator=g)


def reroute_rows(arm: Arm, ctx: Ctx):
    R = ctx.rows
    info: Dict = {}
    if arm.rule == "ours":
        tg = TG.fresh_decodes(ctx.ref, ctx.seed, R, TG.N_OURS, S.ROW_CAND)          # (N, 4, 16)
        o = R.obs.repeat_interleave(TG.N_OURS, 0)
        z = R.xi.repeat_interleave(TG.N_OURS, 0)
        return o, z, tg.reshape(-1, tg.shape[-1]), info
    c = get_cache(ctx)
    os_, zs, ts = [], [], []
    for i in range(len(R)):
        sup = c.support[i] if arm.support == "bsup" else R.xi[i].reshape(1, -1)
        os_.append(R.obs[i].reshape(1, -1).expand(sup.shape[0], -1))
        zs.append(sup)
        ts.append(c.target[i].reshape(1, -1).expand(sup.shape[0], -1))
    kinds = [d.get("kind", "") for d in c.info]
    info.update({"n_rows": len(R), "n_reroute_rows": int(sum(z.shape[0] for z in zs)),
                 "frac_single_cluster": float(np.mean(c.n_clusters == 1)),
                 "frac_hop": float(np.mean([k == "hop" for k in kinds])),
                 "frac_no_exit": float(np.mean([k == "no_exit" for k in kinds])),
                 "frac_single_far": float(np.mean([k == "single_far" for k in kinds])),
                 "frac_neg_typical": float(np.mean(c.neg_typical)),
                 "mean_bad_frac": float(np.mean(c.bad_frac)),
                 "mean_support": float(np.mean([z.shape[0] for z in zs]))})
    return torch.cat(os_), torch.cat(zs), torch.cat(ts), info


def step_fn(arm: Arm, ctx: Ctx):
    B, ref, R = ctx.batch, ctx.ref, ctx.rows
    info: Dict = {}
    if arm.kind == "reroute":
        o, z, tg, info = reroute_rows(arm, ctx)

        def method(pol, g):
            i = _pick(o.shape[0], B, g)
            t = FlowPolicy.sample_time(B, g)
            return pol.flow_loss(tg[i], z[i], t, o[i]).mean()
    elif arm.kind in ("npo", "kto"):
        def method(pol, g):
            i = _pick(len(R), B, g)
            a, o = R.a[i], R.obs[i]
            noise = torch.randn(a.shape, generator=g)
            t = FlowPolicy.sample_time(B, g)
            l_th = pol.flow_loss(a, noise, t, o)
            with torch.no_grad():
                l_ref = ref.flow_loss(a, noise, t, o)
            r = l_ref - l_th
            if arm.kind == "npo":
                return O.npo_term(r, arm.beta).mean()
            k = min(8, B)            # undesirable-only KTO; z0 from rolled rows (other states' chunks)
            with torch.no_grad():
                ar, nr = torch.roll(a, 1, 0)[:k], torch.roll(noise, 1, 0)[:k]
                z0 = torch.clamp((ref.flow_loss(ar, nr, t[:k], o[:k]) - pol.flow_loss(ar, nr, t[:k], o[:k])).mean(), min=0.0)
            return (1.0 - torch.sigmoid(arm.beta * (z0 - r))).mean()
    elif arm.kind == "sdpo":
        Z = None
        if arm.pos == "hop":
            P = get_cache(ctx).target.unsqueeze(1)                                      # (N, 1, 16)
        elif arm.noise == "own":
            Z, P = TG._row_decodes(ref, ctx.seed, R, S.ROW_SDPO, TG.N_SDPO_POS)         # winners + the noise that made them
        else:
            P = TG.fresh_decodes(ref, ctx.seed, R, TG.N_SDPO_POS, S.ROW_SDPO)          # (N, 8, 16)
        info["n_pos_per_row"] = int(P.shape[1])
        if arm.noise == "own":
            K = arm.n_win

            def method(pol, g):
                i = _pick(len(R), B, g)
                a_l, o, z_l = R.a[i], R.obs[i], R.xi[i]
                t = FlowPolicy.sample_time(B, g)
                if K >= P.shape[1]:
                    j = torch.arange(P.shape[1]).expand(B, -1)
                else:
                    j = torch.stack([torch.randperm(P.shape[1], generator=g)[:K] for _ in range(B)])
                a_w = P[i[:, None], j].reshape(B * K, -1)
                z_w = Z[i[:, None], j].reshape(B * K, -1)
                o_w, t_w = o.repeat_interleave(K, 0), t.repeat_interleave(K, 0)
                with torch.no_grad():
                    lr_w, lr_l = ref.flow_loss(a_w, z_w, t_w, o_w), ref.flow_loss(a_l, z_l, t, o)
                r_w = (lr_w - pol.flow_loss(a_w, z_w, t_w, o_w)).reshape(B, K).mean(1)
                r_l = lr_l - pol.flow_loss(a_l, z_l, t, o)
                return O.dpo_pair_term(r_w, r_l, arm.beta).mean()
            return method, info

        def method(pol, g):
            i = _pick(len(R), B, g)
            j = _pick(P.shape[1], B, g)
            a_l, a_w, o = R.a[i], P[i, j], R.obs[i]
            t = FlowPolicy.sample_time(B, g)
            if arm.noise == "pin":
                n_w = n_l = R.xi[i]
            else:
                n_w = torch.randn(a_w.shape, generator=g)
                n_l = torch.randn(a_l.shape, generator=g)
            with torch.no_grad():
                lr_w, lr_l = ref.flow_loss(a_w, n_w, t, o), ref.flow_loss(a_l, n_l, t, o)
            if arm.half == "w":                      # winner half only: -log sigmoid(beta r_w)
                r_w = lr_w - pol.flow_loss(a_w, n_w, t, o)
                return O.dpo_pair_term(r_w, torch.zeros_like(r_w), arm.beta).mean()
            if arm.half == "l":                      # loser half only: -log sigmoid(-beta r_l)
                r_l = lr_l - pol.flow_loss(a_l, n_l, t, o)
                return O.dpo_pair_term(torch.zeros_like(r_l), r_l, arm.beta).mean()
            r_w = lr_w - pol.flow_loss(a_w, n_w, t, o)
            r_l = lr_l - pol.flow_loss(a_l, n_l, t, o)
            return O.dpo_pair_term(r_w, r_l, arm.beta).mean()
    else:
        raise ValueError(arm.kind)

    if not arm.local:
        return method, info
    lo, lz, la, counts = get_cache(ctx).local
    info.update({"local_anchor_n": int(lo.shape[0]), "local_anchor_rows_empty": float(np.mean(counts == 0))})
    if lo.shape[0] == 0:
        return method, info

    def anchored(pol, g):
        lm = method(pol, g)
        j = _pick(lo.shape[0], B, g)
        t = FlowPolicy.sample_time(B, g)
        return 0.5 * (lm + pol.flow_loss(la[j], lz[j], t, lo[j]).mean())
    return anchored, info


def train_arm_shift(arm: Arm, ctx: Ctx, n_shift: int, log_every=500, refresh_rows=None):
    """One continuous SDPO run (one optimiser, one cosine schedule over ctx.steps) whose reference policy is
    replaced by the current policy n_shift - 1 times, at equal intervals. At every shift the winners are
    regenerated from the new reference (an SDPO winner is a sample of pi_ref); the logged negatives and
    their logged noises never change -- unless ``refresh_rows(frozen_policy, shift_index)`` is given, which
    returns the NEW negative rows (re-collected from the current policy) used from that shift on."""
    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref)
    pol.train()
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(pol.parameters(), lr=ctx.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    g = S.gen(ctx.seed, S.P_TRAIN)
    bounds = [int(round(ctx.steps * q / n_shift)) for q in range(n_shift + 1)]
    cur = copy.copy(ctx)
    fn, info = step_fn(arm, cur)
    hist = []
    for step in range(ctx.steps):
        if step in bounds[1:-1]:
            from .policy import frozen
            cur = copy.copy(cur)
            cur.ref = frozen(pol)
            if refresh_rows is not None:
                cur.rows = refresh_rows(cur.ref, bounds.index(step))
                info.setdefault("refresh_rows", []).append(len(cur.rows))
            fn, _ = step_fn(arm, cur) if len(cur.rows) else (lambda p_, g_: sum(x.sum() * 0 for x in p_.parameters()), None)
        lr = cosine(step, ctx.steps, min(500, ctx.steps // 10), ctx.lr, 0.1 * ctx.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == ctx.steps - 1:
            hist.append({"step": step, "loss": float(loss.detach())})
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        info["edit_norm"] = float(sum(float((p - r).norm()) ** 2 for p, r in zip(pol.parameters(), ctx.ref.parameters())) ** 0.5)
    info.update({"train_history": hist, "n_shift": n_shift, "shift_steps": bounds[1:-1]})
    return pol, info


def train_arm(arm: Arm, ctx: Ctx, probe=None, log_every=500):
    if arm.kind == "frozen":
        return ctx.ref, {}
    if arm.kind == "filter":
        f = RejectFilter(ctx.ref, ctx.seed, ctx.rows)
        return f, dict(f.stats)
    if arm.kind == "rand":
        pol_of, info = train_arm(ARMS[arm.of], ctx)
        g = S.gen(ctx.seed, P_RAND)
        pol = copy.deepcopy(ctx.ref)
        num = 0.0
        with torch.no_grad():
            for p, q, r in zip(pol.parameters(), pol_of.parameters(), ctx.ref.parameters()):
                d = torch.randn(p.shape, generator=g)
                nrm = (q - r).norm()
                num += float(nrm) ** 2
                p.copy_(r + d * (nrm / d.norm().clamp_min(1e-12)))
        return pol, {"edit_norm": num ** 0.5, "of": arm.of}
    fn, info = step_fn(arm, ctx)
    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref)
    pol.train()
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(pol.parameters(), lr=ctx.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    g = S.gen(ctx.seed, S.P_TRAIN)
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
        if step % log_every == 0 or step == ctx.steps - 1:
            rec = {"step": step, "loss": float(loss.detach())}
            if probe is not None:
                pol.eval()
                rec.update(probe(pol))
                pol.train()
            hist.append(rec)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        info["edit_norm"] = float(sum(float((p - r).norm()) ** 2 for p, r in zip(pol.parameters(), ctx.ref.parameters())) ** 0.5)
    info["train_history"] = hist
    return pol, info
