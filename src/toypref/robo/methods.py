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
]}


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
        if arm.pos == "hop":
            P = get_cache(ctx).target.unsqueeze(1)                                      # (N, 1, 16)
        else:
            P = TG.fresh_decodes(ref, ctx.seed, R, TG.N_SDPO_POS, S.ROW_SDPO)          # (N, 8, 16)
        info["n_pos_per_row"] = int(P.shape[1])

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


def train_arm(arm: Arm, ctx: Ctx, probe=None, log_every=500):
    if arm.kind == "frozen":
        return ctx.ref, {}
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
    info["train_history"] = hist
    return pol, info
