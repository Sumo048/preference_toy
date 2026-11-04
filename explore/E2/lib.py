"""E2 exploration helpers: cells, decodes with activation capture, direction edits, evaluation.

Edits here read only network inputs (x, t, c) and activations. The oracle (CondMixture.assign via
experiment.compute) is used only in `evaluate` and in analysis printouts marked ORACLE.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TOYPREF_RUNS", os.path.join(ROOT, "runs"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toypref.cond import experiment as E
from toypref.cond import targets as TG
from toypref.cond.task import CondFlow, frozen

K = 20
DT = 1.0 / K


def load_ref(seed):
    sd = torch.load(os.path.join(os.environ["TOYPREF_RUNS"], "cond_base", f"s{seed}", "policy.pt"),
                    map_location="cpu", weights_only=False)
    ref = CondFlow(**sd["kw"])
    ref.load_state_dict(sd["state"])
    return frozen(ref)


@dataclass
class Cell:
    seed: int
    layout: str
    npm: int
    ref: CondFlow
    ed: E.Edit
    anchor: E.Pool
    bank: E.Bank
    pool: E.Pool
    _vote: object = None

    @property
    def name(self):
        return f"{'k4' if self.layout == 'random' else self.layout}_s{self.seed}_n{self.npm}"


def load_cell(seed, layout, npm) -> Cell:
    """Cached on disk under runs/E2cache (the pool decode of 200k samples is the slow part)."""
    assert seed in range(5), "E2 uses dev seeds 0-4 only"
    cp = os.path.join(os.environ["TOYPREF_RUNS"], "E2cache", f"{layout}_s{seed}_n{npm}.pt")
    ref = load_ref(seed)
    if os.path.exists(cp):
        d = torch.load(cp, weights_only=False)
        return Cell(seed, layout, npm, ref, d["ed"], d["anchor"], d["bank"], None)
    cell = _load_cell(seed, layout, npm, ref)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    torch.save({"ed": cell.ed, "anchor": cell.anchor, "bank": cell.bank}, cp)
    return cell


def _load_cell(seed, layout, npm, ref) -> Cell:
    pool = E.make_pool(ref, seed, 200_000)
    ed = E.editset(pool, seed, 4, npm, layout)
    anchor = E.make_pool(ref, seed, 20_000, purpose=E.P_ANCHOR)
    bank = E.Bank(ref, seed, ed)
    return Cell(seed, layout, npm, ref, ed, anchor, bank, pool)


# ------------------------------------------------------------------ edited network
class Edit:
    """Base class. Hooks: 'b{j}.z' (fc1 post-SiLU, 128), 'b{j}.d' (fc2 output added to residual),
    'h{j}' (residual after block j; h0 = inp output), 'phi' (silu(out_norm(h))), 'vel' (output).
    apply(name, act, x, t, c) -> act."""
    hooks: tuple = ()

    def apply(self, name, act, x, t, c):
        return act


def tgate(t, lo, hi, soft=0.0):
    """1 on t in [lo, hi] (t = 1 noise .. 0 action), optional sigmoid edges of width soft."""
    if soft <= 0:
        return ((t >= lo) & (t <= hi)).float()
    return torch.sigmoid((t - lo) / soft) * torch.sigmoid((hi - t) / soft)


class Rank1(Edit):
    """W_layer <- W_layer + s * u v^T (optionally an extra bias term s*u*beta), output side.
    layer: 'fc2.{j}' (input z_j, output residual) or 'out' (input phi, output vel) or
    'fc1.{j}' (input [norm(h), cond], output pre-activation). gate: optional callable(t) -> (N,)."""

    def __init__(self, layer, u, v, s=1.0, beta=0.0, gate=None):
        self.layer, self.u, self.v, self.s, self.beta, self.gate = layer, u.float(), v.float(), s, beta, gate
        kind, _, j = layer.partition(".")
        self.hooks = ({"fc2": f"b{j}.d", "out": "vel", "fc1": f"b{j}.pre"}[kind],)
        self.src = {"fc2": f"b{j}.z", "out": "phi", "fc1": f"b{j}.in"}[kind]

    def apply(self, name, act, x, t, c, cache=None):
        a = cache[self.src]
        sc = a @ self.v + self.beta
        if self.gate is not None:
            sc = sc * self.gate(t)
        return act + self.s * sc[:, None] * self.u[None, :]


class Ablate(Edit):
    """Remove direction v (unit) from the activation `site` as seen downstream: z <- (I - vv^T) z.
    site 'b{j}.z' = side post of fc1 (fc2 input weights become W2 (I - vv^T)); 'b{j}.d' = side pre
    of fc2; 'phi' = post of out_norm (out weights W_out (I - vv^T)). gate: optional callable(t)."""

    def __init__(self, site, v, gate=None, mean=None):
        self.site = site
        self.v = (v / v.norm()).float()
        self.gate = gate
        self.mean = mean        # optional: project onto mean instead of 0 (mean-ablation)
        self.hooks = (site,)

    def apply(self, name, act, x, t, c, cache=None):
        p = act @ self.v
        if self.mean is not None:
            p = p - self.mean
        if self.gate is not None:
            p = p * self.gate(t)
        return act - p[:, None] * self.v[None, :]


class Fn(Edit):
    def __init__(self, hook, fn):
        self.hooks = (hook,)
        self.fn = fn

    def apply(self, name, act, x, t, c, cache=None):
        return self.fn(act, x, t, c, cache)


def forward(net, x, t, c, edits=(), capture=None):
    """CondVelocityNet forward with edit hooks and optional capture dict."""
    cache = {} if capture is None else capture

    def hk(name, act):
        for e in edits:
            if name in e.hooks:
                act = e.apply(name, act, x, t, c, cache=cache)
        cache[name] = act
        return act

    cond = torch.cat([net.time_mlp(net.time_embed(t)), net.c_mlp(net.c_feat(c))], -1)
    h = hk("h0", net.inp(x))
    for j, b in enumerate(net.blocks):
        inp = hk(f"b{j}.in", torch.cat([b.norm(h), cond], -1))
        pre = hk(f"b{j}.pre", b.fc1(inp))
        z = hk(f"b{j}.z", b.act(pre))
        d = hk(f"b{j}.d", b.fc2(z))
        h = hk(f"h{j + 1}", h + d)
    phi = hk("phi", F.silu(net.out_norm(h)))
    return hk("vel", net.out(phi))


class EditedNet(nn.Module):
    def __init__(self, net, edits):
        super().__init__()
        self.base = net
        self.edits = list(edits)

    def forward(self, x, t, c):
        return forward(self.base, x, t, c, self.edits)


class EditedFlow(CondFlow):
    """A CondFlow whose net is the reference net plus edits (works with experiment.compute)."""

    def __init__(self, ref: CondFlow, edits):
        nn.Module.__init__(self)
        self.kw = ref.kw
        self.net = EditedNet(ref.net, edits)


@torch.no_grad()
def decode(net, xi, c, edits=(), capture=(), batch=65536):
    """Returns final action and, for each captured name, a (K, N, d) tensor of activations at each
    step k = 0..19 (state x_k entering step k). Also 'x' (K+1, N, 2)."""
    outs = {n: [] for n in capture}
    xs = []
    finals = []
    for i in range(0, xi.shape[0], batch):
        x = xi[i:i + batch].clone().float()
        cc = c[i:i + batch].float()
        per = {n: [] for n in capture}
        xk = [x]
        for k in range(K):
            t = torch.full((x.shape[0],), 1.0 - k * DT)
            cap = {}
            v = forward(net, x, t, cc, edits, capture=cap)
            for n in capture:
                per[n].append(cap[n])
            x = x - DT * v
            xk.append(x)
        finals.append(x)
        xs.append(torch.stack(xk))
        for n in capture:
            outs[n].append(torch.stack(per[n]))
    res = {n: torch.cat(outs[n], 1) for n in capture}
    res["x"] = torch.cat(xs, 1)
    return torch.cat(finals), res


# ------------------------------------------------------------------ label-free data
def neg_basins(cell: Cell):
    ed = cell.ed
    return [TG.basins_at(cell.ref, float(ed.c[i]), ed.a[i], TG.row_gen(cell.seed, ed.uid[i], 1))[0]
            for i in range(len(ed.uid))]


def matched_samples(cell: Cell, basins, m=256, seed_off=0, jitter=0.0):
    """At each negative's condition (optionally jittered), decode m fresh noises with pi_ref and
    label each by membership of its action in the negative's bad basin B_i (label-free).
    Returns c (N,), xi (N, 2), a (N, 2), bad (N,) bool, owner (N,) negative index, typical (N,)."""
    g = torch.Generator().manual_seed(9_000_000 + 1000 * cell.seed + seed_off)
    cs, xs, own = [], [], []
    for i in range(len(basins)):
        ci = float(cell.ed.c[i])
        cc = torch.full((m,), ci)
        if jitter > 0:
            cc = (cc + jitter * (2 * torch.rand(m, generator=g) - 1)).clamp(0, 0.9999)
        cs.append(cc)
        xs.append(torch.randn(m, 2, generator=g))
        own.append(np.full(m, i))
    c, xi, own = torch.cat(cs), torch.cat(xs), np.concatenate(own)
    a = cell.ref.sample(xi, c)
    bad = np.zeros(len(c), bool)
    lab = np.zeros(len(c), int)
    for i, b in enumerate(basins):
        s = own == i
        l = b.assign(a[s])
        lab[s] = l
        bad[s] = l == b.bad
    return c, xi, a, bad, own, lab


# ------------------------------------------------------------------ evaluation
def evaluate(cell: Cell, edits) -> Dict:
    """ORACLE: D, Removed, Collateral with the repo's compute on the cell's eval bank."""
    pol = cell.ref if not edits else EditedFlow(cell.ref, edits)
    r = E.compute(cell.bank, pol)
    return {"D": r["tv_to_ideal"], "Removed": 1 - r["neg_mass_ratio"], "Collateral": r["retain_tv"],
            "off": r["mass_off"], "D_ref": r["tv_to_ideal_ref"]}


def fmt(r):
    return f"D {r['D']:.3f} Rem {r['Removed']:.3f} Col {r['Collateral']:.3f} off {r['off']:.3f}"
