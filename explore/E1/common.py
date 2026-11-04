"""E1 shared code: cells, decodes with activations, a label-free mode tracker, and grafted policies.

Only `oracle_eval` and the `*_diag` helpers touch toypref's oracle (CondMixture.assign via
experiment.compute / MIX). Everything an edit is fitted on comes from pi_ref samples and the
logged negatives.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TOYPREF_RUNS", os.path.join(ROOT, "runs"))

from toypref.cond import experiment as E          # noqa: E402
from toypref.cond import targets as TG            # noqa: E402
from toypref.cond.task import CondFlow, frozen    # noqa: E402

K_STEPS = 20
DT = 1.0 / K_STEPS
P_E1 = 910_000          # E1's own random streams: gen(seed, P_E1 + j); disjoint from toypref's purposes


def egen(seed, j):
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + P_E1 + j)


# ------------------------------------------------------------------ cell
@dataclass
class Cell:
    seed: int
    layout: str
    npm: int
    ref: CondFlow
    ed: E.Edit
    bank: E.Bank          # final-evaluation bank (oracle side)
    name: str = ""


_POOL_CACHE: Dict = {}


def load_ref(seed):
    sd = torch.load(os.path.join(os.environ["TOYPREF_RUNS"], "cond_base", f"s{seed}", "policy.pt"),
                    map_location="cpu", weights_only=False)
    ref = CondFlow(**sd["kw"])
    ref.load_state_dict(sd["state"])
    return frozen(ref)


def load_cell(seed, layout, npm) -> Cell:
    assert seed in range(5), "E1 uses seeds 0-4 only"
    ref = load_ref(seed)
    if seed not in _POOL_CACHE:
        _POOL_CACHE[seed] = E.make_pool(ref, seed, 200_000)
    ed = E.editset(_POOL_CACHE[seed], seed, 4, npm, layout)
    bank = E.Bank(ref, seed, ed)
    nm = f"{'k4' if layout == 'random' else layout}_s{seed}_n{npm}"
    return Cell(seed=seed, layout=layout, npm=npm, ref=ref, ed=ed, bank=bank, name=nm)


# ------------------------------------------------------------------ decode with activations
LAYERS = ["inp", "b0", "b1", "b2", "b0.z", "b1.z", "b2.z", "pre_out"]


def net_forward(net, x, t, c, capture=False, hooks=None):
    """CondVelocityNet forward, re-implemented to expose activations and to insert grafts.

    Activations: inp (128) = inp(x); b{i} (128) = residual stream after block i;
    b{i}.z (128) = SiLU(fc1) hidden of block i; pre_out (128) = SiLU(out_norm(h)).
    hooks: dict layer -> callable(acts_dict, x, t, c) -> additive delta for that layer's output
    (for 'out': a delta on the velocity)."""
    acts = {}
    cond = torch.cat([net.time_mlp(net.time_embed(t)), net.c_mlp(net.c_feat(c))], -1)
    acts["cond"] = cond
    h = net.inp(x)
    acts["inp"] = h
    for i, b in enumerate(net.blocks):
        z = b.act(b.fc1(torch.cat([b.norm(h), cond], -1)))
        acts[f"b{i}.z"] = z
        h = h + b.fc2(z)
        if hooks and f"b{i}" in hooks:
            h = h + hooks[f"b{i}"](acts, x, t, c)
        acts[f"b{i}"] = h
    p = F.silu(net.out_norm(h))
    acts["pre_out"] = p
    v = net.out(p)
    if hooks and "out" in hooks:
        v = v + hooks["out"](acts, x, t, c)
    return (v, acts) if capture else v


@torch.no_grad()
def decode(net, xi, c, layers=(), steps_keep=None, hooks=None, k_start=0, x_start=None):
    """Euler decode. Returns final x and, if layers, dict layer -> (K, N, d) activations at each step
    plus 'x' -> (K+1, N, 2) states."""
    x = (xi if x_start is None else x_start).clone().float()
    c = c.float()
    xs = [x.clone()]
    rec = {l: [] for l in layers}
    for k in range(k_start, K_STEPS):
        t = torch.full((x.shape[0],), 1.0 - k * DT)
        v, acts = net_forward(net, x, t, c, capture=True, hooks=hooks)
        for l in layers:
            rec[l].append(acts[l] if l != "v" else v)
        x = x - DT * v
        xs.append(x.clone())
    if layers:
        out = {l: torch.stack(rec[l]) for l in layers}
        out["x"] = torch.stack(xs)
        return x, out
    return x


@torch.no_grad()
def decode_batched(net, xi, c, hooks=None, bs=32768):
    return torch.cat([decode(net, xi[i:i + bs], c[i:i + bs], hooks=hooks) for i in range(0, xi.shape[0], bs)])


# ------------------------------------------------------------------ label-free mode tracker
@dataclass
class Tracker:
    """Basins of pi_ref followed continuously across c, from pi_ref samples only.

    A fixed noise set Z is decoded on a grid of conditions. At each grid c, typical decodes are
    clustered by single linkage (toypref's basin rule). Clusters at neighbouring grid points are
    linked when they share noises; linked clusters form one track (a behaviour). The tracks that
    hold logged negatives are the bad tracks. No labels and no ring geometry are used."""
    cg: np.ndarray                 # (G,) grid
    dec: torch.Tensor              # (G, Nz, 2)
    trk: np.ndarray                # (G, Nz) track id or -1 (atypical)
    eps: np.ndarray                # (G,)
    n_tracks: int
    bad: List[int] = field(default_factory=list)

    def label(self, a, c, loose=1.0):
        """(N,) track id of action a at condition c (nearest typical decode at the nearest grid c,
        -1 if farther than loose x eps)."""
        a = torch.as_tensor(a).float()
        c = np.asarray(torch.as_tensor(c).float())
        g = np.clip(np.round((c - self.cg[0]) / (self.cg[1] - self.cg[0])).astype(int), 0, len(self.cg) - 1)
        out = -np.ones(len(c), dtype=int)
        for gi in np.unique(g):
            sel = np.flatnonzero(g == gi)
            ok = self.trk[gi] >= 0
            pts = self.dec[gi][torch.from_numpy(ok)]
            lab = self.trk[gi][ok]
            d = torch.cdist(a[sel], pts)
            j = d.argmin(1).numpy()
            o = lab[j].copy()
            o[(d.min(1).values.numpy() > loose * self.eps[gi])] = -1
            out[sel] = o
        return out

    def is_bad(self, a, c, loose=1.0):
        return np.isin(self.label(a, c, loose), self.bad)


def _components(adj):
    n = adj.shape[0]
    lab = -np.ones(n, dtype=int)
    cid = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = cid
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] & (lab < 0)):
                lab[v] = cid
                stack.append(v)
        cid += 1
    return lab


@torch.no_grad()
def build_tracker(ref, seed, ed, G=201, Nz=768, min_frac=0.02) -> Tracker:
    Z = torch.randn(Nz, 2, generator=egen(seed, 1))
    cg = np.linspace(0.0, 0.999, G).astype(np.float32)
    cc = torch.from_numpy(np.repeat(cg, Nz))
    dec = decode_batched(ref.net, Z.repeat(G, 1), cc).reshape(G, Nz, 2)
    loc = -np.ones((G, Nz), dtype=int)
    eps = np.zeros(G)
    for g in range(G):
        d = torch.cdist(dec[g], dec[g])
        d.fill_diagonal_(float("inf"))
        kth = d.kthvalue(TG.TYP_K, 1).values
        thr = float(torch.quantile(kth, TG.TYP_Q))
        typ = (kth <= thr).numpy()
        dp = d[typ][:, typ]
        e = TG.EPS_MULT * float(dp.min(1).values.median())
        eps[g] = e
        lab = _components((dp <= e).numpy())
        # drop tiny clusters (stragglers)
        cnt = np.bincount(lab)
        lab[cnt[lab] < min_frac * Nz] = -1
        loc[g, typ] = lab
    # link clusters across neighbouring grid points by shared noises (union-find)
    offs = np.zeros(G + 1, dtype=int)
    for g in range(G):
        offs[g + 1] = offs[g] + (loc[g].max() + 1 if (loc[g] >= 0).any() else 0)
    parent = np.arange(offs[-1])

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for g in range(G - 1):
        a, b = loc[g], loc[g + 1]
        for ca in range(a.max() + 1):
            ma = a == ca
            if not ma.any():
                continue
            bb = b[ma]
            bb = bb[bb >= 0]
            if not len(bb):
                continue
            cb = np.bincount(bb).argmax()
            ov = (bb == cb).sum() / max(min(ma.sum(), (b == cb).sum()), 1)
            if ov > 0.5:
                parent[find(offs[g] + ca)] = find(offs[g + 1] + cb)
    roots = {}
    trk = -np.ones((G, Nz), dtype=int)
    for g in range(G):
        for ca in range(loc[g].max() + 1):
            r = find(offs[g] + ca)
            roots.setdefault(r, len(roots))
            trk[g, loc[g] == ca] = roots[r]
    T = Tracker(cg=cg, dec=dec, trk=trk, eps=eps, n_tracks=len(roots))
    lab = T.label(ed.a, ed.c)
    T.bad = sorted(set(int(x) for x in lab if x >= 0))
    return T


def tracker_diag(T: Tracker, cell: Cell, n=20000):
    """Oracle diagnostic: agreement of tracker's bad flag with oracle's forgotten-mode label."""
    b = cell.bank
    a = b.a_ref[:n]
    bad_t = T.is_bad(a, b.c[:n])
    bad_o = np.isin(b.lab_ref[:n], cell.ed.forget)
    return {"n_tracks": T.n_tracks, "n_bad_tracks": len(T.bad),
            "precision": float(bad_o[bad_t].mean()) if bad_t.any() else float("nan"),
            "recall": float(bad_t[bad_o].mean()), "flag_frac": float(bad_t.mean())}


# ------------------------------------------------------------------ policies with grafts
class Policy:
    """Anything with .sample(xi, c) for experiment.compute."""

    def __init__(self, net, hooks=None):
        self.net, self.hooks = net, hooks or {}

    @torch.no_grad()
    def sample(self, xi, c, batch=32768):
        return decode_batched(self.net, xi, c, hooks=self.hooks, bs=batch)


def oracle_eval(cell: Cell, pol) -> Dict:
    m = E.compute(cell.bank, pol)
    return {"D": m["tv_to_ideal"], "Removed": 1.0 - m["neg_mass_ratio"], "Collateral": m["retain_tv"],
            "off": m["mass_off"], "D_ref": m["tv_to_ideal_ref"]}
