"""R0: iterative sampling-time controller: per class-R cluster, a logistic detector on flow states at k_commit (decodes ending in the cluster's basins vs the rest, no labels) runs the best primitive in code when it fires; gate applies.

Details (docs/DEV_LOG.md, W1-C):

* The detector reads the activation ``LAYER`` of the current version at the step where the best
  primitive acts (``primitive.k``: k_commit or one of the two sampled steps before it; 0 for a
  hop). It is linear: fires where ``w . f + b > threshold``.
* Training data (label-free, mining pool): ``N_PER_MEMBER`` decodes at each member's c_i, label 1
  if the action lands in B_i; ``N_REST`` decodes at uniform c, label 1 only if c is within
  ``C_NEAR`` of a member j and the action lands in B_j, else 0. Logistic regression on
  standardised features, folded back into (w, b).
* The controller is code that runs at sampling time: ``ControlledNet`` wraps the current
  network. At step k it evaluates the detector on each row and, where it fires, applies the
  primitive (push: displacement alpha*u/h after each Euler update of steps k..k+h-1; hop:
  xi <- xi + tau*u). Both are written as an exact velocity change, so the plain Euler decoder
  (``policy.decode``) runs it unchanged. Where the detector is off the output is the wrapped
  network's output, bit for bit.
* Candidates per cluster: probability thresholds 0.5, 0.7, 0.3, in that order.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ArmOutput, RepairContext, Services, iterate

LAYER = "blocks.2"
N_PER_MEMBER = 64
N_REST = 2048
C_NEAR = 0.02
THRESHOLDS = (0.5, 0.7, 0.3)
DT = 1.0 / 20


class ControlledNet(nn.Module):
    """Sampling-time controller around ``inner`` (forward(x, t, c) -> v).

    kind "push": u (2,), alpha, k, h; kind "hop": u, alpha = tau, k = 0.
    Detector: fires where acts[layer] . w + b > thr, evaluated at step k (k from t)."""

    def __init__(self, inner: nn.Module, layer: str, w: np.ndarray, b: float, thr: float, kind: str,
                 u, alpha: float, k: int, h: int, velocity_fn, make_version):
        super().__init__()
        self.inner = inner
        self.layer, self.kind, self.k, self.h = layer, kind, int(k), int(h)
        self.register_buffer("w", torch.as_tensor(np.asarray(w, dtype=np.float32)))
        self.register_buffer("u", torch.as_tensor(np.asarray(u, dtype=np.float32)))
        self.b, self.thr, self.alpha = float(b), float(thr), float(alpha)
        self._velocity, self._mk = velocity_fn, make_version
        self._mask: Optional[torch.Tensor] = None
        self.n_fired = 0

    def _feat(self, x, t, c):
        ver = self._mk(id="_inner", parent=None, patches=(), net=self.inner)
        v, acts = self._velocity(ver, x, t, c, capture=(self.layer,))
        return v, acts[self.layer]

    def fires(self, f: torch.Tensor) -> torch.Tensor:
        return (f @ self.w + self.b) > self.thr

    def forward(self, x, t, c):
        ks = torch.round((1.0 - t.reshape(-1)) * 20).long()
        if self.kind == "hop":
            sel = ks == 0
            if not bool(sel.any()):
                return self.inner(x, t, c)
            _, f = self._feat(x, t, c)
            m = self.fires(f) & sel
            if not bool(m.any()):
                return self.inner(x, t, c)
            self.n_fired += int(m.sum())
            d = self.alpha * self.u.reshape(1, 2) * m.reshape(-1, 1).to(x.dtype)
            return self.inner(x + d, t, c) - d / DT
        win = (ks >= self.k) & (ks < self.k + self.h)
        if not bool(win.any()):
            return self.inner(x, t, c)
        v, f = self._feat(x, t, c)
        at_k = ks == self.k
        if bool(at_k.all()):
            m = self.fires(f)
            self._mask = m.clone()
            self.n_fired += int(m.sum())
        elif self._mask is not None and self._mask.shape[0] == x.shape[0] and not bool(at_k.any()):
            m = self._mask                       # later steps of the same decode batch
        else:
            m = self.fires(f)                    # mixed-step call (not a decode): decide on the spot
        m = m & win
        if not bool(m.any()):
            return v
        return v - (self.alpha / self.h) * self.u.reshape(1, 2) * m.reshape(-1, 1).to(v.dtype) / DT


def _rng(cell, salt: int) -> np.random.Generator:
    """A generator derived from the cell's mining stream (pool 1, check index 0), salted by the
    cluster id so each cluster's detector data differ."""
    from toyrepair import pools
    g = pools.rng(pools.MINING, cell.index, 0)
    return np.random.default_rng([int(g.integers(2**62)), int(salt)])


def detector_data(ctx: RepairContext, k: int):
    """(features (P, d) float32, labels (P,) int) at step k under the current version."""
    cell, svc = ctx.cell, ctx.svc
    rows = np.asarray(ctx.cluster.members, dtype=np.int64)
    g = _rng(cell, int(ctx.card.cluster))
    cs = cell.neg.c[rows].numpy().astype(np.float32)
    # member conditions
    c_m = np.repeat(cs, N_PER_MEMBER).astype(np.float32)
    owner = np.repeat(np.arange(len(rows)), N_PER_MEMBER)
    # rest: uniform conditions
    c_r = g.random(N_REST).astype(np.float32)
    c_all = np.concatenate([c_m, c_r])
    xi = torch.from_numpy(g.standard_normal((len(c_all), 2)).astype(np.float32))
    c_t = torch.from_numpy(c_all)
    with torch.no_grad():
        dec = svc.decode(ctx.current, xi, c_t, capture=(LAYER,))
    f = dec.acts[LAYER][k].detach().numpy()
    a = dec.action.detach()
    near = np.abs(c_r[:, None] - cs[None, :])
    j = near.argmin(1)
    owner_r = np.where(near.min(1) < C_NEAR, j, -1)
    owner_all = np.concatenate([owner, owner_r])
    y = np.zeros(len(c_all), dtype=np.int64)
    for jj, i in enumerate(rows):
        sel = np.flatnonzero(owner_all == jj)
        if len(sel):
            y[sel] = cell.in_bad_basin(int(i), a[torch.from_numpy(sel)]).astype(np.int64)
    return f.astype(np.float32), y, float(dec.n_decodes or len(c_all))


def fit_detector(f: np.ndarray, y: np.ndarray):
    """Logistic regression on standardised features, folded into raw-feature (w, b)."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = f.mean(0), f.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((f - mu) / sd, y)
    w = clf.coef_[0] / sd
    b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    return w.astype(np.float32), b, float(clf.score((f - mu) / sd, y))


def repair(ctx: RepairContext):
    svc, io, prim = ctx.svc, ctx.io, ctx.table.best_primitive
    k = 0 if prim.kind == "hop" else int(prim.k)
    f, y, n_dec = detector_data(ctx, k)
    io.log("note", {"r0_cluster": int(ctx.card.cluster), "primitive": prim.name, "k": k,
                    "n_pos": int(y.sum()), "n": int(len(y)), "decodes": n_dec})
    if y.sum() < 2 or y.sum() > len(y) - 2:
        return
    w, b, acc = fit_detector(f, y)
    for p in THRESHOLDS:
        thr = math.log(p / (1 - p))
        vid = io.new_version_id()
        net = ControlledNet(ctx.current.net, LAYER, w, b, thr, prim.kind, prim.u, prim.alpha, k,
                            max(int(prim.h), 1), svc.velocity, svc.make_version)
        net.eval()
        d = os.path.join(io.dir, "r0")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"{vid}.npz"), w=w, b=b, thr=thr, u=np.asarray(prim.u, np.float32),
                 alpha=prim.alpha, k=k, h=max(int(prim.h), 1))
        meta = {"call": "r0_controller", "variant": f"p{p}", "layer": LAYER, "primitive": prim.name,
                "kind": prim.kind, "k": k, "train_acc": acc, "parent_version": ctx.current.id,
                "cluster": int(ctx.card.cluster)}
        ver = svc.make_version(id=vid, parent=ctx.current.id, patches=tuple(ctx.current.patches), net=net,
                               grid=getattr(ctx.current, "grid", "cond"), meta=meta)
        io.log("edit", meta, version=vid)
        yield ver


def run(cell, run, svc: Optional[Services] = None, max_rounds: Optional[int] = None) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run).
    ``max_rounds=1`` is the pilot (one repair, no recursion)."""
    return iterate("R0", cell, run, svc, repair, needs_branch=True, max_rounds=max_rounds)
