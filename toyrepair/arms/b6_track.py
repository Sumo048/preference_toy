"""B6: track-masked density BC (E3's best label-free method), a code baseline. Global scope; one training run.

Ported from ``explore/E3`` on branch ``art/explore-E3`` (method spec ``a=track60k-w=1-st=20000``,
``e3.py`` + ``tracks.py``); see docs/explore/E3.md and docs/DEV_LOG.md (W2b, B6):

1. Tracks: decode 512 fixed noises with pi_ref at 101 conditions; cluster the typical decodes per condition
   (toypref basin rule); link clusters across neighbouring conditions by mutual nearest centroids; connected
   components are behaviour tracks.
2. Each logged negative (c_i, a_i) marks the track of a_i (3x radius) as bad.
3. Data: a fresh pool of 60,000 pi_ref samples (toypref ``make_pool``, purpose 90,001); keep the samples whose
   track is assigned and not bad.
4. Train all weights with flow matching on the kept samples (fresh noise), 20,000 steps, AdamW lr 1e-4 cosine
   (warm-up 500, floor 1e-5), betas (0.9, 0.95), grad clip 1, toypref's training stream ``gen(seed, P_TRAIN)``.

Inputs: pi_ref decodes and the logged negatives (c_i, a_i) only. No oracle labels, no forget set, no ring
geometry, no mode count. The tracker here is E3's, kept separate from ``toyrepair.tracks`` (different noise
stream) so the port reproduces E3; neither is given to the surgeon arms as a data option (PREREG amendment 1).
"""
from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import torch

from .base import ArmOutput, RunIO, Services, real_services, version_from_net

SPEC = "a=track60k-w=1-st=20000"
STEPS, LR, N_POOL, POOL_PURPOSE, BATCH_DEFAULT = 20_000, 1e-4, 60_000, 90_001, None
TYP_K, TYP_Q, EPS_MULT = 5, 0.9, 3.0


# ------------------------------------------------------------------ E3 tracks.py (verbatim logic)
def _cluster_at(ref, c, z):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    dec = ref.sample(z, torch.full((z.shape[0],), float(c)))
    d = torch.cdist(dec, dec)
    d.fill_diagonal_(float("inf"))
    kth = d.kthvalue(TYP_K, dim=1).values
    thr = float(torch.quantile(kth, TYP_Q))
    typ = (kth <= thr).numpy()
    pts = dec[typ]
    dp = torch.cdist(pts, pts)
    dp.fill_diagonal_(float("inf"))
    eps = EPS_MULT * float(dp.min(1).values.median())
    n, lab = connected_components(csr_matrix((dp <= eps).numpy()), directed=False)
    sizes = np.bincount(lab, minlength=n)
    keep = sizes >= max(3, int(0.02 * len(pts)))
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(keep.sum())
    lab = remap[lab]
    ok = lab >= 0
    pts, lab = pts[ok], lab[ok]
    cents = torch.stack([pts[lab == j].mean(0) for j in range(keep.sum())])
    return pts, lab, cents, eps


class E3Tracks:
    def __init__(self, ref, seed, n_grid=101, n_dec=512):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        g = torch.Generator().manual_seed(int(seed) * 7_777 + 4242)
        z = torch.randn(n_dec, 2, generator=g)
        self.grid = np.linspace(0.0, 1.0, n_grid)
        with torch.no_grad():
            self.levels = [_cluster_at(ref, c, z) for c in self.grid]
        offs = np.cumsum([0] + [len(L[2]) for L in self.levels])
        self.offs = offs
        rows, cols = [], []
        for gi in range(n_grid - 1):
            ca, cb = self.levels[gi][2], self.levels[gi + 1][2]
            d = torch.cdist(ca, cb)
            ab, ba = d.argmin(1).numpy(), d.argmin(0).numpy()
            for i, j in enumerate(ab):
                if ba[j] == i:
                    rows.append(offs[gi] + i)
                    cols.append(offs[gi + 1] + j)
        N = offs[-1]
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        self.n_tracks, self.track = connected_components(A, directed=False)

    def assign(self, a, c, eps_mult=1.0):
        gi = np.clip(np.rint(c.numpy() * (len(self.grid) - 1)).astype(int), 0, len(self.grid) - 1)
        out = -np.ones(a.shape[0], dtype=int)
        for k in np.unique(gi):
            sel = np.flatnonzero(gi == k)
            pts, lab, _, eps = self.levels[k]
            d = torch.cdist(a[sel], pts)
            j = d.argmin(1).numpy()
            hit = (d.min(1).values <= eps_mult * eps).numpy()
            out[sel[hit]] = self.track[self.offs[k] + lab[j[hit]]]
        return out


def bad_tracks(tr, c_neg, a_neg):
    t = tr.assign(a_neg, c_neg, eps_mult=3.0)
    return sorted(set(int(x) for x in t if x >= 0)), float((t < 0).mean())


# ------------------------------------------------------------------ training
def train_b6(cell, steps: int = STEPS):
    """-> (trained CondFlow, info). Same op order and streams as E3's run_method for SPEC."""
    from toypref.cond import experiment as E
    from toypref.flow import FlowPolicy
    from toypref.train import cosine
    ctx = cell.toypref_ctx(steps=5000)       # only ref, seed, batch are used (not the forget set)
    t0 = time.time()
    tr = E3Tracks(ctx.ref, ctx.seed)
    bad, miss = bad_tracks(tr, cell.neg.c, cell.neg.a)
    P = E.make_pool(ctx.ref, ctx.seed, N_POOL, purpose=POOL_PURPOSE)
    t = tr.assign(P.a, P.c)
    keep = torch.from_numpy((t >= 0) & ~np.isin(t, bad))
    c, a = P.c[keep], P.a[keep]
    B = ctx.batch

    def fn(pol, g):
        j = E._pick(c.shape[0], B, g)
        t_ = FlowPolicy.sample_time(B, g)
        return pol.flow_loss(a[j], torch.randn(a[j].shape, generator=g), t_, c[j]).mean()

    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref)
    pol.train()
    for q in pol.parameters():
        q.requires_grad_(False)
    params = list(pol.net.parameters())
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=1e-10)
    g = E.gen(ctx.seed, E.P_TRAIN)
    for step in range(steps):
        lr = cosine(step, steps, min(500, steps // 10), LR, 0.1 * LR)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    pol.eval()
    for q in pol.parameters():
        q.requires_grad_(False)
    info = {"spec": SPEC, "steps": steps, "track_n": int(tr.n_tracks), "track_bad": len(bad),
            "track_neg_miss": miss, "track_keep": int(keep.sum()), "track_unassigned": float((t < 0).mean()),
            "train_s": round(time.time() - t0, 1)}
    return pol, info


def run(cell, run, svc: Optional[Services] = None, steps: int = STEPS) -> ArmOutput:
    """B6 on ``cell``: one training run, kept whatever the gate says (like B1a); global scope."""
    svc = svc or real_services()
    io = RunIO(run, svc)
    base = svc.base_version(cell)
    io.log("run", {"arm": "B6", "cell": cell.key, "spec": SPEC})
    pol, info = train_b6(cell, steps=steps)
    ver = version_from_net(io, base, pol.net, "targeted_train", {"b6_spec": SPEC, "source": "toyrepair.arms.b6_track"},
                           meta=info)
    out = ArmOutput(arm="B6", version=ver.id, accepted=[ver.id], meta=info, final=ver)
    io.log("note", {"stop": out.stop, **out.to_json()}, version=ver.id)
    return out
