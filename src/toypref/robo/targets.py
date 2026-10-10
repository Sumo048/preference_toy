"""Per-state rerouting targets for robo2d (16-D ports of cond/targets.py).

Everything a builder may touch for negative row i: its observation o_i, logged noise xi_i,
logged chunk a_i, and the frozen base policy queried at o_i. Never a label beyond "this row
belongs to a negative episode", never the env.

Behaviour at a state is read off the base's own samples there ("basins"): decode M = 128 prior
noises at o_i, drop atypical decodes (5-NN distance > q90), link the rest by single linkage at
eps = 3 x median 1-NN distance; the bad basin is the cluster of the typical decode nearest a_i.

  hop       march xi_i along 16 random unit noise directions (step 0.05 up to 3.0) until the
            decode leaves the bad basin; target = decode at the smallest exit + 0.2 margin.
            With ONE cluster (the row is not a separate behaviour at its state) the hop cannot
            exit: fall back to the nearest typical decode farther from a_i than the median
            pairwise decode distance ("single_far"). With several clusters but no exit, the
            nearest typical decode outside the bad basin ("no_exit").
  bsup      support = xi_i + up to 8 of 16 jitters (sigma 0.6) that still decode into the bad basin
  local     64 base (noise, decode) pairs at o_i outside the bad basin, kept pinned
  ours      4 fresh base decodes at o_i (targets of R_ours, trained from the logged noise)
  sdpo_pos  8 fresh base decodes at o_i (SDPO positives)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from . import seeds as S
from .env import ACT_DIM

M_DECODE = 128
TYP_K = 5
TYP_Q = 0.9
EPS_MULT = 3.0
BSUP_K, BSUP_SIGMA, BSUP_KEEP = 16, 0.6, 8
HOP_DIRS, HOP_STEP, HOP_MAX, HOP_MARGIN = 16, 0.05, 3.0, 0.2
M_LOCAL = 64
N_OURS = 4
N_SDPO_POS = 8
ROW_GROUP = 16           # rows decoded together


@dataclass
class Basins:
    pts: torch.Tensor        # typical decodes (P, 16)
    lab: np.ndarray          # cluster id per typical decode
    eps: float
    bad: int
    thr: float
    n_clusters: int
    neg_typical: bool
    bad_frac: float          # share of typical decodes in the bad basin
    median_pair: float
    dec: torch.Tensor        # all M decodes
    typ: torch.Tensor        # (M,) typical mask

    def assign(self, x: torch.Tensor) -> np.ndarray:
        d = torch.cdist(x, self.pts)
        j = d.argmin(1)
        out = self.lab[j.numpy()].copy()
        out[(d.min(1).values > self.eps).numpy()] = -1
        return out


def _components(adj: np.ndarray) -> np.ndarray:
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


def basins_from(dec: torch.Tensor, a_bad: torch.Tensor) -> Basins:
    d = torch.cdist(dec, dec)
    d.fill_diagonal_(float("inf"))
    kth = d.kthvalue(TYP_K, dim=1).values
    thr = float(torch.quantile(kth, TYP_Q))
    typ = kth <= thr
    pts = dec[typ]
    dp = torch.cdist(pts, pts)
    off = dp[~torch.eye(len(pts), dtype=torch.bool)]
    median_pair = float(off.median())
    dp.fill_diagonal_(float("inf"))
    eps = EPS_MULT * float(dp.min(1).values.median())
    lab = _components((dp <= eps).numpy())
    da = torch.cdist(a_bad.reshape(1, -1), pts)[0]
    bad = int(lab[int(da.argmin())])
    kneg = float(torch.cdist(a_bad.reshape(1, -1), dec)[0].kthvalue(TYP_K).values)
    return Basins(pts=pts, lab=lab, eps=eps, bad=bad, thr=thr, n_clusters=int(lab.max() + 1),
                  neg_typical=kneg <= thr, bad_frac=float((lab == bad).mean()), median_pair=median_pair,
                  dec=dec, typ=typ)


def _row_decodes(ref, seed, rows, stream, M) -> (torch.Tensor, torch.Tensor):
    """(N, M, 16) noises and decodes at each row's observation from the row's own stream."""
    N = len(rows)
    Z = torch.stack([torch.randn(M, ACT_DIM, generator=S.row_gen(seed, int(u), stream)) for u in rows.uid]) \
        if N else torch.zeros(0, M, ACT_DIM)
    dec = torch.zeros_like(Z)
    for s in range(0, N, ROW_GROUP):
        e = min(s + ROW_GROUP, N)
        o = rows.obs[s:e].repeat_interleave(M, 0)
        dec[s:e] = ref.sample(Z[s:e].reshape(-1, ACT_DIM), o).reshape(e - s, M, ACT_DIM)
    return Z, dec


@torch.no_grad()
def all_basins(ref, seed, rows) -> List[Basins]:
    _, dec = _row_decodes(ref, seed, rows, S.ROW_CAND, M_DECODE)
    return [basins_from(dec[i], rows.a[i]) for i in range(len(rows))]


@torch.no_grad()
def hop_targets(ref, seed, rows, basins: List[Basins]):
    """(targets (N, 16), info list). Directions come from the row's ROW_CAND stream AFTER the
    basin decodes (the same generator, continued), so they are shared by every arm."""
    N = len(rows)
    tgt = torch.zeros(N, ACT_DIM)
    info = [dict() for _ in range(N)]
    taus = torch.arange(HOP_STEP, HOP_MAX + 1e-9, HOP_STEP)
    multi = [i for i in range(N) if basins[i].n_clusters > 1]
    dirs = {}
    for i in multi:
        g = S.row_gen(seed, int(rows.uid[i]), S.ROW_CAND)
        torch.randn(M_DECODE, ACT_DIM, generator=g)          # skip the basin decodes
        d = torch.randn(HOP_DIRS, ACT_DIM, generator=g)
        dirs[i] = d / d.norm(dim=1, keepdim=True)
    G = max(1, ROW_GROUP // 4)
    for s in range(0, len(multi), G):
        grp = multi[s:s + G]
        Zs = [rows.xi[i].reshape(1, 1, -1) + taus.reshape(1, -1, 1) * dirs[i].reshape(HOP_DIRS, 1, -1) for i in grp]
        Z = torch.cat([z.reshape(-1, ACT_DIM) for z in Zs])
        o = torch.cat([rows.obs[i].reshape(1, -1).expand(HOP_DIRS * len(taus), -1) for i in grp])
        dec = ref.sample(Z, o).reshape(len(grp), HOP_DIRS, len(taus), ACT_DIM)
        for n, i in enumerate(grp):
            b = basins[i]
            cl = b.assign(dec[n].reshape(-1, ACT_DIM)).reshape(HOP_DIRS, len(taus))
            best = None
            for k in range(HOP_DIRS):
                out = np.flatnonzero((cl[k] >= 0) & (cl[k] != b.bad))
                if len(out):
                    tau = float(taus[out[0]]) + HOP_MARGIN
                    if best is None or tau < best[0]:
                        best = (tau, k)
            if best is not None:
                zt = rows.xi[i] + best[0] * dirs[i][best[1]]
                t = ref.sample(zt.reshape(1, -1), rows.obs[i].reshape(1, -1))[0]
                if b.assign(t.reshape(1, -1))[0] not in (-1, b.bad):
                    tgt[i] = t
                    info[i] = {"kind": "hop", "tau": best[0]}
                    continue
            ok = torch.from_numpy(b.lab != b.bad)
            cand = b.pts[ok]
            tgt[i] = cand[int(torch.linalg.norm(cand - rows.a[i], dim=1).argmin())]
            info[i] = {"kind": "no_exit"}
    for i in range(N):
        if basins[i].n_clusters > 1:
            continue
        b = basins[i]
        dist = torch.linalg.norm(b.pts - rows.a[i], dim=1)
        far = dist > b.median_pair
        cand_d = torch.where(far, dist, torch.full_like(dist, float("inf")))
        j = int(cand_d.argmin()) if far.any() else int(dist.argmax())
        tgt[i] = b.pts[j]
        info[i] = {"kind": "single_far"}
    return tgt, info


@torch.no_grad()
def bsup_support(ref, seed, rows, basins: List[Basins]) -> List[torch.Tensor]:
    N = len(rows)
    jit = torch.stack([rows.xi[i] + BSUP_SIGMA * torch.randn(BSUP_K, ACT_DIM, generator=S.row_gen(seed, int(rows.uid[i]), S.ROW_SUPPORT))
                       for i in range(N)]) if N else torch.zeros(0, BSUP_K, ACT_DIM)
    dec = ref.sample(jit.reshape(-1, ACT_DIM), rows.obs.repeat_interleave(BSUP_K, 0)).reshape(N, BSUP_K, ACT_DIM) \
        if N else jit
    out = []
    for i in range(N):
        keep = basins[i].assign(dec[i]) == basins[i].bad
        out.append(torch.cat([rows.xi[i].reshape(1, -1), jit[i][torch.from_numpy(keep)][:BSUP_KEEP]]))
    return out


@torch.no_grad()
def local_anchor(ref, seed, rows, basins: List[Basins]):
    """(obs, noise, action) pinned pairs at each row's state outside its bad basin."""
    Z, dec = _row_decodes(ref, seed, rows, S.ROW_LOCAL, M_LOCAL)
    os_, zs, acts, counts = [], [], [], []
    for i in range(len(rows)):
        lab = basins[i].assign(dec[i])
        keep = torch.from_numpy((lab != basins[i].bad) & (lab != -1))
        k = int(keep.sum())
        counts.append(k)
        os_.append(rows.obs[i].reshape(1, -1).expand(k, -1))
        zs.append(Z[i][keep])
        acts.append(dec[i][keep])
    return torch.cat(os_), torch.cat(zs), torch.cat(acts), np.array(counts)


@torch.no_grad()
def fresh_decodes(ref, seed, rows, n, stream) -> torch.Tensor:
    """(N, n, 16) fresh base decodes at each row's observation (R_ours: n=4 from ROW_CAND;
    SDPO positives: n=8 from ROW_SDPO)."""
    return _row_decodes(ref, seed, rows, stream, n)[1]


@dataclass
class HopCache:
    target: torch.Tensor                 # (N, 16)
    support: List[torch.Tensor]          # per row (k_i, 16)
    info: List[Dict]
    n_clusters: np.ndarray
    neg_typical: np.ndarray
    bad_frac: np.ndarray
    local: tuple                         # (obs, noise, action, counts)
    uid: np.ndarray = field(default=None)


def build_hop_cache(ref, seed, rows) -> HopCache:
    bas = all_basins(ref, seed, rows)
    tgt, info = hop_targets(ref, seed, rows, bas)
    sup = bsup_support(ref, seed, rows, bas)
    loc = local_anchor(ref, seed, rows, bas)
    return HopCache(tgt, sup, info, np.array([b.n_clusters for b in bas]), np.array([b.neg_typical for b in bas]),
                    np.array([b.bad_frac for b in bas]), loc, uid=np.asarray(rows.uid).copy())
