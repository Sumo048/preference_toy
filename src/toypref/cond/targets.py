"""Rerouting targets that need ONE logged negative per condition and no labels.

Everything a builder may touch for negative i: its condition c_i, its logged noise xi_i, its
action a_i = pi_ref(c_i, xi_i), and pi_ref itself (queried at c_i, or at other logged
conditions for the transplant). Never another negative at the same condition (there is none)
and never a mode label. The oracle labeller is used ONLY for diagnostics.

Behaviour at a condition is read off pi_ref's own samples there ("basins"):
  decode M prior noises at c_i, drop atypical decodes (kNN density, label-free), and link the
  rest by single linkage at a self-calibrated radius eps = 3 x median 1-NN spacing. The bad
  behaviour is the cluster holding a_i.

Targets
  ours     4 fresh pi_ref(c_i) decodes (repo's Ours; may land in the bad behaviour)
  own      nearest typical decode farther than d0 from a_i (the R_near_nf rule with a single
           negative; d0 = 4 mode-std is ORACLE geometry, a reference, not a practical method)
  basin    nearest typical decode outside a_i's cluster
  hop      march xi_i along random noise directions until the decode leaves a_i's cluster,
           step a margin further; the nearest such noise (smallest noise-space edit) gives the target
  pu       nearest typical decode the cross-condition bad-behaviour classifier calls good
Supports
  nhood    xi_i + 4 jitters at 0.3 (as the unconditional winner)
  bsup     up to 8 of 16 jitters at 0.6 that still decode into a_i's cluster: the basin itself
Transplant (`tx`): the row is also replayed at T other logged conditions c', treating
  pi_ref(c', xi_i) as the bad action there and building its basin target at c'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

M_DECODE = 128
TYP_K = 5
TYP_Q = 0.9
EPS_MULT = 3.0
NHOOD_K, NHOOD_SIGMA = 4, 0.3
BSUP_K, BSUP_SIGMA, BSUP_KEEP = 16, 0.6, 8
HOP_DIRS, HOP_STEP, HOP_MAX, HOP_MARGIN = 16, 0.05, 3.0, 0.2
N_TX = 8


def row_gen(seed, uid, stream):
    return torch.Generator().manual_seed((int(seed) * 7919 + 13) * 10_000_019 + int(uid) * 32 + stream)


@dataclass
class Basins:
    pts: torch.Tensor        # typical decodes (P, 2)
    lab: np.ndarray          # cluster id per typical decode
    eps: float
    bad: int                 # cluster of the negative
    thr: float               # typicality threshold

    def assign(self, x: torch.Tensor) -> np.ndarray:
        """cluster of the nearest typical decode if within eps, else -1."""
        d = torch.cdist(x, self.pts)
        j = d.argmin(1)
        out = self.lab[j.numpy()].copy()
        out[(d.min(1).values > self.eps).numpy()] = -1
        return out


def basins_at(ref, c: float, a_bad: torch.Tensor, g: torch.Generator) -> (Basins, torch.Tensor):
    z = torch.randn(M_DECODE, 2, generator=g)
    dec = ref.sample(z, torch.full((M_DECODE,), c))
    d = torch.cdist(dec, dec)
    d.fill_diagonal_(float("inf"))
    kth = d.kthvalue(TYP_K, dim=1).values
    thr = float(torch.quantile(kth, TYP_Q))
    typ = kth <= thr
    pts = dec[typ]
    dp = torch.cdist(pts, pts)
    dp.fill_diagonal_(float("inf"))
    eps = EPS_MULT * float(dp.min(1).values.median())
    adj = (dp <= eps).numpy()
    lab = -np.ones(len(pts), dtype=int)
    cid = 0
    for s in range(len(pts)):           # connected components (single linkage)
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
    bad = int(lab[int(torch.cdist(a_bad.reshape(1, 2), pts).argmin())])
    return Basins(pts=pts, lab=lab, eps=eps, bad=bad, thr=thr), z


def _typical(b: Basins, x):
    d = torch.cdist(x, b.pts)
    return d.kthvalue(min(TYP_K, b.pts.shape[0]), dim=1).values <= b.thr


def target_one(ref, rule, c, xi, a, g, d0=None, pu=None, stats=None):
    """One target (2,) for the negative (c, xi, a) under `rule`."""
    if rule == "ours":
        raise ValueError("ours is built separately")
    b, _ = basins_at(ref, c, a, g)
    pts, lab = b.pts, b.lab
    if rule == "basin":
        ok = lab != b.bad
    elif rule == "own":
        ok = (torch.linalg.norm(pts - a, dim=1) > d0).numpy()
    elif rule == "pu":
        with torch.no_grad():
            ok = (pu(torch.full((pts.shape[0],), c), pts) < 0).numpy()
    elif rule == "hop":
        dirs = torch.randn(HOP_DIRS, 2, generator=g)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        taus = torch.arange(HOP_STEP, HOP_MAX + 1e-9, HOP_STEP)
        Z = xi.reshape(1, 1, 2) + taus.reshape(1, -1, 1) * dirs.reshape(-1, 1, 2)    # (D, T, 2)
        dec = ref.sample(Z.reshape(-1, 2), torch.full((Z.shape[0] * Z.shape[1],), c)).reshape(Z.shape)
        cl = b.assign(dec.reshape(-1, 2)).reshape(Z.shape[0], Z.shape[1])
        best = None
        for k in range(Z.shape[0]):
            out = np.flatnonzero((cl[k] >= 0) & (cl[k] != b.bad))
            if len(out):
                tau = float(taus[out[0]]) + HOP_MARGIN
                if best is None or tau < best[0]:
                    best = (tau, k)
        if best is not None:
            zt = xi + best[0] * dirs[best[1]]
            t = ref.sample(zt.reshape(1, 2), torch.full((1,), c))[0]
            if b.assign(t.reshape(1, 2))[0] not in (-1, b.bad):
                return t
        if stats is not None:
            stats["hop_fallback"] = stats.get("hop_fallback", 0) + 1
        ok = lab != b.bad
    else:
        raise ValueError(rule)
    if not ok.any():
        if stats is not None:
            stats["no_eligible"] = stats.get("no_eligible", 0) + 1
        ok = np.ones_like(lab, dtype=bool)
    cand = pts[torch.from_numpy(ok)]
    return cand[int(torch.linalg.norm(cand - a, dim=1).argmin())]


@dataclass
class Rows:
    c: torch.Tensor
    noise: torch.Tensor
    target: torch.Tensor
    diag: Dict = field(default_factory=dict)


def build(ref, mix, seed, uids, c_neg, xi_neg, a_neg, rule, support, transplant=False,
          tx_conditions: Optional[torch.Tensor] = None, d0=None, pu=None, forget=None) -> Rows:
    cs, ns, ts = [], [], []
    stats: Dict = {}
    tx_bad = []
    for i in range(len(uids)):
        g = row_gen(seed, uids[i], 1)
        gs = row_gen(seed, uids[i], 2)
        ci, xi, a = float(c_neg[i]), xi_neg[i], a_neg[i]
        if rule == "ours":
            z = torch.randn(4, 2, generator=g)
            tgt = ref.sample(z, torch.full((4,), ci))
            cs += [ci] * 4
            ns += [xi] * 4
            ts += list(tgt)
            continue
        tgt = target_one(ref, rule, ci, xi, a, g, d0=d0, pu=pu, stats=stats)
        if support == "nhood":
            sup = torch.cat([xi.reshape(1, 2), xi + NHOOD_SIGMA * torch.randn(NHOOD_K, 2, generator=gs)])
        elif support == "bsup":
            b, _ = basins_at(ref, ci, a, row_gen(seed, uids[i], 1))      # same basins as the target
            jit = xi + BSUP_SIGMA * torch.randn(BSUP_K, 2, generator=gs)
            keep = b.assign(ref.sample(jit, torch.full((BSUP_K,), ci))) == b.bad
            sup = torch.cat([xi.reshape(1, 2), jit[torch.from_numpy(keep)][:BSUP_KEEP]])
        elif support == "log":
            sup = xi.reshape(1, 2)
        else:
            raise ValueError(support)
        cs += [ci] * len(sup)
        ns += list(sup)
        ts += [tgt] * len(sup)
        if transplant:
            gt = row_gen(seed, uids[i], 3)
            pick = torch.randint(0, tx_conditions.shape[0], (N_TX,), generator=gt)
            for cc in tx_conditions[pick]:
                cc = float(cc)
                a2 = ref.sample(xi.reshape(1, 2), torch.full((1,), cc))[0]
                t2 = target_one(ref, rule if rule != "own" else "basin", cc, xi, a2, gt, d0=d0, pu=pu, stats=stats)
                cs.append(cc)
                ns.append(xi)
                ts.append(t2)
                if forget is not None:
                    tx_bad.append(int(mix.assign(a2.reshape(1, 2), np.array([cc]))[0] in forget))
    C = torch.tensor(cs, dtype=torch.float32)
    N = torch.stack(ns)
    T = torch.stack(ts)
    diag = {"n_rows": int(len(cs)), **{f"stat_{k}": v for k, v in stats.items()}}
    if forget is not None:
        lab = mix.assign(T, C.numpy())
        diag["target_frac_in_forget"] = float(np.isin(lab, forget).mean())
        diag["target_frac_off"] = float((lab == -1).mean())
        diag["support_frac_in_forget"] = float(np.isin(mix.assign(ref.sample(N, C), C.numpy()), forget).mean())
        if tx_bad:
            diag["transplant_frac_bad"] = float(np.mean(tx_bad))
    return Rows(c=C, noise=N, target=T, diag=diag)
