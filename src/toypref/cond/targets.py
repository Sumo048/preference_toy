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


def _hop_candidates(cl, bad, taus, max_per_dir=3):
    """(tau, direction) of the first decode of each NEW non-bad behaviour along each direction."""
    out = []
    for k in range(cl.shape[0]):
        seen, prev = 0, bad
        for t in range(cl.shape[1]):
            v = cl[k, t]
            if v >= 0 and v != bad and v != prev:
                out.append((float(taus[t]), k))
                seen += 1
                if seen >= max_per_dir:
                    break
            if v >= 0:
                prev = v
    return sorted(out)


def target_one(ref, rule, c, xi, a, g, d0=None, pu=None, stats=None, veto=None, info=None):
    """One target (2,) for the negative (c, xi, a) under `rule`.

    ``veto`` (hop only): callable(noises (P, 2)) -> bool mask of vetoed exits. Exits are tried in
    order of noise distance; with no veto the first exit is exactly the plain hop's choice."""
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
        if veto is not None:
            cands = _hop_candidates(cl, b.bad, taus)
            if cands:
                zs = torch.stack([xi + (tau + HOP_MARGIN) * dirs[k] for tau, k in cands])
                ts = ref.sample(zs, torch.full((zs.shape[0],), c))
                lab_ok = np.array([v not in (-1, b.bad) for v in b.assign(ts)])
                vet = veto(zs).numpy() if lab_ok.any() else np.zeros(len(cands), dtype=bool)
                ok = np.flatnonzero(lab_ok & ~vet)
                if info is not None:
                    info["n_cand"] = len(cands)
                    info["n_vetoed"] = int((lab_ok & vet).sum())
                if len(ok):
                    if info is not None:
                        info["tau"] = cands[ok[0]][0] + HOP_MARGIN
                    return ts[ok[0]]
            if stats is not None:
                stats["veto_fallback"] = stats.get("veto_fallback", 0) + 1
        best = None
        for k in range(Z.shape[0]):
            out = np.flatnonzero((cl[k] >= 0) & (cl[k] != b.bad))
            if len(out):
                tau = float(taus[out[0]]) + HOP_MARGIN
                if best is None or tau < best[0]:
                    best = (tau, k)
        if best is not None:
            if info is not None:
                info["tau"] = best[0]
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
    vetoes = None
    if rule in ("hopV", "hopB", "hopVB"):
        vetoes = make_vetoes(ref, seed, uids, c_neg, xi_neg, a_neg, rule, stats)
        rule_eff = "hop"
    else:
        rule_eff = rule
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
        tgt = target_one(ref, rule_eff, ci, xi, a, g, d0=d0, pu=pu, stats=stats,
                         veto=vetoes[i] if vetoes is not None else None)
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
                t2 = target_one(ref, rule_eff if rule_eff != "own" else "basin", cc, xi, a2, gt, d0=d0, pu=pu, stats=stats)
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


def make_vetoes(ref, seed, uids, c_neg, xi_neg, a_neg, rule, stats):
    """Per-negative veto callables built from the OTHER negatives (round 2, see prereg).

    V: noise proximity, rho = 0.5 x median plain-hop exit distance over this cell's negatives.
    B: basin vote, decode the exit noise at every other negative's condition and count how
       often it lands in that negative's bad basin; veto at >= 0.5 x the median such rate for
       the logged bad noises themselves."""
    N = len(uids)
    use_v, use_b = "V" in rule[3:], "B" in rule[3:]
    basins = [basins_at(ref, float(c_neg[j]), a_neg[j], row_gen(seed, uids[j], 1))[0] for j in range(N)]
    rho = None
    if use_v:
        taus = []
        for j in range(N):
            inf = {}
            target_one(ref, "hop", float(c_neg[j]), xi_neg[j], a_neg[j], row_gen(seed, uids[j], 1), info=inf)
            if "tau" in inf:
                taus.append(inf["tau"])
        rho = 0.5 * float(np.median(taus)) if taus else 0.0
        stats["veto_rho"] = rho

    def bad_rate(z, skip):
        """(P,) fraction of other negatives j whose bad basin z decodes into at c_j."""
        P = z.shape[0]
        others = [j for j in range(N) if j != skip]
        if not others:
            return torch.zeros(P)
        zz = z.repeat(len(others), 1)
        cc = torch.cat([torch.full((P,), float(c_neg[j])) for j in others])
        dec = ref.sample(zz, cc)
        hits = torch.zeros(P)
        for n, j in enumerate(others):
            lab = basins[j].assign(dec[n * P:(n + 1) * P])
            hits += torch.from_numpy((lab == basins[j].bad).astype(np.float32))
        return hits / len(others)

    theta = None
    if use_b:
        f = [float(bad_rate(xi_neg[j].reshape(1, 2), j)[0]) for j in range(N)] if N > 1 else []
        med = float(np.median(f)) if f else 0.0
        theta = 0.5 * med if med > 0 else None
        stats["veto_theta"] = -1.0 if theta is None else theta
        stats["veto_self_rate_median"] = med

    out = []
    for i in range(N):
        others = torch.cat([xi_neg[:i], xi_neg[i + 1:]]) if N > 1 else xi_neg[:0]

        def veto(z, i=i, others=others):
            m = torch.zeros(z.shape[0], dtype=torch.bool)
            if use_v and others.shape[0] and rho:
                m |= torch.cdist(z, others).min(1).values < rho
            if use_b and theta is not None:
                m |= bad_rate(z, i) >= theta
            return m
        out.append(veto)
    return out
