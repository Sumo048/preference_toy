"""Rerouting: re-pair each logged negative noise with a new pi_ref action.

Ported from the preference repo (``reroute_dataset.py``, ``reroute_targets.py``,
``reroute_ablation.py``, ``reroute_inverse.py``, ``select_neighbourhood.py``) with
the defaults it ran:

TARGET (what the rerouted noise is trained to decode to; frozen pi_ref, stop-grad)
  resample4  Ours / ``ours_v2_r2``: 4 rows per negative, row k targets an
             independent fresh decode a_k = pi_ref(z_k)
  spread     M=256 prior pool, farthest-point sampling in NOISE space seeded by the
             pool point farthest from xi_log, 8 chosen, decoded, and the decode
             farthest from a_neg (action space) is the target
  guide      8 i.i.d. noises decoded under the repulsive sampler
             v' = v - alpha (x - a_neg), alpha = 0.3; farthest decode is the target

SUPPORT (which input noises carry that target; slot 0 is always xi_log)
  log              [xi_log]
  nhood            + 4 jitters xi_log + 0.3 eps (additive)
  nhood_sph_<s>    + 4 jitters sqrt(1-s^2) xi_log + s eps (s = 0.485, 0.867)
  inv_r<rho>       + 4 inversions of a' uniform in the L2 ball of radius rho about
                   a_neg (patch 0019 Euler inverse, forward-grid time indexing)
  reach_<lo>_<hi>  + the prior noises of ranks lo:hi among M=32 prior decodes
                   sorted by distance to a_neg (near 0:4, mid 8:12, far 16:20)
  select_<s>       + the 4 of M=16 spherical jitters (sigma s) whose decodes are
                   nearest a_neg
  ninv_<t0>_<e>    + 4 noised partial inversions: x_t0 = (1-t0) a_neg + t0 e,
                   e fresh N(0,I) ("fresh") or sqrt(1-0.25) xi_log + 0.5 eps
                   ("mix"), reversed t0 -> 1 with FRS indexing

Every draw comes from a per-row stream keyed by the negative's pool index
(``seeds.row_gen``), so targets and supports are identical across arms and cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch

from . import seeds

N_CANDIDATES = 8
POOL_M = 256
GUIDE_ALPHA = 0.3
N_RESAMPLE = 4
NHOOD_K = 4
NHOOD_SIGMA = 0.3
REACH_M = 32
SELECT_M = 16
MIX_SIGMA = 0.5


def farthest_point_noises(pool: torch.Tensor, xi_log: torch.Tensor, n: int) -> torch.Tensor:
    d0 = torch.linalg.norm(pool - xi_log.reshape(1, -1), dim=1)
    chosen = [int(torch.argmax(d0))]
    mind = torch.linalg.norm(pool - pool[chosen[0]].reshape(1, -1), dim=1)
    for _ in range(n - 1):
        nxt = int(torch.argmax(mind))
        chosen.append(nxt)
        mind = torch.minimum(mind, torch.linalg.norm(pool - pool[nxt].reshape(1, -1), dim=1))
    return pool[torch.tensor(chosen)]


@torch.no_grad()
def make_targets(ref, xi_log, a_neg, method: str, seed: int, pool_idx, round_: int = 0):
    """(R, n_t, 2) targets per negative: n_t = 4 for resample4, else 1."""
    out = []
    D = xi_log.shape[1]
    for i in range(xi_log.shape[0]):
        g = seeds.row_gen(seed, int(pool_idx[i]), seeds.ROW_CAND if method != "resample4" else seeds.ROW_RESAMPLE, round_)
        if method == "resample4":
            z = torch.randn(N_RESAMPLE, D, generator=g)
            out.append(ref.sample(z))
            continue
        if method == "spread":
            pool = torch.randn(POOL_M, D, generator=g)
            z = farthest_point_noises(pool, xi_log[i], N_CANDIDATES)
            cand = ref.sample(z)
        elif method == "guide":
            z = torch.randn(N_CANDIDATES, D, generator=g)
            cand = ref.sample(z, repel_from=a_neg[i].expand(N_CANDIDATES, -1), repel_alpha=GUIDE_ALPHA)
        else:
            raise ValueError(method)
        d = torch.linalg.norm(cand - a_neg[i].reshape(1, -1), dim=1)
        out.append(cand[int(torch.argmax(d))].reshape(1, D))
    return torch.stack(out)


@torch.no_grad()
def make_support(ref, xi_log, a_neg, support: str, seed: int, pool_idx, round_: int = 0):
    """(R, S, 2) input noises per negative; slot 0 is xi_log."""
    R, D = xi_log.shape
    rows = []
    for i in range(R):
        g = seeds.row_gen(seed, int(pool_idx[i]), seeds.ROW_SUPPORT, round_)
        xi, an = xi_log[i:i + 1], a_neg[i:i + 1]
        if support == "log":
            extra = xi[:0]
        elif support == "nhood":
            extra = xi + NHOOD_SIGMA * torch.randn(NHOOD_K, D, generator=g)
        elif support.startswith("nhood_sph_"):
            s = float(support.split("_")[-1])
            extra = (1 - s * s) ** 0.5 * xi + s * torch.randn(NHOOD_K, D, generator=g)
        elif support.startswith("inv_r"):
            rho = float(support[len("inv_r"):])
            d = torch.randn(NHOOD_K, D, generator=g)
            d = d / torch.linalg.norm(d, dim=1, keepdim=True)
            u = torch.rand(NHOOD_K, 1, generator=g)
            a_ball = an + rho * u ** (1.0 / D) * d
            extra = ref.invert(a_ball, time_offset=1.0)
        elif support.startswith("reach_"):
            lo, hi = (int(v) for v in support.split("_")[1:3])
            z = torch.randn(REACH_M, D, generator=g)
            dist = torch.linalg.norm(ref.sample(z) - an, dim=1)
            extra = z[torch.argsort(dist)[lo:hi]]
        elif support.startswith("select_"):
            s = float(support.split("_")[1])
            z = (1 - s * s) ** 0.5 * xi + s * torch.randn(SELECT_M, D, generator=g)
            dist = torch.linalg.norm(ref.sample(z) - an, dim=1)
            extra = z[torch.argsort(dist)[:NHOOD_K]]
        elif support.startswith("ninv_"):
            _, t0, kind = support.split("_")
            t0 = float(t0)
            if kind == "fresh":
                e = torch.randn(NHOOD_K, D, generator=g)
            elif kind == "mix":
                e = (1 - MIX_SIGMA ** 2) ** 0.5 * xi + MIX_SIGMA * torch.randn(NHOOD_K, D, generator=g)
            else:
                raise ValueError(support)
            x_t0 = (1 - t0) * an + t0 * e
            extra = ref.invert(x_t0, t_start=t0, time_offset=0.0)
        else:
            raise ValueError(f"unknown support {support!r}")
        rows.append(torch.cat([xi, extra], 0))
    S = {r.shape[0] for r in rows}
    assert len(S) == 1, S
    return torch.stack(rows)


@dataclass
class RerouteRows:
    noise: torch.Tensor      # (N, 2) pinned input noise
    target: torch.Tensor     # (N, 2)
    neg_row: torch.Tensor    # (N,) index into T_n
    slot: torch.Tensor       # (N,)
    diag: Dict = field(default_factory=dict)


@torch.no_grad()
def build_rows(ref, xi_neg, a_neg, pool_idx, target: str, support: str, seed: int,
               mix_cfg=None, forget=None, round_: int = 0) -> RerouteRows:
    T = make_targets(ref, xi_neg, a_neg, target, seed, pool_idx, round_)          # (R, n_t, 2)
    if target == "resample4":
        if support != "log":
            raise ValueError("Ours (resample4) is defined on S-log only")
        R, n_t, D = T.shape
        noise = xi_neg.repeat_interleave(n_t, 0)
        tgt = T.reshape(R * n_t, D)
        neg_row = torch.arange(R).repeat_interleave(n_t)
        slot = torch.arange(n_t).repeat(R)
        sup = xi_neg[:, None, :]
    else:
        sup = make_support(ref, xi_neg, a_neg, support, seed, pool_idx, round_)   # (R, S, 2)
        R, S, D = sup.shape
        noise = sup.reshape(R * S, D)
        tgt = T[:, 0, :].repeat_interleave(S, 0)
        neg_row = torch.arange(R).repeat_interleave(S)
        slot = torch.arange(S).repeat(R)

    diag = {
        "n_rows": int(noise.shape[0]),
        "target_dist_to_neg_mean": float(torch.linalg.norm(tgt - a_neg[neg_row], dim=1).mean()),
        "support_noise_norm_mean": float(torch.linalg.norm(noise, dim=1).mean()),
        "support_noise_norm_max": float(torch.linalg.norm(noise, dim=1).max()),
        # how far the extra support noises decode from a_neg under pi_ref (ball_dist)
        "support_decode_dist_to_neg_mean": float(
            torch.linalg.norm(ref.sample(noise) - a_neg[neg_row], dim=1)[slot > 0].mean()
        ) if bool((slot > 0).any()) else 0.0,
        "support_dist_to_xi_log_mean": float(
            torch.linalg.norm(noise - xi_neg[neg_row], dim=1)[slot > 0].mean()
        ) if bool((slot > 0).any()) else 0.0,
    }
    if mix_cfg is not None and forget is not None:
        from .mixture import OFF, assign_modes
        lab = assign_modes(mix_cfg, tgt)
        diag["target_frac_in_forget"] = float(np.isin(lab, forget).mean())
        diag["target_frac_off"] = float((lab == OFF).mean())
        diag["support_decode_frac_in_forget"] = float(np.isin(assign_modes(mix_cfg, ref.sample(noise)), forget).mean())
    return RerouteRows(noise=noise, target=tgt, neg_row=neg_row, slot=slot, diag=diag)
