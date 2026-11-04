"""Rollouts of pi_ref, the forget set, and the T_n / T_p split for one sweep cell.

A "rollout" in the toy is one draw ``xi ~ N(0, I)`` decoded by the frozen base
policy. Its label is the mode its action lands in. The forget set is a set of
mode labels; every logged rollout landing in one is a negative. Off-mode rollouts
carry no label either way and are dropped from both sets, as unresolved episodes
are in the preference repo.

Sweep axes (everything else fixed):
  n_forget  number of forgotten modes, the first n_forget of a per-seed random
            permutation of the modes (so forget sets are nested in n_forget)
  n_neg     logged negatives PER forgotten mode, the first n_neg pool rollouts
            landing in that mode (so edit sets are nested in n_neg)
T_p is the first ``n_pos`` pool rollouts landing in a retained mode, a fixed size
independent of both axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from . import seeds
from .mixture import OFF, MixtureConfig, assign_modes


@dataclass
class RolloutPool:
    xi: torch.Tensor        # (P, 2) logged noise
    a: torch.Tensor         # (P, 2) pi_ref action
    mode: np.ndarray        # (P,)   label, OFF = -1


@torch.no_grad()
def make_pool(policy, mix: MixtureConfig, seed: int, n: int, purpose: int = seeds.POOL) -> RolloutPool:
    g = seeds.gen(seed, purpose)
    xi = torch.randn(n, mix.dim, generator=g)
    a = policy.sample(xi)
    return RolloutPool(xi=xi, a=a, mode=assign_modes(mix, a))


def forget_modes(mix: MixtureConfig, seed: int, n_forget: int) -> List[int]:
    perm = seeds.np_rng(seed, seeds.FORGET_PERM).permutation(mix.n_modes)
    return sorted(int(m) for m in perm[:n_forget])


@dataclass
class EditSet:
    forget: List[int]
    retain: List[int]
    neg_idx: np.ndarray      # pool indices of T_n
    pos_idx: np.ndarray      # pool indices of T_p
    xi_neg: torch.Tensor
    a_neg: torch.Tensor
    mode_neg: np.ndarray
    xi_pos: torch.Tensor
    a_pos: torch.Tensor
    mode_pos: np.ndarray


def build_editset(pool: RolloutPool, mix: MixtureConfig, seed: int, n_forget: int,
                  n_neg: int, n_pos: int) -> EditSet:
    F = forget_modes(mix, seed, n_forget)
    R = [m for m in range(mix.n_modes) if m not in F]
    neg = []
    for m in F:
        idx = np.flatnonzero(pool.mode == m)[:n_neg]
        if len(idx) < n_neg:
            raise ValueError(f"pool has only {len(idx)} rollouts in mode {m}, need {n_neg}")
        neg.append(idx)
    neg_idx = np.concatenate(neg)
    in_R = np.isin(pool.mode, R)
    pos_idx = np.flatnonzero(in_R)[:n_pos]
    if len(pos_idx) < n_pos:
        raise ValueError("pool too small for n_pos")
    return EditSet(
        forget=F, retain=R, neg_idx=neg_idx, pos_idx=pos_idx,
        xi_neg=pool.xi[neg_idx], a_neg=pool.a[neg_idx], mode_neg=pool.mode[neg_idx],
        xi_pos=pool.xi[pos_idx], a_pos=pool.a[pos_idx], mode_pos=pool.mode[pos_idx],
    )


def apply_regime(es: EditSet, pool: RolloutPool, mix: MixtureConfig, seed: int, n_pos: int,
                 cover: float = 1.0, contam: float = 0.0) -> EditSet:
    """Positive-set regimes (round "where does rerouting win").

    cover < 1   T_p comes only from the first ceil(cover * |R|) retained modes of a
                per-seed permutation (positives that do not cover every retained
                behaviour, as when successful rollouts visit only some states)
    contam > 0  the first round(contam * n_pos) rows of T_p are replaced by pool
                rollouts that landed in a FORGOTTEN mode but are not logged
                negatives (label noise in the positive set)
    T_n is untouched. Both are nested in cover / contam at fixed seed.
    """
    import math
    if cover >= 1.0 and contam <= 0.0:
        return es
    covered = list(es.retain)
    if cover < 1.0:
        perm = seeds.np_rng(seed, seeds.COVER_PERM).permutation(sorted(es.retain))
        covered = sorted(int(m) for m in perm[:max(1, math.ceil(cover * len(es.retain)))])
    pos_idx = np.flatnonzero(np.isin(pool.mode, covered))[:n_pos]
    if contam > 0.0:
        m = int(round(contam * n_pos))
        bad = np.flatnonzero(np.isin(pool.mode, es.forget) & ~np.isin(np.arange(len(pool.mode)), es.neg_idx))[:m]
        pos_idx = np.concatenate([bad, pos_idx[: n_pos - len(bad)]])
    es.pos_idx = pos_idx
    es.xi_pos, es.a_pos, es.mode_pos = pool.xi[pos_idx], pool.a[pos_idx], pool.mode[pos_idx]
    es.covered = covered
    return es
