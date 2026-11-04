"""Baseline losses, ported from patches 0010/0014/0016 of the preference repo.

All likelihood-ratio methods share the Diffusion-DPO-style surrogate

    r_i = sg(l_ref_i) - l_theta_i

with l the per-row flow-matching loss at the SAME (t, noise) for theta and the
frozen reference, one (t, noise) sample per row per step, beta applied directly
to r (no T or w(t) factor). Negatives get FRESH noise every step
(``FRESH_NOISE_ROWS=non_rerouted``), exactly like positives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def surrogate_r(policy, ref, a, noise, t):
    l_theta = policy.flow_loss(a, noise, t)
    with torch.no_grad():
        l_ref = ref.flow_loss(a, noise, t)
    return l_ref - l_theta, l_theta


def npo_term(r, beta):
    """(2/beta) softplus(beta r) = (2/beta) log(1 + (pi_theta/pi_ref)^beta)."""
    return (2.0 / beta) * F.softplus(beta * r)


def kto_values(r, z0, is_neg, beta, lambda_d, lambda_u):
    v = torch.where(is_neg, lambda_u * torch.sigmoid(beta * (z0 - r)),
                    lambda_d * torch.sigmoid(beta * (r - z0)))
    lam = torch.where(is_neg, torch.full_like(v, lambda_u), torch.full_like(v, lambda_d))
    return lam - v


def kto_z0(policy, ref, a, noise, t, rows: int = 8):
    """Detached max(0, mean r') over MISMATCHED rows (action+noise rolled by one).

    Unconditional toy: with no observation to keep fixed, the roll only re-pairs
    rows within the sub-batch, so z0 estimates KL(theta || ref) from batch draws.
    """
    k = min(rows, a.shape[0])
    with torch.no_grad():
        r_mis, _ = surrogate_r(policy, ref, torch.roll(a, 1, 0)[:k], torch.roll(noise, 1, 0)[:k], t[:k])
    return torch.clamp(r_mis.mean(), min=0.0)


def dpo_pair_term(r_w, r_l, beta):
    """Flow-DPO (``cpl``): -log sigmoid(beta (r_w - r_l)), shared t within a pair."""
    return -F.logsigmoid(beta * (r_w - r_l))


def kl_velocity(policy, ref, a, noise, t):
    """Patch 0018: mean (v_theta - sg(v_ref))^2 at one point of the path."""
    tt = t.reshape(-1, 1)
    x_t = tt * noise + (1 - tt) * a
    v = policy.net(x_t, t)
    with torch.no_grad():
        v_ref = ref.net(x_t, t)
    return ((v - v_ref) ** 2).mean(-1)
