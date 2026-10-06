"""The fairness properties the sweep relies on, checked on an UNTRAINED policy
(the properties are about RNG plumbing, not about what the policy learned)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from toypref.arms import ARMS
from toypref.editset import build_editset, forget_modes, make_pool
from toypref.flow import FlowPolicy, PolicyConfig, frozen_copy
from toypref.mixture import MixtureConfig
from toypref.reroute import build_rows

MIX = MixtureConfig()


def _ref():
    torch.manual_seed(0)
    pol = FlowPolicy(PolicyConfig(hidden_dim=32, n_layers=1))
    # zero-init output => identity map; perturb so decodes depend on the net
    for p in pol.net.out.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return frozen_copy(pol)


def _pool_with_modes(ref):
    pool = make_pool(ref, MIX, 0, 4000)
    # an untrained net lands nowhere in particular: assign modes by angle instead
    ang = torch.atan2(pool.a[:, 1], pool.a[:, 0]).numpy() % (2 * np.pi)
    pool.mode = (np.round(ang / (2 * np.pi / MIX.n_modes)).astype(int)) % MIX.n_modes
    return pool


def test_forget_sets_nested():
    for s in range(5):
        prev = set()
        for k in (1, 2, 3, 4):
            F = set(forget_modes(MIX, s, k))
            assert prev <= F and len(F) == k
            prev = F


def test_editsets_nested_in_n_neg():
    ref = _ref()
    pool = _pool_with_modes(ref)
    small = build_editset(pool, MIX, 0, 2, 2, 64)
    big = build_editset(pool, MIX, 0, 2, 4, 64)
    for m in small.forget:
        s_idx = small.neg_idx[small.mode_neg == m]
        b_idx = big.neg_idx[big.mode_neg == m]
        assert np.array_equal(s_idx, b_idx[:len(s_idx)])
    assert np.array_equal(small.pos_idx, big.pos_idx)


def test_targets_shared_across_supports_and_cells():
    ref = _ref()
    pool = _pool_with_modes(ref)
    es = build_editset(pool, MIX, 0, 2, 3, 64)
    es2 = build_editset(pool, MIX, 0, 2, 6, 64)
    r_log = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "spread", "log", 0)
    r_nh = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "spread", "nhood", 0)
    r_inv = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "spread", "inv_r1.5", 0)
    assert torch.equal(r_log.target, r_nh.target[r_nh.slot == 0])
    assert torch.equal(r_log.target, r_inv.target[r_inv.slot == 0])
    assert torch.equal(r_nh.noise[r_nh.slot == 0], es.xi_neg)
    # the same negative gets the same target and jitters in a bigger cell
    r_nh2 = build_rows(ref, es2.xi_neg, es2.a_neg, es2.neg_idx, "spread", "nhood", 0)
    for i, pidx in enumerate(es.neg_idx):
        j = int(np.flatnonzero(es2.neg_idx == pidx)[0])
        assert torch.equal(r_nh.noise[r_nh.neg_row == i], r_nh2.noise[r_nh2.neg_row == j])
        assert torch.equal(r_nh.target[r_nh.neg_row == i], r_nh2.target[r_nh2.neg_row == j])


def test_every_support_builds():
    ref = _ref()
    pool = _pool_with_modes(ref)
    es = build_editset(pool, MIX, 0, 1, 2, 16)
    for spec in ARMS.values():
        if spec.kind in ("reroute", "online"):
            rows = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, spec.target, spec.support, 0)
            assert torch.isfinite(rows.noise).all() and torch.isfinite(rows.target).all(), spec.name
            n_per = 4 if spec.target == "resample4" else (1 if spec.support == "log" else 5)
            assert rows.noise.shape[0] == n_per * len(es.neg_idx), spec.name


def test_inversion_roundtrip_small_on_identityish_flow():
    ref = _ref()
    xi = torch.randn(64, 2, generator=torch.Generator().manual_seed(1))
    a = ref.sample(xi)
    xi_hat = ref.invert(a, time_offset=1.0)
    assert float((xi_hat - xi).norm(dim=1).mean()) < 0.5
