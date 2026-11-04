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
            rows = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, spec.target, spec.support, 0,
                              mix_cfg=MIX, forget=es.forget, d0=1.0)
            assert torch.isfinite(rows.noise).all() and torch.isfinite(rows.target).all(), spec.name
            n_per = 4 if spec.target.startswith("resample4") else (
                1 if spec.support == "log" else (1 + int(spec.support.split("_")[-1]) if spec.support.startswith("nhoodx_") else 5))
            assert rows.noise.shape[0] == n_per * len(es.neg_idx), spec.name


def test_inversion_roundtrip_small_on_identityish_flow():
    ref = _ref()
    xi = torch.randn(64, 2, generator=torch.Generator().manual_seed(1))
    a = ref.sample(xi)
    xi_hat = ref.invert(a, time_offset=1.0)
    assert float((xi_hat - xi).norm(dim=1).mean()) < 0.5


class _RingRef:
    """Stub reference whose decode of z is the ring centre nearest z's angle (+ a
    small deterministic offset), so decodes land in real modes and labels mean
    something. Only ``sample`` is needed by the target builders."""

    def sample(self, z, repel_from=None, repel_alpha=0.0):
        C = torch.from_numpy(MIX.centers())
        ang = torch.atan2(z[:, 1], z[:, 0]) % (2 * np.pi)
        m = torch.round(ang / (2 * np.pi / MIX.n_modes)).long() % MIX.n_modes
        return C[m] + 0.05 * torch.tanh(z)


def _ring_editset(k, n, n_pos=64):
    ref = _RingRef()
    xi = torch.randn(4000, 2, generator=torch.Generator().manual_seed(3))
    from toypref.editset import RolloutPool
    from toypref.mixture import assign_modes
    a = ref.sample(xi)
    pool = RolloutPool(xi=xi, a=a, mode=assign_modes(MIX, a))
    return ref, build_editset(pool, MIX, 0, k, n, n_pos)


def test_filtered_targets_never_in_forget_modes():
    from toypref.mixture import assign_modes
    for k in (1, 2, 4):
        ref, es = _ring_editset(k, 4)
        for tgt, sup in (("spread_filt", "log"), ("spread_filt", "nhood"), ("resample4_filt", "log")):
            rows = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, tgt, sup, 0, mix_cfg=MIX, forget=es.forget)
            lab = assign_modes(MIX, rows.target)
            assert not np.isin(lab, es.forget).any(), (k, tgt)
            assert rows.diag["filt_fallback_unfiltered"] == 0


def test_filtered_equals_unfiltered_when_already_retained():
    from toypref.mixture import assign_modes
    ref, es = _ring_editset(2, 8)
    u = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "spread", "log", 0, mix_cfg=MIX, forget=es.forget)
    f = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "spread_filt", "log", 0, mix_cfg=MIX, forget=es.forget)
    ok = ~np.isin(assign_modes(MIX, u.target), es.forget)
    assert ok.any()
    assert torch.equal(u.target[torch.from_numpy(ok)], f.target[torch.from_numpy(ok)])
    assert torch.equal(u.noise, f.noise)
    # Ours: retained draws of the first block are kept, in order
    u4 = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "resample4", "log", 0)
    f4 = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "resample4_filt", "log", 0, mix_cfg=MIX, forget=es.forget)
    for i in range(len(es.neg_idx)):
        ut = u4.target[u4.neg_row == i]
        keep = ut[torch.from_numpy(~np.isin(assign_modes(MIX, ut), es.forget))]
        assert torch.equal(f4.target[f4.neg_row == i][:keep.shape[0]], keep)


def test_positives_nested_in_n_pos():
    _, small = _ring_editset(2, 2, n_pos=4)
    _, big = _ring_editset(2, 2, n_pos=64)
    assert np.array_equal(small.pos_idx, big.pos_idx[:4])
    assert np.array_equal(small.neg_idx, big.neg_idx)


def test_label_free_targets_avoid_negatives():
    for k in (1, 2, 4):
        ref, es = _ring_editset(k, 4)
        for tgt in ("near_nf", "spread_nf", "resample4_nf"):
            rows = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, tgt, "log", 0, mix_cfg=MIX,
                              forget=es.forget, d0=1.0)
            d = torch.cdist(rows.target, es.a_neg).min(1).values
            assert bool((d > 1.0).all()), (k, tgt)
            assert rows.diag["filt_fallback_unfiltered"] == 0


def test_near_targets_are_nearest_eligible():
    from toypref.mixture import assign_modes
    ref, es = _ring_editset(2, 3)
    rows = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "near_filt", "log", 0, mix_cfg=MIX, forget=es.forget)
    assert not np.isin(assign_modes(MIX, rows.target), es.forget).any()
    far = build_rows(ref, es.xi_neg, es.a_neg, es.neg_idx, "spread_filt", "log", 0, mix_cfg=MIX, forget=es.forget)
    dn = torch.linalg.norm(rows.target - es.a_neg, dim=1)
    df = torch.linalg.norm(far.target - es.a_neg, dim=1)
    assert float(dn.mean()) < float(df.mean())


def test_regime_cover_and_contam():
    from toypref.editset import RolloutPool, apply_regime
    from toypref.mixture import assign_modes
    ref = _RingRef()
    xi = torch.randn(8000, 2, generator=torch.Generator().manual_seed(3))
    a = ref.sample(xi)
    pool = RolloutPool(xi=xi, a=a, mode=assign_modes(MIX, a))
    es = build_editset(pool, MIX, 0, 2, 4, 64)
    c = apply_regime(build_editset(pool, MIX, 0, 2, 4, 64), pool, MIX, 0, 64, cover=0.5)
    assert set(np.unique(c.mode_pos)) <= set(c.covered) and len(c.covered) == 3
    assert np.array_equal(c.neg_idx, es.neg_idx)
    k = apply_regime(build_editset(pool, MIX, 0, 2, 4, 64), pool, MIX, 0, 64, contam=0.25)
    assert int(np.isin(k.mode_pos, es.forget).sum()) == 16 and len(k.pos_idx) == 64
    assert not np.isin(k.pos_idx, es.neg_idx).any()
