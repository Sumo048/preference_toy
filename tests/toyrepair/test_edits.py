"""W1-B: edit API, graft identity, unit-test hook, patches and rebuild (PLAN 6.7, 4.7)."""
import copy
import os

import numpy as np
import pytest
import torch

import toyrepair
from toyrepair import edits as E
from toyrepair import patches as P
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))
have_uncond = os.path.exists(os.path.join(RUNS, "base", "s0", "policy.pt"))
R = np.random.default_rng(0)


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0 (run scripts/cloud_setup.sh)")
    return load_cell("cond", "k4", 0, 1)


@pytest.fixture(scope="module")
def v0(cell):
    return E.base_version(cell)


@pytest.fixture(scope="module")
def states(v0):
    return E.reference_states(v0.net, E.N_IDENTITY_STATES, seed=3)


def det(source, d, b=0.0, thr=0.0, k=None, scale=1.0):
    return E.Detector(source, tuple((R.standard_normal(d) * scale).astype(np.float32)), b, thr, k)


def graft(v, new_id, layer="blocks.1.fc2", d=None, n=8, scale=0.1, zero=False, **kw):
    src_dim = E.source_dims(v.net)[E.DEFAULT_SRC[layer]]
    out = E.linear(v.net, layer).out_features
    w_out = np.zeros((out, n)) if zero else R.standard_normal((out, n)) * scale
    return E.graft_gated_path(v, new_id, layer, d or det("cond", 128), n, R.standard_normal((n, src_dim)) * scale,
                              w_out, **kw)


# ------------------------------------------------------------------ forward / decode
def test_run_forward_and_decode_bit_identical(cell, v0, states):
    x, t, c = states
    assert x.shape == (10_000, 2)
    v_net = v0.net(x, t, c)
    v_rf, acts = E.run_forward(v0.net, x, t, c, capture=("cond", "blocks.2.act", "out"))
    assert torch.equal(v_net, v_rf) and torch.equal(acts["out"], v_net)
    assert acts["cond"].shape == (10_000, 128) and acts["blocks.2.act"].shape == (10_000, 128)
    xi, cc = torch.randn(2048, 2, generator=torch.Generator().manual_seed(1)), torch.rand(2048)
    assert torch.equal(cell.ref.sample(xi, cc), E.decode_actions(v0.net, xi, cc))


# ------------------------------------------------------------------ graft identity (PLAN 6.7)
def test_graft_identity_detector_off_10000_states(v0, states):
    x, t, c = states
    # a detector that never fires on these states
    d = E.Detector("inputs", (0.0, 0.0, 0.0, 1.0), b=0.0, threshold=10.0)
    for layer in E.EDITABLE:
        r = graft(v0, "v1", layer=layer, d=d, scale=1.0)
        assert r.record.ok, r.record.error
        assert torch.equal(r.version.net(x, t, c), v0.net(x, t, c)), layer
    # a detector that fires on part of the states: forced off -> identical everywhere,
    # naturally off rows -> identical, firing rows -> changed
    r = graft(v0, "v2", d=det("blocks.0.act", 128, thr=0.5), scale=1.0)
    nn_ = r.version.net
    gid = nn_.grafts[0].gid
    off, _ = E.run_forward(nn_, x, t, c, disabled=[gid])
    assert torch.equal(off, v0.net(x, t, c))
    fired = {}
    on, _ = E.run_forward(nn_, x, t, c, fired=fired)
    m = fired[gid]
    assert 0 < int(m.sum()) < m.numel()
    assert torch.equal(on[~m], v0.net(x, t, c)[~m])
    assert not torch.equal(on[m], v0.net(x, t, c)[m])
    assert E.graft_identity(v0, r.version)
    assert E.unit_tests(r.version, v0).passed


def test_zero_output_graft_bit_identical_decodes(cell, v0):
    always = E.Detector("t", (1.0,), b=0.0, threshold=-1.0)          # t > -1: fires everywhere
    xi, c = torch.randn(4096, 2, generator=torch.Generator().manual_seed(2)), torch.rand(4096)
    a0 = E.decode_actions(v0.net, xi, c)
    for layer in E.EDITABLE:
        r = graft(v0, "v1", layer=layer, d=always, zero=True, scale=1.0)
        assert torch.equal(E.decode_actions(r.version.net, xi, c), a0), layer
    # and a non-zero graft that always fires does change decodes (the path is live)
    r = graft(v0, "v1", layer="out", d=always, scale=1.0)
    assert not torch.equal(E.decode_actions(r.version.net, xi, c), a0)


def test_k_range_gate(v0, states):
    x, t, c = states
    d = E.Detector("t", (1.0,), 0.0, -1.0, k_range=(4, 8))
    r = graft(v0, "v1", layer="out", d=d, scale=1.0)
    k = torch.round((1 - t) * 20)
    inside = (k >= 4) & (k < 8)
    diff = (r.version.net(x, t, c) != v0.net(x, t, c)).any(1)
    assert torch.equal(diff, inside)


# ------------------------------------------------------------------ rejection of illegal detectors
@pytest.mark.parametrize("bad", [
    E.Detector("oracle", (1.0,)),
    E.Detector("label", (1.0,)),
    E.Detector("mode", (1.0,) * 8),
    E.Detector("forget", (1.0,)),
    E.Detector("a_neg", (1.0, 1.0)),
    E.Detector("cond", (1.0,) * 3),                       # wrong size
    E.Detector("cond", (float("nan"),) * 128),            # NaN
    E.Detector("blocks.2.act", (1.0,) * 128),             # not causal for blocks.1.fc2
    E.Detector("x", (1.0, 1.0), threshold=float("inf")),
    E.Detector("x", (1.0, 1.0), k_range=(5, 30)),
])
def test_api_rejects_illegal_detectors(v0, bad):
    r = graft(v0, "v1", layer="blocks.1.fc2", d=bad)
    assert not r.record.ok and r.version is None and r.patch is None and r.record.error


def test_api_rejects_non_detector_objects(v0):
    class Sneaky(E.Detector):
        pass
    for d in (Sneaky("x", (1.0, 1.0)), lambda z: z, {"source": "x", "w": (1.0, 1.0)}):
        r = graft(v0, "v1", d=d)
        assert not r.record.ok


def test_api_rejects_bad_graft_args(v0):
    d = det("cond", 128)
    assert not E.graft_gated_path(v0, "v1", "time_mlp.0", d, 2, np.zeros((2, 64)), np.zeros((64, 2))).record.ok
    assert not E.graft_gated_path(v0, "v1", "out", d, 2, np.zeros((2, 127)), np.zeros((2, 2))).record.ok
    assert not E.graft_gated_path(v0, "v1", "out", d, 2, np.zeros((2, 128)), np.full((2, 2), np.nan)).record.ok
    assert not E.graft_gated_path(v0, "v1", "out", d, 2, np.zeros((2, 128)), np.zeros((2, 2)), act="tanh").record.ok
    assert not E.graft_gated_path(v0, "v1", "out", d, 2, np.zeros((2, 128)), np.zeros((2, 2)), src="out").record.ok
    assert E.graft_gated_path(v0, "v1", "out", d, 2, np.zeros((2, 4)), np.zeros((2, 2)), src="inputs").record.ok


def test_unit_tests_flag_bad_versions(v0):
    r = graft(v0, "v1", d=det("blocks.0.act", 128, thr=0.5))
    assert E.unit_tests(r.version, v0).passed
    # NaN weight
    bad = copy.deepcopy(r.version)
    E.base_of(bad.net).out.weight.data[0, 0] = float("nan")
    u = E.unit_tests(bad, v0)
    assert not u.passed and not u.checks["finite"]
    # a detector source swapped behind the API's back
    bad = copy.deepcopy(r.version)
    bad.net.grafts[0].det_source = "oracle"
    u = E.unit_tests(bad, v0)
    assert not u.passed and not u.checks["legal_detectors"]
    # shapes
    bad = copy.deepcopy(r.version)
    bad.net.grafts[0].w_out = torch.nn.Parameter(torch.zeros(128, 3), requires_grad=False)
    assert not E.unit_tests(bad, v0).checks["shapes"]
    # identity broken: a network whose forward leaks past the gate
    class Leaky(E.EditedNet):
        def forward(self, x, t, c=None):
            return E.run_forward(self, x, t, c)[0] + 1e-4
    bad = copy.deepcopy(r.version)
    bad.net = Leaky(bad.net.base, list(bad.net.grafts))
    u = E.unit_tests(bad, v0)
    assert not u.passed and not u.checks["graft_identity"] and not u.checks["legal_detectors"]


# ------------------------------------------------------------------ weight edits
def test_rank1(v0):
    u, v = R.standard_normal(128).astype(np.float32), R.standard_normal(256).astype(np.float32)
    r = E.rank1(v0, "v1", "blocks.2.fc1", u, v, 0.03)
    assert r.record.ok
    W0 = E.linear(v0.net, "blocks.2.fc1").weight.numpy()
    W1 = E.linear(r.version.net, "blocks.2.fc1").weight.numpy()
    np.testing.assert_allclose(W1 - W0, 0.03 * np.outer(u, v), atol=1e-6)
    assert r.record.frac_weights_changed > 0 and abs(r.record.frob - 0.03 * np.linalg.norm(u) * np.linalg.norm(v)) < 1e-3
    assert torch.equal(E.linear(v0.net, "blocks.2.fc1").weight, torch.from_numpy(W0))      # parent untouched
    assert not E.rank1(v0, "v1", "blocks.2.fc1", u, v[:5], 1.0).record.ok
    assert not E.rank1(v0, "v1", "blocks.9.fc1", u, v, 1.0).record.ok
    assert not E.rank1(v0, "v1", "out", np.ones(2), np.ones(128), float("nan")).record.ok


def test_ablate_direction_pre_and_post(v0):
    v = R.standard_normal(128).astype(np.float32)
    vn = v / np.linalg.norm(v)
    r = E.ablate_direction(v0, "v1", "blocks.0.fc2", v, "pre")
    lin = E.linear(r.version.net, "blocks.0.fc2")
    assert np.abs(vn @ lin.weight.numpy()).max() < 1e-5 and abs(float(vn @ lin.bias.numpy())) < 1e-5
    r = E.ablate_direction(v0, "v1", "blocks.0.fc1", v, "post")       # next = blocks.0.fc2 (act space)
    assert np.abs(E.linear(r.version.net, "blocks.0.fc2").weight.numpy() @ vn).max() < 1e-5
    r = E.ablate_direction(v0, "v1", "blocks.2.fc2", v, "post")       # next = out
    assert np.abs(E.linear(r.version.net, "out").weight.numpy() @ vn).max() < 1e-5
    r = E.ablate_direction(v0, "v1", "blocks.0.fc2", v, "post")       # next = blocks.1.fc1[:, :128]
    W = E.linear(r.version.net, "blocks.1.fc1").weight.numpy()
    assert np.abs(W[:, :128] @ vn).max() < 1e-5
    assert np.array_equal(W[:, 128:], E.linear(v0.net, "blocks.1.fc1").weight.numpy()[:, 128:])
    assert not E.ablate_direction(v0, "v1", "out", np.ones(2), "post").record.ok
    assert not E.ablate_direction(v0, "v1", "out", np.zeros(2), "pre").record.ok
    assert not E.ablate_direction(v0, "v1", "out", np.ones(2), "middle").record.ok


def test_set_units(v0):
    idx = [3, 17]
    w_in = R.standard_normal((2, 256)).astype(np.float32)
    b = np.array([0.5, -0.5], dtype=np.float32)
    w_out = R.standard_normal((128, 2)).astype(np.float32)
    r = E.set_units(v0, "v1", "blocks.1.fc1", idx, w_in, b, w_out)
    assert r.record.ok
    f1, f2 = E.linear(r.version.net, "blocks.1.fc1"), E.linear(r.version.net, "blocks.1.fc2")
    assert np.array_equal(f1.weight.numpy()[idx], w_in) and np.array_equal(f1.bias.numpy()[idx], b)
    assert np.array_equal(f2.weight.numpy()[:, idx], w_out)
    other = [i for i in range(128) if i not in idx]
    assert np.array_equal(f1.weight.numpy()[other], E.linear(v0.net, "blocks.1.fc1").weight.numpy()[other])
    r = E.set_units(v0, "v1", "blocks.1.fc1", [5], None, None, np.zeros((128, 1)))      # silence one unit
    assert r.record.ok and np.all(E.linear(r.version.net, "blocks.1.fc2").weight.numpy()[:, 5] == 0)
    assert not E.set_units(v0, "v1", "blocks.1.fc2", idx, w_in, b, w_out).record.ok
    assert not E.set_units(v0, "v1", "blocks.1.fc1", [3, 3], None, b, None).record.ok
    assert not E.set_units(v0, "v1", "blocks.1.fc1", [200], None, b[:1], None).record.ok
    assert not E.set_units(v0, "v1", "blocks.1.fc1", idx, None, None, None).record.ok


# ------------------------------------------------------------------ patches and rebuild
def test_patch_stack_rebuild_bit_identical(cell, v0, tmp_path):
    from toyrepair import train as T
    run_dir = str(tmp_path)
    v = v0
    steps = [
        lambda p: graft(p, "v1", d=det("blocks.0.act", 128, thr=0.3), scale=0.3),
        lambda p: E.rank1(p, "v2", "blocks.2.fc2", R.standard_normal(128), R.standard_normal(128), 0.01),
        lambda p: E.ablate_direction(p, "v3", "blocks.1.fc1", R.standard_normal(128), "pre"),
        lambda p: E.ablate_direction(p, "v4", "inp", R.standard_normal(128), "post"),
        lambda p: E.set_units(p, "v5", "blocks.0.fc1", [1, 2, 3], None, np.zeros(3), None),
        lambda p: graft(p, "v6", layer="out", d=det("inputs", 4, k=(2, 9)), scale=0.3),
        lambda p: T.targeted_train(cell, p, "v7", T.TrainSpec(params=("out", "grafts"), objective="bc", steps=30),
                                   T.rescue_rows(cell, cell.neg.uid[:2], cell.neg.a[:2] * 0.5, support="nhood")),
    ]
    for f in steps:
        r = f(v)
        assert r.record.ok, r.record.error
        P.save_patch(run_dir, r.patch)
        assert E.unit_tests(r.version, v).passed
        v = r.version
    assert v.patches == ("p_v1", "p_v2", "p_v3", "p_v4", "p_v5", "p_v6", "p_v7")
    rb = P.rebuild(cell, run_dir, list(v.patches), v.id)
    sa, sb = v.net.state_dict(), rb.net.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert P.state_sha256(rb) == P.state_sha256(v)
    xi, c = torch.randn(4096, 2, generator=torch.Generator().manual_seed(5)), torch.rand(4096)
    assert torch.equal(E.decode_actions(rb.net, xi, c), E.decode_actions(v.net, xi, c))
    assert not torch.equal(E.decode_actions(rb.net, xi, c), E.decode_actions(v0.net, xi, c))
    # a tampered patch is detected
    p = P.load_patch(run_dir, "p_v2")
    p.arrays["u"] = p.arrays["u"] * 2
    P.save_patch(run_dir, p)
    with pytest.raises(ValueError):
        P.rebuild(cell, run_dir, list(v.patches), v.id)


@pytest.mark.skipif(not have_uncond, reason="no runs/base/s0")
def test_uncond_base_k2_edits():
    cell = load_cell("uncond", "base_k2", 0, 4)
    v0 = E.base_version(cell)
    x, t, c = E.reference_states(v0.net, 2000, uncond=True)
    assert torch.equal(E.run_forward(v0.net, x, t, c)[0], v0.net(x, t, c))
    assert E.source_dims(v0.net)["cond"] == 64 and E.source_dims(v0.net)["blocks.0.fc1.in"] == 192
    r = graft(v0, "v1", layer="blocks.0.fc1", d=det("cond", 64, thr=0.5), scale=0.3)
    assert r.record.ok and E.unit_tests(r.version, v0).passed
    r2 = E.rank1(r.version, "v2", "out", np.ones(2), np.ones(128), 0.01)
    assert r2.record.ok and E.unit_tests(r2.version, r.version).passed
    xi = torch.randn(1000, 2)
    assert torch.equal(cell.ref.sample(xi, None), E.decode_actions(v0.net, xi, torch.zeros(1000)))
