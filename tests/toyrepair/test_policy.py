import os

import numpy as np
import pytest
import torch

import toyrepair
from toyrepair import policy
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))
have_uncond = os.path.exists(os.path.join(RUNS, "base", "s0", "policy.pt"))


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0 (run scripts/cloud_setup.sh)")
    return load_cell("cond", "k4", 0, 1)


def _pairs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, generator=g), torch.rand(n, generator=g)


def test_decode_matches_toypref_sampler(cell):
    v0 = policy.base_version(cell)
    xi, c = _pairs(5000)
    assert torch.equal(policy.decode(v0, xi, c).action, cell.ref.sample(xi, c))
    # the same with a small batch split
    assert torch.equal(policy.decode(v0, xi, c, batch=1024).action, cell.ref.sample(xi, c, batch=1024))


def test_capture_is_bit_identical_and_shaped(cell):
    v0 = policy.base_version(cell)
    xi, c = _pairs(300, 1)
    d = policy.decode(v0, xi, c, save_states=True, capture=policy.CAPTURE_LAYERS)
    assert torch.equal(d.action, policy.decode(v0, xi, c).action)
    assert d.states.shape == (21, 300, 2) and d.v.shape == (20, 300, 2)
    assert torch.equal(d.states[20], d.action) and torch.equal(d.states[0], xi)
    sizes = {l.name: l.out_dim for l in policy.layers(v0)}
    for nm in policy.CAPTURE_LAYERS:
        assert d.acts[nm].shape == (20, 300, sizes[nm]), nm
    assert torch.equal(d.acts["out"], d.v)
    # cond capture equals the network's own cond vector
    net = v0.net
    t = torch.full((300,), 1.0 - 3 / 20)
    cond = torch.cat([net.time_mlp(net.time_embed(t)), net.c_mlp(net.c_feat(c))], -1)
    assert torch.equal(d.acts["cond"][3], cond)
    assert torch.equal(d.acts["out_act"][3], torch.nn.functional.silu(d.acts["out_norm"][3]))


def test_velocity_and_counter(cell):
    v0 = policy.base_version(cell)
    xi, c = _pairs(64, 2)
    t = torch.full((64,), 0.5)
    before = policy.counter.total
    v, acts = policy.velocity(v0, xi, t, c, capture=("blocks.1.act",))
    assert torch.equal(v, v0.net(xi, t, c)) and acts["blocks.1.act"].shape == (64, 128)
    assert policy.counter.total - before == pytest.approx(64 / 20)
    policy.decode(v0, xi, c)
    assert policy.counter.total - before == pytest.approx(64 / 20 + 64)


def test_branch_validity_10k(cell):
    """PLAN 5.4: restore at every step k and continue with no perturbation gives the same action
    as the uninterrupted decode, bit for bit, on 10,000 noises (same batch composition)."""
    from toyrepair.flowstate import FlowState, restore_and_decode
    v0 = policy.base_version(cell)
    xi, c = _pairs(10_000, 3)
    d = policy.decode(v0, xi, c, save_states=True)
    for k in range(21):
        a = restore_and_decode(v0, FlowState(x=d.states[k], k=k, c=c))
        assert torch.equal(a, d.action), f"step {k}"


def test_layers_cond(cell):
    L = {l.name: l for l in policy.layers(policy.base_version(cell))}
    assert L["inp"].in_dim == 2 and L["inp"].out_dim == 128
    assert L["blocks.0.fc1"].in_dim == 256 and L["out"].out_dim == 2 and L["cond"].out_dim == 128


@pytest.mark.skipif(not have_uncond, reason="no runs/base/s0")
def test_uncond_decode_and_capture():
    cell = load_cell("uncond", "base_k2", 0, 4)
    v0 = policy.base_version(cell)
    xi, c = _pairs(2000, 4)
    assert torch.equal(policy.decode(v0, xi, c).action, cell.ref.pol.sample(xi))
    d = policy.decode(v0, xi, c, capture=("cond", "blocks.0.fc1", "out_act"))
    assert d.acts["cond"].shape == (20, 2000, 64) and d.acts["blocks.0.fc1"].shape == (20, 2000, 128)
    L = {l.name: l for l in policy.layers(v0)}
    assert L["blocks.0.fc1"].in_dim == 192 and L["cond"].out_dim == 64
