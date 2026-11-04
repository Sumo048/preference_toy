"""W2a merge checks: policy capture on grafted nets (W1-A1 x W1-B), the CLI's cell cache, and
the R0 unit-test delegation (W1-C item 8)."""
import os
import types

import numpy as np
import pytest
import torch

import toyrepair  # noqa: F401  (sets TOYPREF_RUNS)
from toyrepair import edits as E
from toyrepair import policy
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_uncond = os.path.exists(os.path.join(RUNS, "base", "s0", "policy.pt"))
pytestmark = pytest.mark.skipif(not have_uncond, reason="no runs/base/s0")


@pytest.fixture(scope="module")
def cell():
    return load_cell("uncond", "base_k2", 0, 4)


def _graft(v, layer="blocks.1.fc2", n=4):
    R = np.random.default_rng(1)
    d = E.source_dims(v.net)
    det = E.Detector("t", (1.0,), 0.0, 0.5, None)       # fires where t > 0.5 (steps 0..9)
    src = d[E.DEFAULT_SRC[layer]]
    out = E.linear(v.net, layer).out_features
    res = E.graft_gated_path(v, "v1", layer, det, n, R.standard_normal((n, src)) * 0.1,
                             R.standard_normal((out, n)) * 0.1)
    assert res.record.ok, res.record.error
    return res.version


def test_capture_on_grafted_net_matches_run_forward(cell):
    v0 = policy.base_version(cell)
    v1 = _graft(v0)
    xi = torch.randn(64, 2, generator=torch.Generator().manual_seed(0))
    c = torch.zeros(64)
    names = ("blocks.1.fc2", "out_act", "cond")
    d = policy.decode(v1, xi, c, save_states=True, capture=names, count=False)
    assert torch.equal(d.action, policy.decode(v1, xi, c, count=False).action)
    k = 3
    x, t = d.states[k], torch.full((64,), 1.0 - k / 20)
    _, ref = E.run_forward(v1.net, x, t, c, capture=names)
    for n in names:
        assert torch.equal(d.acts[n][k], ref[n]), n
    # the graft fires at step 3, so the captured fc2 differs from pi_ref's
    d0 = policy.decode(v0, xi, c, save_states=True, capture=names, count=False)
    assert not torch.equal(d0.acts["blocks.1.fc2"][k], d.acts["blocks.1.fc2"][k])


def test_layers_lists_grafts(cell):
    v1 = _graft(policy.base_version(cell))
    L = policy.layers(v1)
    g = [x for x in L if x.kind == "graft"]
    assert len(g) == 1 and g[0].out_dim == 4 and g[0].name.startswith("blocks.1.fc2.graft")
    assert {x.name for x in L if x.kind == "linear"} >= {"inp", "out", "blocks.0.fc1"}


def test_cli_cell_cache_roundtrip(cell, tmp_path):
    from toyrepair.cli import cached_cell
    run = types.SimpleNamespace(dir=str(tmp_path), cell_args=("uncond", "base_k2", 0, 4))
    a = cached_cell(run, cache_dir=str(tmp_path))
    assert os.path.exists(tmp_path / "cell_cache.pt")
    b = cached_cell(run, cache_dir=str(tmp_path))
    assert np.array_equal(a.neg.uid, b.neg.uid)
    xi = torch.randn(256, 2, generator=torch.Generator().manual_seed(2))
    c = torch.zeros(256)
    ya = policy.decode(policy.base_version(a), xi, c, count=False).action
    yb = policy.decode(policy.base_version(b), xi, c, count=False).action
    assert torch.equal(ya, yb)
    for i in range(len(a.neg)):
        assert np.array_equal(a.in_bad_basin(i, ya), b.in_bad_basin(i, ya))


def test_unit_tests_delegate_to_controller():
    calls = []

    class Ctl(torch.nn.Module):
        def controller_unit_tests(self, current):
            calls.append(current)
            return {"passed": True, "checks": {"controller": True}, "detail": {}}

    new = types.SimpleNamespace(net=Ctl())
    r = E.unit_tests(new, "cur")
    assert r.passed and r.checks == {"controller": True} and calls == ["cur"]


def test_static_prompt_lists_only_the_arms_edit_calls():
    from toyrepair import dossier
    a1, a2, a3 = (dossier.static_part(a) for a in ("A1", "A2", "A3"))
    assert "**targeted_train**" in a1 and "**graft_gated_path**" not in a1 and "**rank1**" not in a1
    assert "**graft_gated_path**" in a2 and "**targeted_train**" not in a2
    assert "**graft_gated_path**" in a3 and "**targeted_train**" in a3
    assert "Your edit calls: `targeted_train`;" in a1
