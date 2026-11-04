import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import toyrepair
from toyrepair import flowstate as F
from toyrepair import policy
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))


class _Zero(nn.Module):
    def forward(self, x, t, c):
        return torch.zeros_like(x)


def _zero_version():
    return policy.Version(id="z", parent=None, patches=(), net=_Zero())


def test_wilson():
    lo, hi = F.wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert F.wilson(0.0, 10)[0] == 0.0 and F.wilson(1.0, 10)[1] == 1.0
    assert F.wilson(0.3, 0) == (0.0, 1.0)


def test_k_commit():
    s = [0, 2, 4, 6, 8]
    assert F.k_commit(s, [0.9, 0.8, 0.5, 0.1, 0.05]) == 6
    assert F.k_commit(s, [0.1, 0.1, 0.1, 0.1, 0.1]) == 0
    assert F.k_commit(s, [0.9, 0.1, 0.5, 0.1, 0.05]) == 6      # non-monotone: later rise counts
    assert F.k_commit(s, [0.9, 0.9, 0.9, 0.9, 0.9]) == 8       # none qualifies: last step
    assert F.k_commit(s, [0.9, 0.9, 0.9, 0.9, 0.2]) == 8


def test_push_and_hop_with_zero_field():
    v = _zero_version()
    xi = torch.randn(7, 2)
    c = torch.zeros(7)
    p = F.push((3.0, 4.0), 1.5, k=5, h=3)
    assert p.u == pytest.approx((0.6, 0.8))
    out = F.restore_and_decode(v, F.FlowState(x=xi, k=0, c=c), p)
    assert torch.allclose(out, xi + 1.5 * torch.tensor([0.6, 0.8]), atol=1e-6)
    # restoring after the push window: no effect
    out2 = F.restore_and_decode(v, F.FlowState(x=xi, k=9, c=c), p)
    assert torch.equal(out2, xi)
    h = F.hop(0.7, (0, -2))
    out3 = F.restore_and_decode(v, F.FlowState(x=xi, k=0, c=c), h)
    assert torch.allclose(out3, xi + torch.tensor([0.0, -0.7]))
    with pytest.raises(ValueError):
        F.restore_and_decode(v, F.FlowState(x=xi, k=3, c=c), h)


def test_renoise_zero_field():
    v = _zero_version()
    x = torch.randn(5, 2)
    st = F.FlowState(x=x, k=10, c=torch.zeros(5))
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    eps = torch.randn(5, 2, generator=g2)
    r = F.renoise(v, st, g1)
    assert torch.allclose(r.x, 0.5 * eps + 0.5 * x) and r.k == 10
    r0 = F.renoise(v, F.FlowState(x=x, k=0, c=torch.zeros(5)), torch.Generator().manual_seed(1))
    assert torch.equal(r0.x, eps)


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0")
    return load_cell("cond", "k4", 0, 1)


def test_branch_smoke(cell):
    from toyrepair import miner
    v0 = policy.base_version(cell)
    cl = miner.Cluster(id=0, members=np.arange(len(cell.neg)), uids=cell.neg.uid.copy())
    bt = F.branch(cell, v0, cl, steps=(0, 4, 8, 12), M=8)
    assert [r.k for r in bt.rows] == [0, 4, 8, 12]
    assert bt.failure_class in ("R", "U") and bt.k_commit in (0, 4, 8, 12)
    assert bt.n_decodes > 0 and len(bt.render().splitlines()) <= 43
    assert all(0 <= r.p <= 1 for r in bt.rows)
    # renoise from step 0 is fresh noise: escapes the basin often
    assert bt.rows[0].p > 0.2
    # determinism
    bt2 = F.branch(cell, v0, cl, steps=(0, 4, 8, 12), M=8)
    assert [r.p for r in bt2.rows] == [r.p for r in bt.rows]


def test_hop_search_matches_toypref(cell):
    """Under pi_ref, the hop search exit equals toypref's plain hop target."""
    from toypref.cond import targets as TG
    v0 = policy.base_version(cell)
    ex = F.hop_search(cell, v0, np.array([0]))[0]
    n = cell.neg
    inf = {}
    TG.target_one(cell.ref, "hop", float(n.c[0]), n.xi[0], n.a[0], TG.row_gen(cell.seed, n.uid[0], 1), info=inf)
    assert ex is not None and ex[0] == pytest.approx(inf["tau"])


def test_inspect(cell):
    v0 = policy.base_version(cell)
    uid = int(cell.neg.uid[0])
    tab = F.inspect(cell, v0, uid, units=[("blocks.2.act", 5), ("out_norm", 0)])
    assert len(tab.rows) == 21
    assert tab.rows[-1]["x"] == pytest.approx(policy.decode(v0, cell.neg.xi[:1], cell.neg.c[:1]).action[0].tolist())
    assert tab.rows[-1]["in_bad"]
    txt = tab.render()
    assert "blocks.2.act[5]" in txt and len(txt.splitlines()) == 23
    assert len(F.inspect(cell, v0, uid, every=5).rows) == 5
