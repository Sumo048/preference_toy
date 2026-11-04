"""W1-C arms on a real cell (needs runs/cond_base/s0): B1a's training equals scripts/cond_cell.py."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

import toyrepair

RUNS = os.environ["TOYPREF_RUNS"]
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt")),
                                reason="no runs/cond_base/s0 (run scripts/cloud_setup.sh)")


def test_toypref_train_matches_cond_cell_ctx():
    """Short (40-step) run: the arm's toypref training equals cond_cell's construction bit for bit."""
    from toypref.cond import experiment as E
    from toypref.cond.task import CondFlow, frozen
    from toyrepair.arms.base import toypref_train
    from toyrepair.cell import load_cell
    cell = load_cell("cond", "k4", 0, 1)
    pol_a, _ = toypref_train(cell, "R_hopB_bsup_neg_AMdist", steps=40)
    # cond_cell.py's own construction
    sd = torch.load(os.path.join(RUNS, "cond_base", "s0", "policy.pt"), map_location="cpu", weights_only=False)
    ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); ref = frozen(ref)
    pool = E.make_pool(ref, 0, 200_000)
    ed = E.editset(pool, 0, 4, 1, "random")
    anchor = E.make_pool(ref, 0, 20_000, purpose=E.P_ANCHOR)
    ctx = E.Ctx(ref=ref, seed=0, ed=ed, pool=pool, anchor=anchor, steps=40)
    pol_b, _ = E.train_arm(E.ARMS["R_hopB_bsup_neg_AMdist"], ctx, probe=E.make_probe(E.Bank(ref, 0, ed)))
    a, b = pol_a.state_dict(), pol_b.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_b0_run_and_version(tmp_path):
    from toyrepair.arms import get
    from toyrepair.cell import load_cell
    cell = load_cell("cond", "k4", 0, 1)
    run = SimpleNamespace(id="b0test", dir=str(tmp_path / "b0"))
    out = get("B0")(cell, run)
    assert out.version == "v0" and out.final.id == "v0"
    xi, c = torch.randn(64, 2), torch.rand(64)
    with torch.no_grad():
        assert torch.equal(out.final.net(xi, torch.full((64,), 0.5), c), cell.ref.net(xi, torch.full((64,), 0.5), c))
