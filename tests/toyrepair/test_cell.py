import os

import numpy as np
import pytest
import torch

import toyrepair
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))
have_uncond = os.path.exists(os.path.join(RUNS, "base", "s0", "policy.pt"))


def test_runs_env_set():
    assert os.environ["TOYPREF_RUNS"]
    import toypref.config as C
    assert C.RUNS == os.environ["TOYPREF_RUNS"]


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0 (run scripts/cloud_setup.sh)")
    return load_cell("cond", "k4", 0, 1)


def test_cond_cell_matches_toypref(cell):
    from toypref.cond import experiment as E
    from toypref.cond.task import CondFlow, frozen
    sd = torch.load(os.path.join(RUNS, "cond_base", "s0", "policy.pt"), map_location="cpu", weights_only=False)
    ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); ref = frozen(ref)
    pool = E.make_pool(ref, 0, 200_000)
    ed = E.editset(pool, 0, 4, 1, "random")
    assert np.array_equal(cell.neg.uid, ed.uid)
    assert torch.equal(cell.neg.c, ed.c) and torch.equal(cell.neg.xi, ed.xi) and torch.equal(cell.neg.a, ed.a)
    assert len(cell.neg) == 4 and cell.index == 200 and cell.key == "cond_k4_s0_n1"
    assert "_oracle" not in repr(cell) and "retain" not in repr(cell)
    from toyrepair import oracle
    assert oracle.forget_modes(cell) == ed.forget
    # actions are pi_ref decodes of the logged noises (decoded inside a 200k batch; a small
    # batch differs in the last bits: CPU matmul results depend on batch size)
    assert torch.allclose(cell.ref.sample(cell.neg.xi, cell.neg.c), cell.neg.a, atol=1e-4)


def test_cond_basins_and_vote(cell):
    from toypref.cond import targets as TG
    n = cell.neg
    bank = TG.NegBank(cell.ref, 0, n.uid, n.c, n.xi, n.a)
    assert cell.vote.theta == bank.theta
    for i in range(len(n)):
        assert torch.equal(cell.basins[i].pts, bank.basins[i].pts)
        assert cell.basins[i].bad == bank.basins[i].bad
        assert bool(cell.in_bad_basin(i, n.a[i:i + 1])[0])
    order = cell.public_order
    assert np.all(np.diff(n.uid[order]) > 0)


@pytest.mark.skipif(not have_uncond, reason="no runs/base/s0")
def test_uncond_cell():
    c = load_cell("uncond", "base_k2", 0, 1)
    assert len(c.neg) == 2 and c.index == 1100
    assert torch.allclose(c.ref.sample(c.neg.xi, c.neg.c), c.neg.a, atol=1e-4)
    assert float(c.neg.c.abs().sum()) == 0.0
    assert bool(c.in_bad_basin(0, c.neg.a[:1])[0])
