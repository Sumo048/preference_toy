"""W1-B: targeted_train reproduces toypref's losses bit for bit (PLAN 6.7, SESSION_PROMPTS W1-B 4).

The full check (5,000 steps of the frozen recipe on cond k4 seed 0 npm 4, against
``scripts/cond_cell.py`` and toypref ``train_arm``) takes ~5 CPU minutes; set
TOYREPAIR_SKIP_SLOW=1 to skip it. The short checks run every toypref loss family for a few
hundred steps against ``train_arm`` with the same step count.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import toyrepair
from toyrepair import edits as Ed
from toyrepair import train as T
from toyrepair.cell import load_cell

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))
slow = pytest.mark.skipif(os.environ.get("TOYREPAIR_SKIP_SLOW") == "1", reason="TOYREPAIR_SKIP_SLOW=1")


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0 (run scripts/cloud_setup.sh)")
    return load_cell("cond", "k4", 0, 4)


def _toypref(cell, arm, steps):
    from toypref.cond import experiment as E
    ctx = cell.toypref_ctx(steps=steps)
    pol, _ = E.train_arm(E.ARMS[arm], ctx)
    return pol


def _same(pol, version):
    a, b = pol.net.state_dict(), version.net.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


SHORT = [
    ("R_hopB_bsup_neg_AMdist", "bc", "dist", dict(rule="hopB", support="bsup")),
    ("R_hop_bsup_neg", "bc", "none", dict(rule="hop", support="bsup")),
    ("R_basin_nhood_neg", "bc", "none", dict(rule="basin", support="nhood")),
    ("R_ours_neg", "bc", "none", dict(rule="ours", support="log")),
    ("R_hopB_bsup_neg_AMmap", "bc", "map", dict(rule="hopB", support="bsup")),
    ("R_hopB_bsup_neg_AL", "bc", "local", dict(rule="hopB", support="bsup")),
    ("R_hopB_bsup_mine_neg", "bc", "none", dict(rule="hopB", support="bsup", mine=True)),
    ("NPO_b10_neg_AMdist", "npo", "dist", None),
    ("KTO_b10_neg_AMmap", "kto", "map", None),
    ("NPO_b10_neg_AL", "npo", "local", None),
]


@pytest.mark.parametrize("arm,obj,anchor,rows", SHORT, ids=[s[0] for s in SHORT])
def test_short_runs_match_toypref_train_arm(cell, arm, obj, anchor, rows):
    steps = 150
    v0 = Ed.base_version(cell)
    data = T.toypref_rows(cell, cell.neg.uid, **rows) if rows else T.negatives(cell, cell.neg.uid)
    r = T.targeted_train(cell, v0, "v1", T.TrainSpec(params=("all",), objective=obj, steps=steps, anchor=anchor), data)
    assert r.record.ok, r.record.error
    assert _same(_toypref(cell, arm, steps), r.version), arm


def test_param_subset_and_rejections(cell):
    v0 = Ed.base_version(cell)
    data = T.rescue_rows(cell, cell.neg.uid[:4], cell.neg.a[:4] * 0.8, support="bsup")
    assert data.c.shape[0] >= 4 and data.decodes > 0
    r = T.targeted_train(cell, v0, "v1", T.TrainSpec(params=("blocks.2", "out.bias"), objective="bc", steps=40), data)
    assert r.record.ok and r.record.decodes > 0
    changed = {k for k in r.patch.arrays}
    assert changed and all(k.startswith("blocks.2.") or k == "out.bias" for k in changed)
    a, b = v0.net.state_dict(), r.version.net.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if k not in changed)
    for spec in (T.TrainSpec(params=("all",), objective="dpo", steps=10),
                 T.TrainSpec(params=("all",), objective="bc", steps=6000),
                 T.TrainSpec(params=("nope",), objective="bc", steps=10),
                 T.TrainSpec(params=("all",), objective="bc", steps=10, anchor="oracle")):
        assert not T.targeted_train(cell, v0, "v1", spec, data).record.ok
    with pytest.raises(ValueError):
        T.toypref_rows(cell, cell.neg.uid, rule="own")
    with pytest.raises(ValueError):
        T.rescue_rows(cell, np.array([123456789]), torch.zeros(1, 2))


def test_trains_grafts_only(cell):
    v0 = Ed.base_version(cell)
    d = Ed.Detector("t", (1.0,), 0.0, -1.0)
    g = Ed.graft_gated_path(v0, "v1", "out", d, 4, np.random.default_rng(0).standard_normal((4, 128)) * 0.1,
                            np.zeros((2, 4)))
    data = T.rescue_rows(cell, cell.neg.uid[:4], cell.neg.a[:4] * 0.8, support="log")
    r = T.targeted_train(cell, g.version, "v2", T.TrainSpec(params=("grafts",), objective="bc", steps=30, anchor="none"), data)
    assert r.record.ok and set(r.patch.arrays) <= {"grafts.0.w_in", "grafts.0.b_in", "grafts.0.w_out"}
    assert Ed.unit_tests(r.version, g.version).passed


@slow
def test_frozen_recipe_matches_cond_cell(cell, tmp_path):
    """targeted_train as R_hopB_bsup_neg_AMdist on k4 seed 0 npm 4 == scripts/cond_cell.py."""
    from toypref.cond import experiment as E
    arm = "R_hopB_bsup_neg_AMdist"
    v0 = Ed.base_version(cell)
    data = T.toypref_rows(cell, cell.neg.uid, rule="hopB", support="bsup")
    r = T.targeted_train(cell, v0, "v1", T.TrainSpec(params=("all",), objective="bc", steps=5000, anchor="dist"), data)
    assert r.record.ok
    # weights: bit for bit against toypref train_arm
    assert _same(_toypref(cell, arm, 5000), r.version)
    # metrics: against the JSON that scripts/cond_cell.py writes
    repo = toyrepair.REPO
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, os.path.join(repo, "scripts", "cond_cell.py"), "--seed", "0", "--n_forget", "4",
                    "--n_neg", "4", "--arms", arm, "--out", str(tmp_path)], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    ref = json.load(open(tmp_path / "s0_k4_n4" / f"{arm}.json"))

    class Pol:
        def __init__(self, net):
            self.net = net

        def sample(self, xi, c):
            return Ed.decode_actions(self.net, xi, c)
    bank = E.Bank(cell.ref, 0, cell._oracle.toypref_edit)      # test-only oracle use
    met = E.compute(bank, Pol(r.version.net))
    for k in ("tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off"):
        assert met[k] == ref[k], (k, met[k], ref[k])
    assert abs(met["tv_to_ideal"] - 0.2588) < 1e-3
