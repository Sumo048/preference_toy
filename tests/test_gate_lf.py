"""Label-free gate (session A3LF): thresholds match the oracle gate, bucket statistics, and no oracle
import on the A3-LF loop path (checked in a fresh interpreter on development seed 0)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from toyrepair import gate_lf


def test_thresholds_match_oracle_gate():
    from toyrepair import gate
    for k in ("N_G", "N_BOOT", "TARGET_FALL", "DRIFT_MAX", "OTHER_RISE_MAX", "BETWEEN_RISE_MAX", "ALPHA",
              "TEST_NAMES"):
        assert getattr(gate_lf, k) == getattr(gate, k), k
    assert gate_lf.GateConfig() == gate_lf.GateConfig(**{f: getattr(gate.GateConfig(), f)
                                                        for f in gate_lf.GateConfig.__dataclass_fields__})


def test_buckets_and_stats():
    bk = gate_lf.Buckets(target=(3,), other=(5, 6), good=(0, 1, 2))
    assert bk.nb == 6
    np.testing.assert_array_equal(bk.of(np.array([3, 5, 6, 0, 1, 2, -1, 4])), [0, 1, 1, 2, 3, 4, 5, 5])
    # ideal mass: uniform on the good tracks -> D_lf 0
    m = np.array([0, 0, 1 / 3, 1 / 3, 1 / 3, 0])
    assert gate_lf.d_lf(m, 3) == pytest.approx(0.0)
    assert gate_lf.d_lf(np.array([1.0, 0, 0, 0, 0, 0]), 3) == pytest.approx(1.0)
    assert gate_lf.collateral_lf(m, m) == pytest.approx(0.0)


def _batch(rng, n, p):
    return rng.choice(len(p), size=n, p=p)


def test_tests_accept_a_clean_removal_and_reject_noop():
    rng = np.random.default_rng(0)
    bk = gate_lf.Buckets(target=(9,), other=(8,), good=(0, 1, 2))
    n = 16384
    p_ref = np.array([0.2, 0.1, 0.2, 0.2, 0.2, 0.1])          # T, O, G0, G1, G2, U
    bb = _batch(rng, n, p_ref)
    bc = bb.copy()
    bn = bb.copy()
    hit = bn == 0                                              # every target sample moves to a good track
    bn[hit] = 2 + rng.integers(0, 3, size=int(hit.sum()))
    ss = np.random.SeedSequence([2, 0, 0])
    passed, pt, bounds, _ = gate_lf.tests_from_buckets(bb, bc, bn, bk, gate_lf.GateConfig(), ss)
    assert all(passed.values()), (passed, bounds)
    assert pt["target_fall_frac"] == pytest.approx(1.0)
    passed, *_ = gate_lf.tests_from_buckets(bb, bc, bc.copy(), bk, gate_lf.GateConfig(), ss)
    assert not passed["target"] and not passed["progress"]


def test_row_bootstrap_matches_joint():
    rng = np.random.default_rng(1)
    n, nb = 4000, 5
    bb, bc, bn = (rng.integers(0, nb, n) for _ in range(3))
    a = gate_lf._boot_masses(bb, bc, bn, nb, 400, np.random.default_rng(2))
    old = gate_lf.MAX_JOINT
    try:
        gate_lf.MAX_JOINT = 0
        b = gate_lf._boot_masses(bb, bc, bn, nb, 400, np.random.default_rng(3))
    finally:
        gate_lf.MAX_JOINT = old
    np.testing.assert_allclose(a.mean(0), b.mean(0), atol=0.01)
    np.testing.assert_allclose(a.std(0), b.std(0), atol=0.004)


CHILD = r"""
import json, os, sys, tempfile, types
import torch
torch.set_num_threads(1)
os.environ["TOYPREF_RUNS"] = sys.argv[1]
import toyrepair
from toyrepair import orchestrate as orc
from toyrepair.cli import RealBackend, ARM_COMMANDS
root = tempfile.mkdtemp()
run = orc.init_run("lf_test", "A3-LF", "cond", "k4", 0, 4, phase4_tools=True, runs_root=root)
be = RealBackend()
atlas = orc.get_atlas(run)
cid = atlas.cards[0].cluster
res = be.gate(run, "v0", cid, 0)
from toyrepair import dossier
st = dossier.static_part("A3-LF", True) == dossier.static_part("A3", True)
print(json.dumps({"oracle": [m for m in sys.modules if m.startswith("toyrepair.oracle") or m == "toyrepair.gate"],
                  "gate": type(res).__module__, "accepted": res.accepted, "fall": res.target_fall,
                  "nb": len(res.numbers["mass_new"]), "buckets": res.numbers["buckets"],
                  "same_static": st, "same_cmds": ARM_COMMANDS["A3-LF"] == ARM_COMMANDS["A3"]}))
"""


@pytest.mark.slow
def test_a3lf_loop_never_imports_oracle():
    runs = os.environ.get("TOYPREF_RUNS") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    r = subprocess.run([sys.executable, "-c", CHILD, runs], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["oracle"] == [], d
    assert d["gate"] == "toyrepair.gate_lf"
    assert d["fall"] == 0.0 and not d["accepted"]      # v0 vs v0: a no-op
    assert d["same_static"] and d["same_cmds"]
