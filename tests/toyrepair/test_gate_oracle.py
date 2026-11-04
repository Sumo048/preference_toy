"""W1-A2: oracle, gate, final evaluation (docs/PLAN.md 4.8, 6.2, 10)."""

import copy
import os

import numpy as np
import pytest
import torch

import toyrepair  # noqa: F401  (sets TOYPREF_RUNS)
from toyrepair import gate, oracle, pools
from toyrepair.oracle import _adapt, metrics

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))
have_uncond = os.path.exists(os.path.join(RUNS, "base", "s0", "policy.pt"))
SMALL = gate.GateConfig(n_g=4096, n_boot=500)


# ------------------------------------------------------------------ pure tests (no cell)
def test_masses_counts():
    lab = np.array([0, 0, 1, -1, 7, -1])
    m = oracle.masses(lab)
    assert m.shape == (9,) and abs(m.sum() - 1) < 1e-12
    assert m[0] == 2 / 6 and m[8] == 2 / 6 and m[7] == 1 / 6
    assert oracle.counts(lab).tolist() == [2, 1, 0, 0, 0, 0, 0, 1, 2]


def test_scores_definitions():
    R, F = [0, 1, 2, 3], [4, 5, 6, 7]
    ideal = metrics.ideal(R)
    s = metrics.scores_from_masses(ideal, ideal, F, R)
    assert s.D == 0 and s.collateral == 0 and s.retain_tv == 0
    ref = np.array([.125] * 8 + [0.0])
    s = metrics.scores_from_masses(ref, ref, F, R)
    assert abs(s.D - 0.5) < 1e-12 and s.removed == 0
    # move half of retained mode 0's mass to OFF: retain_tv 0, PLAN collateral > 0
    m = ref.copy(); m[0] -= .0625; m[8] += .0625
    s = metrics.scores_from_masses(m, ref, F, R)
    assert s.collateral > 0
    m2 = ref.copy(); m2[0] += .0625; m2[1] -= .0625
    s2 = metrics.scores_from_masses(m2, ref, F, R)
    assert abs(s2.retain_tv - 0.125) < 1e-12


def _labels(n, rng, probs):
    return rng.choice(np.arange(-1, 8), size=n, p=probs)


def test_joint_bootstrap_matches_index_bootstrap():
    rng = np.random.default_rng(0)
    n = 3000
    p = np.array([0.05] + [0.95 / 8] * 8)
    lb = _labels(n, rng, p)
    lc = np.where(rng.random(n) < 0.9, lb, _labels(n, rng, p))
    ln = np.where(np.isin(lc, [2, 3]) & (rng.random(n) < 0.7), 0, lc)
    T, F, R = [2, 3], [2, 3, 4, 7], [0, 1, 5, 6]
    counts = gate._joint_counts(lb, lc, ln)
    joint = gate._stats(gate._joint_bootstrap(counts, 2000, np.random.default_rng(1)), T, F, R)["target_fall"]
    B = np.stack([gate._bucket(x) for x in (lb, lc, ln)])

    def stat(idx):
        M = np.stack([np.bincount(B[v, idx], minlength=9) / n for v in range(3)])
        return float(gate._stats(M, T, F, R)["target_fall"])
    ind = gate.paired_bootstrap(stat, n, 2000, np.random.default_rng(2))
    point = float(gate._stats(gate._point_masses(counts), T, F, R)["target_fall"])
    assert abs(stat(np.arange(n)) - point) < 1e-12
    assert abs(joint.mean() - ind.mean()) < 0.003
    assert abs(joint.std() / ind.std() - 1) < 0.1


def _tests(lb, lc, ln, T, F, R, scope="cluster", cfg=SMALL):
    return gate.tests_from_labels(lb, lc, ln, T, F, R, scope, cfg, np.random.SeedSequence([9]))


def test_gate_tests_synthetic():
    rng = np.random.default_rng(0)
    n = 16384
    F, R = [2, 3, 4, 7], [0, 1, 5, 6]
    lb = rng.choice(np.arange(8), size=n)
    lb[rng.random(n) < 0.03] = -1
    lc = lb.copy()
    T = [2]
    # good repair: target mode 2 -> spread over retained modes
    ln = lc.copy()
    hit = ln == 2
    ln[hit] = rng.choice(R, size=hit.sum())
    passed, pt, bounds, _ = _tests(lb, lc, ln, T, F, R)
    assert passed == {"target": True, "progress": True, "drift": True, "other_forgotten": True, "between": True}
    assert pt["target_fall_frac"] == 1.0 and bounds["target_fall_lo"] > 0
    # repair that sends the target mode between modes: test 5 fails
    ln2 = lc.copy(); ln2[hit] = -1
    passed, *_ = _tests(lb, lc, ln2, T, F, R)
    assert passed["target"] and not passed["between"]
    # repair that sends the target mode to another forgotten mode: test 4 fails; global scope n/a
    ln3 = lc.copy(); ln3[hit] = 3
    passed, *_ = _tests(lb, lc, ln3, T, F, R)
    assert passed["target"] and not passed["other_forgotten"] and not passed["progress"]
    passed, *_ = _tests(lb, lc, ln3, T, F, R, scope="global")
    assert passed["other_forgotten"]
    # a half-hearted repair: only 30% of the target removed -> test 1 fails
    ln4 = lc.copy(); part = hit & (rng.random(n) < 0.3); ln4[part] = 0
    passed, pt, *_ = _tests(lb, lc, ln4, T, F, R)
    assert not passed["target"] and 0.2 < pt["target_fall_frac"] < 0.4
    # drift: shuffle retained modes heavily -> test 3 fails
    ln5 = ln.copy(); ret = np.isin(ln5, R) & (rng.random(n) < 0.5); ln5[ret] = 0
    passed, *_ = _tests(lb, lc, ln5, T, F, R)
    assert not passed["drift"]
    # no-op: target and progress fail (upper bound of 0 is not < 0), the rest passes
    passed, pt, bounds, _ = _tests(lb, lc, lc, T, F, R)
    assert not passed["target"] and not passed["progress"] and passed["drift"] and passed["between"]
    assert bounds["dD_ub"] == 0


def test_stub_unit_tests_catch_nan():
    from toyrepair.policy import Version
    from toypref.cond.task import CondVelocityNet
    net = CondVelocityNet()
    v = Version(id="a", parent=None, patches=(), net=net)
    assert _adapt.stub_unit_tests(v, v)["passed"]
    bad = copy.deepcopy(net)
    with torch.no_grad():
        bad.blocks[0].fc1.weight[0, 0] = float("nan")
    r = _adapt.stub_unit_tests(Version(id="b", parent="a", patches=(), net=bad), v)
    assert not r["passed"] and not r["checks"]["finite"]


# ------------------------------------------------------------------ cell tests
@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0 (run scripts/cloud_setup.sh)")
    torch.set_num_threads(1)
    from toyrepair.cell import load_cell
    return load_cell("cond", "k4", 0, 4)


@pytest.fixture(scope="module")
def base(cell):
    return _adapt.base_version(cell)


def test_label_matches_toypref(cell):
    from toypref.cond.task import CondMixture
    c, xi = pools.draw(cell, pools.GATE, 2000, index=5)
    a = cell.ref.sample(xi, c)
    assert np.array_equal(oracle.label(cell, a, c), CondMixture().assign(a, c))
    a2 = a.clone(); a2[0] = float("nan")
    assert oracle.label(cell, a2, c)[0] == oracle.OFF


def test_target_modes_and_purity(cell):
    modes = cell._oracle.neg_modes
    uids = cell.neg.uid
    for m in oracle.forget_modes(cell):
        sel = uids[modes == m]
        assert oracle.target_modes(cell, sel) == [m] and oracle.purity(cell, sel) == 1.0
    mixed = list(uids[modes == modes[0]][:2]) + list(uids[modes != modes[0]][:1])
    assert oracle.target_modes(cell, mixed) == [int(modes[0])]
    assert abs(oracle.purity(cell, mixed) - 2 / 3) < 1e-12
    tie = [uids[modes == modes[0]][0], uids[modes != modes[0]][0]]
    assert len(oracle.target_modes(cell, tie)) == 2
    with pytest.raises(KeyError):
        oracle.target_modes(cell, [-5])


def test_decode_adapter_bit_identical(cell, base):
    c, xi = pools.draw(cell, pools.GATE, 3000, index=7)
    assert torch.equal(_adapt.actions(base, xi, c), cell.ref.sample(xi, c))


def test_toypref_compute_b0(cell, base):
    tp = metrics.toypref_compute(cell, base)
    assert abs(tp["tv_to_ideal"] - 0.51595) < 1e-4      # W0 SURVEY: B0 D 0.516 on this cell
    assert tp["retain_tv"] == 0 and tp["neg_mass_ratio"] == 1.0


def test_eval_final_b0(cell, base):
    from toyrepair import eval_final
    sel = cell.neg.uid[cell._oracle.neg_modes == oracle.forget_modes(cell)[0]]
    r = eval_final.evaluate(cell, base, "B0", target_clusters=[list(sel)])
    assert abs(r.D - 0.51595) < 1e-4 and r.removed == 0 and r.collateral == 0 and r.collateral_plan == 0
    assert r.target_removed == 0 and r.target_removed_toypref == 0
    assert r.final_pool["collateral"] == 0 and abs(r.final_pool["D"] - r.D) < 0.02


def test_gate_noop_and_repro(cell, base):
    res = gate.noop_check(cell, base, list(cell.neg.uid[:4]), cfg=SMALL)
    assert res["decodes_identical"] and res["gate_repro"] and res["max_abs_diff"] == 0
    p = res["noop_passed"]
    assert not p["target"] and not p["progress"] and p["drift"] and p["between"] and p["unit"]


def test_gate_noise_rejected_and_view(cell, base):
    noisy = gate.noisy_version(base, 0.5)
    r = gate.gate_check(cell, base, base, noisy, list(cell.neg.uid[:4]), check_index=3, cfg=SMALL)
    assert not r.accepted and not r.passed["progress"]
    v = r.surgeon_view()
    assert set(v) == {"accepted", "passed", "target_fall", "target_fall_ci"}
    assert r.numbers["decodes"] == 2 * SMALL.n_g          # pi_ref == current: decoded once
    d = r.ledger_data()
    assert d["surgeon_view"] == v and "mass_new" in d["numbers"]
    # A3-nogate: accepted, every test still measured
    r2 = gate.gate_check(cell, base, base, noisy, list(cell.neg.uid[:4]), check_index=3,
                         cfg=gate.GateConfig(n_g=4096, n_boot=500, accept_all=True))
    assert r2.accepted and r2.passed == r.passed and r2.numbers["dD_ub"] == r.numbers["dD_ub"]


def test_gate_detects_repair_global(cell, base):
    """current = noisy, new = pi_ref: a true improvement; and global scope marks test 4 n/a."""
    noisy = gate.noisy_version(base, 0.3)
    groups = [list(cell.neg.uid[cell._oracle.neg_modes == m]) for m in oracle.forget_modes(cell)]
    r = gate.gate_check(cell, base, noisy, base, groups, check_index=4, scope="global", cfg=SMALL)
    assert r.passed["progress"] and r.passed["other_forgotten"] and r.numbers["other_forgotten_na"]
    assert r.numbers["target_modes"] == oracle.forget_modes(cell)
    assert r.numbers["decodes"] == 3 * SMALL.n_g


@pytest.mark.skipif(not have_uncond, reason="no runs/base/s0")
def test_uncond_smoke():
    from toyrepair.cell import load_cell
    cu = load_cell("uncond", "base_k2", 0, 4)
    b = _adapt.base_version(cu)
    tp = metrics.toypref_compute(cu, b)
    assert tp["retain_tv"] == 0 and 0 < tp["tv_to_ideal"] < 1
    r = gate.gate_check(cu, b, b, b, list(cu.neg.uid[:2]), check_index=0, cfg=SMALL)
    assert not r.passed["progress"] and r.passed["drift"]
