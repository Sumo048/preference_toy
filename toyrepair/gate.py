"""The gate: accept or reject an edit with six fixed tests (docs/PLAN.md 10). Owner: W1-A2.

One of only two modules allowed to import ``toyrepair.oracle``.

Each gate check draws a fresh batch of N_G (c, xi) pairs from the GATE pool
(``toyrepair.pools.draw(cell, pools.GATE, N_G, index=check_index)``), decodes it with pi_ref,
the current and the new version through ``toyrepair.policy.decode``, and labels each action
with ``toyrepair.oracle.label``. Intervals: paired bootstrap over the batch, 2,000 resamples
(numpy Generator seeded from a child of the gate pool stream, so a check is reproducible).

Bootstrap implementation: every statistic is a function of the three label vectors only, so a
row resample is fully described by the counts of the 9 x 9 x 9 joint label categories
(ref, current, new). Resampling n rows with replacement is exactly a multinomial draw of those
counts with the empirical category shares, which is what ``_joint_bootstrap`` does (same
distribution as index resampling, about 1000x faster). ``paired_bootstrap`` is the generic
index-resampling form, used in the tests to check the equivalence.

Target modes: ``oracle.target_modes(cell, uids)`` (cluster scope) or the union over all
first-atlas clusters (global scope). Masses are over the whole batch (all c).

Tests (all must pass):
  1 target:   mass on target modes falls by >= 50% of current (point estimate), and the 95%
              lower bound (2.5th percentile) of the paired fall is > 0
  2 progress: one-sided 95% upper bound of D(new) - D(current) < 0
  3 drift:    one-sided 95% upper bound of Collateral(new, pi_ref) <= 0.10
              (Collateral = PLAN definition, retained modes + OFF; retain_tv is logged too)
  4 other:    one-sided 95% UB of rise in mass on non-target forgotten modes <= 0.01
              (cluster scope only; passed and marked n/a in global scope)
  5 between:  one-sided 95% UB of rise in between-mode mass <= 0.01
  6 unit:     ``toyrepair.edits.unit_tests(new, current)`` (owner W1-B): shapes, no NaN,
              detectors read inputs/activations only, graft identity for each graft
              (local stub with shape and NaN checks until the merge: ``oracle._adapt``)

Power and reproducibility (PLAN 10.3): ``noop_check`` and ``power_check`` below;
``python -m toyrepair.gate power --setting k4 --seed 0 --npm 4``.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from . import oracle, pools
from .oracle import _adapt
from .oracle.metrics import collateral, d_ideal, retain_tv

N_G = 16_384
N_BOOT = 2_000
TARGET_FALL = 0.5
DRIFT_MAX = 0.10
OTHER_RISE_MAX = 0.01
BETWEEN_RISE_MAX = 0.01
ALPHA = 0.05
SCOPES = ("cluster", "global")
TEST_NAMES = ("target", "progress", "drift", "other_forgotten", "between", "unit")
NB = oracle.N_MODES + 1   # 9 label buckets, OFF last
#: gate-pool check indices at or above this are reserved for calibration (power check), never runs
CALIBRATION_INDEX0 = 1_000_000


@dataclass(frozen=True)
class GateConfig:
    """Thresholds (development values; Phase 3 may loosen tests 3-5 once to 0.15/0.02/0.02;
    Phase 5 freezes). ``accept_all`` is the A3-nogate arm: every test is still measured."""
    n_g: int = N_G
    n_boot: int = N_BOOT
    target_fall: float = TARGET_FALL
    drift_max: float = DRIFT_MAX
    other_rise_max: float = OTHER_RISE_MAX
    between_rise_max: float = BETWEEN_RISE_MAX
    accept_all: bool = False


@dataclass
class GateResult:
    """Result of one gate check.

    accepted:       all applicable tests passed (or cfg.accept_all)
    passed:         {test name in TEST_NAMES: bool}
    target_fall:    test 1 number: paired fall in target-mode mass, share of the batch
                    (mass_current - mass_new; positive = fewer target actions)
    target_fall_ci: 95% paired-bootstrap interval (lo, hi) of target_fall
    numbers:        every gate number (masses per mode for ref/current/new, D, Collateral,
                    bounds, rises) for the ledger; never shown to the surgeon
    check_index:    gate-pool check index used; n_g: batch size; scope: "cluster" | "global"
    cpu_s:          wall-clock seconds of the check
    """
    accepted: bool
    passed: Dict[str, bool]
    target_fall: float
    target_fall_ci: Tuple[float, float]
    numbers: Dict[str, float] = field(default_factory=dict)
    check_index: int = 0
    n_g: int = N_G
    scope: str = "cluster"
    cpu_s: float = 0.0

    def surgeon_view(self) -> Dict:
        """What the surgeon sees (docs/PLAN.md 7.2): pass/fail per test plus test 1's number and CI."""
        return {"accepted": self.accepted, "passed": dict(self.passed),
                "target_fall": self.target_fall, "target_fall_ci": list(self.target_fall_ci)}

    def ledger_data(self) -> Dict:
        """JSON-safe payload for the ledger ``gate`` record (all numbers; never for the surgeon)."""
        d = asdict(self)
        d["target_fall_ci"] = list(self.target_fall_ci)
        d["surgeon_view"] = self.surgeon_view()
        return d


# ------------------------------------------------------------------ bootstrap
def paired_bootstrap(stat_fn: Callable[[np.ndarray], float], n: int, n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    """(n_boot,) float64 replicates of ``stat_fn(idx)``, idx (n,) int64 row indices drawn with
    replacement; the same idx serves every version in one replicate (paired)."""
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        out[b] = stat_fn(rng.integers(0, n, size=n, dtype=np.int64))
    return out


def _bucket(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.int64)
    return np.where(lab == oracle.OFF, NB - 1, lab)


def _joint_counts(lb: np.ndarray, lc: np.ndarray, ln: np.ndarray) -> np.ndarray:
    code = (_bucket(lb) * NB + _bucket(lc)) * NB + _bucket(ln)
    return np.bincount(code, minlength=NB ** 3).astype(np.int64)


def _joint_bootstrap(counts: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, 3, 9) bootstrap masses (ref, current, new) from joint category counts (729,)."""
    n = int(counts.sum())
    rep = rng.multinomial(n, counts / n, size=n_boot).reshape(n_boot, NB, NB, NB).astype(np.float64) / n
    return np.stack([rep.sum((2, 3)), rep.sum((1, 3)), rep.sum((1, 2))], 1)


def _point_masses(counts: np.ndarray) -> np.ndarray:
    n = counts.sum()
    c = counts.reshape(NB, NB, NB).astype(np.float64) / n
    return np.stack([c.sum((1, 2)), c.sum((0, 2)), c.sum((0, 1))], 0)


def _stats(M: np.ndarray, T: List[int], F: List[int], R: List[int]) -> Dict[str, np.ndarray]:
    """Gate statistics of masses M (..., 3, 9) = (ref, current, new)."""
    mb, mc, mn = M[..., 0, :], M[..., 1, :], M[..., 2, :]
    other = [f for f in F if f not in T]
    s = {
        "target_cur": mc[..., T].sum(-1) if T else np.zeros(mc.shape[:-1]),
        "target_new": mn[..., T].sum(-1) if T else np.zeros(mc.shape[:-1]),
        "D_ref": d_ideal(mb, R), "D_cur": d_ideal(mc, R), "D_new": d_ideal(mn, R),
        "collateral_new": collateral(mn, mb, R), "collateral_cur": collateral(mc, mb, R),
        "retain_tv_new": retain_tv(mn, mb, R),
        "other_rise": (mn[..., other].sum(-1) - mc[..., other].sum(-1)) if other else np.zeros(mc.shape[:-1]),
        "between_rise": mn[..., -1] - mc[..., -1],
    }
    s["target_fall"] = s["target_cur"] - s["target_new"]
    s["dD"] = s["D_new"] - s["D_cur"]
    return s


# ------------------------------------------------------------------ decoding
_REF_CACHE: Dict = {}


def _decode(version, xi, c) -> torch.Tensor:
    return _adapt.actions(version, xi, c)


def _ref_actions(cell, base, xi, c, check_index: int, n_g: int) -> torch.Tensor:
    key = (cell.key, id(cell), id(base.net), getattr(base, "id", None), check_index, n_g)
    if key not in _REF_CACHE:
        if len(_REF_CACHE) > 64:
            _REF_CACHE.clear()
        _REF_CACHE[key] = _decode(base, xi, c)
    return _REF_CACHE[key]


def _target_modes(cell, target_uids, scope: str) -> List[int]:
    def uids_of(x):
        return list(getattr(x, "uids", x))
    if scope == "cluster":
        return oracle.target_modes(cell, [int(u) for u in uids_of(target_uids)])
    modes = set()
    for cl in target_uids:
        modes.update(oracle.target_modes(cell, [int(u) for u in uids_of(cl)]))
    return sorted(modes)


def tests_from_labels(lb, lc, ln, T: List[int], F: List[int], R: List[int], scope: str,
                      cfg: GateConfig, seed_seq: np.random.SeedSequence):
    """Gate tests 1-5 from the oracle labels (B,) of the batch under pi_ref, current, new.

    Returns (passed {test: bool} without "unit", point statistics, bootstrap bounds, joint
    counts). The bootstrap Generator is seeded from a child of ``seed_seq``."""
    counts = _joint_counts(lb, lc, ln)
    rng = np.random.default_rng(seed_seq.spawn(1)[0])
    boot = _stats(_joint_bootstrap(counts, cfg.n_boot, rng), T, F, R)
    pt = {k: float(v) for k, v in _stats(_point_masses(counts), T, F, R).items()}

    def q(k, p):
        return float(np.percentile(boot[k], p))
    up = 100 * (1 - ALPHA)
    fall_lo, fall_hi = q("target_fall", 100 * ALPHA / 2), q("target_fall", 100 * (1 - ALPHA / 2))
    bounds = {"dD_ub": q("dD", up), "collateral_ub": q("collateral_new", up),
              "other_rise_ub": q("other_rise", up), "between_rise_ub": q("between_rise", up),
              "target_fall_lo": fall_lo, "target_fall_hi": fall_hi}
    pt["target_fall_frac"] = pt["target_fall"] / pt["target_cur"] if pt["target_cur"] > 0 else 0.0
    passed = {
        "target": bool(len(T) > 0 and pt["target_cur"] > 0 and pt["target_fall_frac"] >= cfg.target_fall
                       and fall_lo > 0),
        "progress": bool(bounds["dD_ub"] < 0),
        "drift": bool(bounds["collateral_ub"] <= cfg.drift_max),
        "other_forgotten": True if scope == "global" else bool(bounds["other_rise_ub"] <= cfg.other_rise_max),
        "between": bool(bounds["between_rise_ub"] <= cfg.between_rise_max),
    }
    return passed, pt, bounds, counts


# ------------------------------------------------------------------ the gate
def gate_check(cell, base, current, new, target_uids: Union[Sequence[int], Sequence[Sequence[int]]],
               check_index: int, scope: str = "cluster", cfg: GateConfig = GateConfig()) -> GateResult:
    """Run one gate check.

    cell: ``toyrepair.cell.Cell``; base / current / new: ``toyrepair.policy.Version`` (pi_ref,
    current version, candidate). target_uids: the target cluster's member negative uids
    (cluster scope) or one uid list per first-atlas cluster (global scope). Elements may also be
    objects with a ``uids`` attribute (``miner.Cluster``).
    check_index: the run's gate-check counter (0, 1, ...); it seeds the gate pool.
    The caller (orchestrate / cli) increments the budget (``numbers["decodes"]``) and writes
    the ledger record (``GateResult.ledger_data()``)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    t0, p0 = time.time(), time.process_time()
    T = _target_modes(cell, target_uids, scope)
    F, R = oracle.forget_modes(cell), oracle.retain_modes(cell)

    c, xi = pools.draw(cell, pools.GATE, cfg.n_g, index=check_index)
    a_b = _ref_actions(cell, base, xi, c, check_index, cfg.n_g)
    n_dec = 2 * cfg.n_g
    if current is base or current.net is base.net:
        a_c = a_b
    else:
        a_c = _decode(current, xi, c)
        n_dec += cfg.n_g
    a_n = _decode(new, xi, c)
    lb, lc, ln = (oracle.label(cell, a, c) for a in (a_b, a_c, a_n))
    t_dec = time.time() - t0

    passed, pt, bounds, counts = tests_from_labels(lb, lc, ln, T, F, R, scope, cfg,
                                                   pools.seed_sequence(pools.GATE, cell.index, check_index))
    unit = _adapt.unit_tests(new, current)
    passed["unit"] = bool(unit["passed"])
    fall_lo, fall_hi = bounds["target_fall_lo"], bounds["target_fall_hi"]
    accepted = bool(cfg.accept_all or all(passed.values()))
    M = _point_masses(counts)
    numbers = {
        **pt, **bounds,
        "mass_ref": M[0].tolist(), "mass_cur": M[1].tolist(), "mass_new": M[2].tolist(),
        "target_modes": T, "other_forgotten_na": scope == "global", "unit": unit,
        "all_passed": all(passed.values()), "accept_all": cfg.accept_all,
        "thresholds": {"target_fall": cfg.target_fall, "drift_max": cfg.drift_max,
                       "other_rise_max": cfg.other_rise_max, "between_rise_max": cfg.between_rise_max},
        "n_boot": cfg.n_boot, "decodes": n_dec, "decode_s": t_dec,
        "cpu_process_s": time.process_time() - p0,
        "versions": {"base": getattr(base, "id", None), "current": getattr(current, "id", None),
                     "new": getattr(new, "id", None)},
    }
    return GateResult(accepted=accepted, passed=passed, target_fall=pt["target_fall"],
                      target_fall_ci=(fall_lo, fall_hi), numbers=numbers, check_index=int(check_index),
                      n_g=cfg.n_g, scope=scope, cpu_s=time.time() - t0)


# ------------------------------------------------------------------ 10.3 checks
class _ZeroGraftNet(torch.nn.Module):
    """Local no-op graft adapter (until W1-B's ``graft_gated_path``): n hidden units on the
    ``out`` input, always on, with zero output weights. v + 0 equals v bit for bit."""

    def __init__(self, net, n_units: int = 8):
        super().__init__()
        self.inner = copy.deepcopy(net)
        d = 128
        g = torch.Generator().manual_seed(0)
        self.w_in = torch.nn.Parameter(torch.randn(n_units, d, generator=g), requires_grad=False)
        self.w_out = torch.nn.Parameter(torch.zeros(2, n_units), requires_grad=False)
        self._feat = None
        self.inner.out.register_forward_pre_hook(self._grab)

    def _grab(self, mod, inp):
        self._feat = inp[0]

    def forward(self, x, t, c):
        v = self.inner(x, t, c)
        z = torch.nn.functional.silu(self._feat @ self.w_in.T)
        return v + z @ self.w_out.T


def noop_version(cell, base):
    """A no-op edit of ``base``: W1-B's graft with zero output weights if available, else the
    local zero-graft adapter."""
    from . import edits, policy
    try:
        det = edits.Detector(source="t", w=(0.0,), b=1.0, threshold=0.0)   # always fires
        r = edits.graft_gated_path(base, "v_noop", "out", det, 8,
                                   np.ones((8, 128), np.float32), np.zeros((2, 8), np.float32))
        return r.version, "edits.graft_gated_path"
    except NotImplementedError:
        net = _ZeroGraftNet(base.net).eval()
        return policy.Version(id="v_noop", parent=base.id, patches=("noop",), net=net, grid=base.grid), "local"


def noop_check(cell, base, uids, n_checks: int = 2, cfg: GateConfig = GateConfig()) -> Dict:
    """Reproducibility (PLAN 10.3): the no-op edit gives bit-identical decodes on the gate batch,
    and two runs of the same gate check give identical numbers."""
    new, how = noop_version(cell, base)
    c, xi = pools.draw(cell, pools.GATE, cfg.n_g, index=0)
    a0, a1 = _decode(base, xi, c), _decode(new, xi, c)
    identical = bool(torch.equal(a0, a1))
    r = [gate_check(cell, base, base, new, uids, check_index=0, cfg=cfg) for _ in range(n_checks)]
    strip = lambda d: {k: v for k, v in d.items() if k not in ("decode_s", "cpu_process_s")}  # noqa: E731
    repro = all(strip(x.numbers) == strip(r[0].numbers) and x.passed == r[0].passed for x in r)
    return {"noop_impl": how, "decodes_identical": identical, "max_abs_diff": float((a0 - a1).abs().max()),
            "gate_repro": bool(repro), "noop_target_fall": r[0].target_fall,
            "noop_passed": r[0].passed, "noop_dD_ub": r[0].numbers["dD_ub"]}


def noisy_version(base, sigma: float, seed: int = 0):
    """base with N(0, sigma^2) noise added to the ``out`` layer's weight and bias (fixed direction
    per seed, so D(sigma) is monotone-ish and bisection works)."""
    from . import policy
    net = copy.deepcopy(base.net)
    g = torch.Generator().manual_seed(seed)
    out = _out_layer(net)
    with torch.no_grad():
        out.weight.add_(sigma * torch.randn(out.weight.shape, generator=g))
        out.bias.add_(sigma * torch.randn(out.bias.shape, generator=g))
    return policy.Version(id=f"v_noise{sigma:.4g}", parent=base.id, patches=(), net=net, grid=base.grid)


def _out_layer(net):
    for name in ("out", "inner.out"):
        m = net
        try:
            for p in name.split("."):
                m = getattr(m, p)
            return m
        except AttributeError:
            continue
    raise AttributeError("no out layer")


def power_check(cell, base, uids, target_dD: float = 0.01, n_sep: int = 65_536, n_batches: int = 20,
                cfg: GateConfig = GateConfig(), log=print) -> Dict:
    """Gate power check (PLAN 10.3).

    1. Find sigma such that last-layer noise raises D by ``target_dD`` on ``n_sep`` separate
       samples (gate pool, calibration index ``CALIBRATION_INDEX0``; never used by a run).
    2. On ``n_batches`` fresh gate batches (calibration indices CALIBRATION_INDEX0 + 1 + j):
       * literal: current = pi_ref, new = noisy. Test 2 must reject (fail) in >= 80%.
       * detect:  current = noisy, new = pi_ref (a true D improvement of target_dD). Share of
         batches where test 2 passes. This is the informative direction: the literal one is
         rejected by construction whenever the UB is >= 0, even for a no-op.
    Returns the rates, sigma, the realised dD, and seconds per gate check."""
    from . import oracle as O
    R = O.retain_modes(cell)
    c, xi = pools.draw(cell, pools.GATE, n_sep, index=CALIBRATION_INDEX0)
    d_base = float(d_ideal(O.masses(O.label(cell, _decode(base, xi, c), c)), R))

    def dD(sig):
        v = noisy_version(base, sig)
        return float(d_ideal(O.masses(O.label(cell, _decode(v, xi, c), c)), R)) - d_base

    lo, hi = 0.0, 0.01
    while dD(hi) < target_dD and hi < 10:
        lo, hi = hi, hi * 2
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if dD(mid) < target_dD:
            lo = mid
        else:
            hi = mid
    sigma = hi
    realised = dD(sigma)
    noisy = noisy_version(base, sigma)
    log(f"[power] {cell.key}: sigma={sigma:.5g} dD={realised:.4f} (target {target_dD}) n_g={cfg.n_g}")
    lit, det, secs, fall = [], [], [], []
    for j in range(n_batches):
        ci = CALIBRATION_INDEX0 + 1 + j
        r1 = gate_check(cell, base, base, noisy, uids, check_index=ci, cfg=cfg)
        r2 = gate_check(cell, base, noisy, base, uids, check_index=ci, cfg=cfg)
        lit.append(not r1.passed["progress"])
        det.append(r2.passed["progress"])
        secs += [r1.cpu_s, r2.cpu_s]
        fall.append(r2.numbers["dD"])
    out = {"cell": cell.key, "n_g": cfg.n_g, "sigma": sigma, "dD_sep": realised, "n_sep": n_sep,
           "n_batches": n_batches, "literal_reject_rate": float(np.mean(lit)),
           "detect_improvement_rate": float(np.mean(det)), "gate_s_mean": float(np.mean(secs)),
           "gate_s_max": float(np.max(secs)), "dD_gate_mean": float(np.mean(fall)),
           "dD_gate_sd": float(np.std(fall))}
    log(f"[power] {json.dumps(out)}")
    return out


def recipe_check(cell, base, arm: str, n_batches: int = 5, cfg: GateConfig = GateConfig()) -> Dict:
    """Train one toypref arm on the whole cell (as ``scripts/cond_cell.py`` does), then run the
    global-scope gate on ``n_batches`` calibration batches and the final evaluation. Shows how
    the gate treats a real repair and the gate-vs-final gap. Development seeds only."""
    from toypref.cond import experiment as E
    from . import eval_final, policy
    t0 = time.time()
    pol, _ = E.train_arm(E.ARMS[arm], cell.toypref_ctx())
    train_s = time.time() - t0
    new = policy.Version(id=f"v_{arm}", parent=base.id, patches=(), net=pol.net, grid=cell.grid)
    groups = [[int(u) for u in cell.neg.uid]]   # global scope: target = all forgotten modes
    rs = [gate_check(cell, base, base, new, groups, check_index=CALIBRATION_INDEX0 + 100 + j,
                     scope="global", cfg=cfg) for j in range(n_batches)]
    fin = eval_final.evaluate(cell, new, arm)
    keys = ("D_new", "D_cur", "dD", "dD_ub", "collateral_new", "collateral_ub", "retain_tv_new",
            "between_rise", "between_rise_ub", "target_fall", "target_fall_frac")
    return {"arm": arm, "train_s": train_s, "accept_rate": float(np.mean([r.accepted for r in rs])),
            "pass_rates": {t: float(np.mean([r.passed[t] for r in rs])) for t in TEST_NAMES},
            "gate_mean": {k: float(np.mean([r.numbers[k] for r in rs])) for k in keys},
            "gate_s": float(np.mean([r.cpu_s for r in rs])),
            "final": {"D": fin.D, "removed": fin.removed, "collateral": fin.collateral,
                      "collateral_plan": fin.collateral_plan, "mass_off": fin.mass_off,
                      "final_pool": {k: fin.final_pool[k] for k in ("D", "removed", "collateral", "retain_tv", "mass_off")}}}


def _main(argv=None):
    import argparse
    from .cell import load_cell
    ap = argparse.ArgumentParser(prog="python -m toyrepair.gate")
    ap.add_argument("cmd", choices=["power", "noop", "recipe"])
    ap.add_argument("--arm", default="R_hopB_bsup_neg_AMdist", help="toypref arm for `recipe`")
    ap.add_argument("--grid", default="cond")
    ap.add_argument("--setting", default="k4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npm", type=int, default=4)
    ap.add_argument("--n_g", type=int, default=N_G)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--out", default=None, help="JSON output path")
    a = ap.parse_args(argv)
    if a.seed > 4:
        raise SystemExit("calibration uses development seeds 0-4 only")
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    cell = load_cell(a.grid, a.setting, a.seed, a.npm)
    base = _adapt.base_version(cell)
    uids = [int(u) for u in cell.neg.uid]   # all negatives: power/noop do not depend on the target
    cfg = GateConfig(n_g=a.n_g)
    if a.cmd == "noop":
        res = noop_check(cell, base, uids, cfg=cfg)
    elif a.cmd == "power":
        res = power_check(cell, base, uids, n_batches=a.n_batches, cfg=cfg)
    else:
        res = recipe_check(cell, base, a.arm, n_batches=a.n_batches, cfg=cfg)
    res = {"cmd": a.cmd, "cell": cell.key, "n_g": a.n_g, **res}
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    _main()
