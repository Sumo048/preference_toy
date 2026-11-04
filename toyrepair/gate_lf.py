"""The label-free gate (arm A3-LF; docs/A3LF_PREREG.md). Owner: session A3LF.

Same six tests, same thresholds, same gate batch and same bootstrap as ``toyrepair.gate``
(PREREG Section 2), but every mass comes from the label-free behaviour tracks of
``toyrepair.tracks`` instead of the oracle. The tracks are built from pi_ref decodes (mining
pool, stream 9) and are marked by the logged negatives' (c, a) only. This module never imports
``toyrepair.oracle`` (``tests/test_gate_lf.py`` checks it in a fresh interpreter).

Buckets of an action a at condition c (``Tracks.assign``):
  target  T: the tracks marked by the target cluster's members' logged actions
  other   O: the tracks marked by any other logged negative (bad tracks minus T)
  good    G: every other track that holds >= MIN_TRACK_SHARE (2%) of pi_ref's track-build decodes
             (one bucket per track)
  between U: assigned to no track (-1), or to an unmarked track below that share (a fragment)

Masses are shares of the whole gate batch (all c), as in the oracle gate.

Tests (all must pass; thresholds = ``toyrepair.gate`` constants, PREREG Section 2):
  1 target:   mass on T falls by >= 50% of its current value (point estimate) and the 2.5th
              percentile of the paired fall is > 0
  2 progress: one-sided 95% UB of D_lf(new) - D_lf(current) < 0, where
              D_lf(m) = 1/2 sum_b |m_b - ideal_b| over all buckets (T, O, each good track, U),
              ideal uniform over the good tracks
  3 drift:    one-sided 95% UB of Collateral_lf(new, pi_ref) <= 0.15, the TV between new and
              pi_ref over the good tracks plus U, each renormalised over those buckets
  4 other:    one-sided 95% UB of the rise in mass on O <= 0.02
  5 between:  one-sided 95% UB of the rise in mass on U <= 0.02
  6 unit:     ``toyrepair.edits.unit_tests(new, current)`` (the same hook as the oracle gate)

Bootstrap: the label vectors are compressed to the buckets above, then the joint (ref, current,
new) category counts are resampled as one multinomial (the oracle gate's method,
``gate._joint_bootstrap`` re-implemented here), seeded from the same gate-pool seed sequence.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from . import pools, tracks

# Thresholds: the frozen values of toyrepair.gate (PREREG Section 2). Repeated here, not imported,
# because toyrepair.gate imports the oracle at import time. tests/test_gate_lf.py checks equality.
N_G = 16_384
N_BOOT = 2_000
TARGET_FALL = 0.5
DRIFT_MAX = 0.15
OTHER_RISE_MAX = 0.02
BETWEEN_RISE_MAX = 0.02
ALPHA = 0.05
TEST_NAMES = ("target", "progress", "drift", "other_forgotten", "between", "unit")
#: a track that no negative marks is a good track only if it holds at least this share of pi_ref's track-build
#: decodes (the 2% floor tracks.py applies per condition); smaller unmarked fragments count as between modes (U).
#: Chosen on development seed 0 before any seed 5-9 cell (docs/A3LF_PREREG.md Section 2).
MIN_TRACK_SHARE = 0.02
#: joint-category bootstrap while NB**3 stays below this; else index resampling
MAX_JOINT = 300_000


@dataclass(frozen=True)
class GateConfig:
    n_g: int = N_G
    n_boot: int = N_BOOT
    target_fall: float = TARGET_FALL
    drift_max: float = DRIFT_MAX
    other_rise_max: float = OTHER_RISE_MAX
    between_rise_max: float = BETWEEN_RISE_MAX
    accept_all: bool = False


@dataclass
class GateResult:
    """Same fields and surgeon view as ``toyrepair.gate.GateResult``. ``numbers`` holds the
    label-free masses (never shown to the surgeon)."""
    accepted: bool
    passed: Dict[str, bool]
    target_fall: float
    target_fall_ci: Tuple[float, float]
    numbers: Dict = field(default_factory=dict)
    check_index: int = 0
    n_g: int = N_G
    scope: str = "cluster"
    cpu_s: float = 0.0
    gate: str = "lf"

    def surgeon_view(self) -> Dict:
        return {"accepted": self.accepted, "passed": dict(self.passed),
                "target_fall": self.target_fall, "target_fall_ci": list(self.target_fall_ci)}

    def ledger_data(self) -> Dict:
        d = asdict(self)
        d["target_fall_ci"] = list(self.target_fall_ci)
        d["surgeon_view"] = self.surgeon_view()
        return d


# ------------------------------------------------------------------ buckets
@dataclass(frozen=True)
class Buckets:
    """Bucket layout: 0 = target T, 1 = other bad O, 2..2+n_good-1 = good tracks, last = U."""
    target: Tuple[int, ...]
    other: Tuple[int, ...]
    good: Tuple[int, ...]

    @property
    def nb(self) -> int:
        return len(self.good) + 3

    def of(self, track_ids: np.ndarray) -> np.ndarray:
        t = np.asarray(track_ids, dtype=np.int64)
        out = np.full(t.shape, self.nb - 1, dtype=np.int64)      # unassigned
        out[np.isin(t, self.target)] = 0
        out[np.isin(t, self.other)] = 1
        for j, g in enumerate(self.good):
            out[t == g] = 2 + j
        return out


def track_mass(tr: "tracks.Tracks") -> np.ndarray:
    """(n_tracks,) share of pi_ref's typical track-build decodes (all grid conditions) on each track."""
    cnt = np.zeros(tr.n_tracks)
    for k, (pts, lab, _, _) in enumerate(tr.levels):
        lab = np.asarray(lab)
        np.add.at(cnt, tr.track[tr.offs[k] + lab], 1)
    return cnt / max(cnt.sum(), 1)


def buckets_for(tr: "tracks.Tracks", cell, target_uids: Sequence[int]) -> Buckets:
    """Target = tracks the cluster marks; other = tracks any other logged negative marks; good = unmarked
    tracks with at least MIN_TRACK_SHARE of pi_ref's track mass; smaller unmarked tracks fall into U."""
    all_uids = [int(u) for u in np.asarray(cell.neg.uid)]
    tgt = tr.marked(cell, [int(u) for u in target_uids])
    bad = tr.marked(cell, all_uids)
    other = bad - tgt
    share = track_mass(tr)
    good = [k for k in range(tr.n_tracks) if k not in bad and share[k] >= MIN_TRACK_SHARE]
    return Buckets(target=tuple(sorted(tgt)), other=tuple(sorted(other)), good=tuple(good))


# ------------------------------------------------------------------ statistics
def d_lf(m: np.ndarray, n_good: int) -> np.ndarray:
    """D_lf of bucket masses m (..., NB): 1/2 sum |m - ideal|, ideal uniform on the good tracks."""
    m = np.asarray(m, dtype=np.float64)
    ideal = np.zeros(m.shape[-1])
    if n_good:
        ideal[2:2 + n_good] = 1.0 / n_good
    return 0.5 * np.abs(m - ideal).sum(-1)


def collateral_lf(m: np.ndarray, m_ref: np.ndarray) -> np.ndarray:
    """Renormalised TV between m and m_ref over the good tracks plus U (buckets 2..NB-1)."""
    p = np.asarray(m, dtype=np.float64)[..., 2:]
    q = np.asarray(m_ref, dtype=np.float64)[..., 2:]
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-12)
    return 0.5 * np.abs(p - q).sum(-1)


def _stats(M: np.ndarray, n_good: int) -> Dict[str, np.ndarray]:
    mb, mc, mn = M[..., 0, :], M[..., 1, :], M[..., 2, :]
    s = {"target_cur": mc[..., 0], "target_new": mn[..., 0],
         "D_ref": d_lf(mb, n_good), "D_cur": d_lf(mc, n_good), "D_new": d_lf(mn, n_good),
         "collateral_new": collateral_lf(mn, mb), "collateral_cur": collateral_lf(mc, mb),
         "other_rise": mn[..., 1] - mc[..., 1], "between_rise": mn[..., -1] - mc[..., -1]}
    s["target_fall"] = s["target_cur"] - s["target_new"]
    s["dD"] = s["D_new"] - s["D_cur"]
    return s


def _joint_counts(bb, bc, bn, nb):
    return np.bincount((bb * nb + bc) * nb + bn, minlength=nb ** 3).astype(np.int64)


def _point_masses(counts, nb):
    c = counts.reshape(nb, nb, nb).astype(np.float64) / counts.sum()
    return np.stack([c.sum((1, 2)), c.sum((0, 2)), c.sum((0, 1))], 0)


def _boot_masses(bb, bc, bn, nb, n_boot, rng) -> np.ndarray:
    """(n_boot, 3, NB) paired-bootstrap masses."""
    n = len(bb)
    if nb ** 3 <= MAX_JOINT:
        counts = _joint_counts(bb, bc, bn, nb)
        rep = rng.multinomial(n, counts / n, size=n_boot).reshape(n_boot, nb, nb, nb).astype(np.float64) / n
        return np.stack([rep.sum((2, 3)), rep.sum((1, 3)), rep.sum((1, 2))], 1)
    out = np.empty((n_boot, 3, nb))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        for v, lab in enumerate((bb, bc, bn)):
            out[b, v] = np.bincount(lab[idx], minlength=nb) / n
    return out


def tests_from_buckets(bb, bc, bn, bk: Buckets, cfg: GateConfig, seed_seq: np.random.SeedSequence):
    nb = bk.nb
    rng = np.random.default_rng(seed_seq.spawn(1)[0])
    boot = _stats(_boot_masses(bb, bc, bn, nb, cfg.n_boot, rng), len(bk.good))
    M = np.stack([np.bincount(x, minlength=nb) / len(x) for x in (bb, bc, bn)], 0)
    pt = {k: float(v) for k, v in _stats(M, len(bk.good)).items()}

    def q(k, p):
        return float(np.percentile(boot[k], p))
    up = 100 * (1 - ALPHA)
    fall_lo, fall_hi = q("target_fall", 100 * ALPHA / 2), q("target_fall", 100 * (1 - ALPHA / 2))
    bounds = {"dD_ub": q("dD", up), "collateral_ub": q("collateral_new", up),
              "other_rise_ub": q("other_rise", up), "between_rise_ub": q("between_rise", up),
              "target_fall_lo": fall_lo, "target_fall_hi": fall_hi}
    pt["target_fall_frac"] = pt["target_fall"] / pt["target_cur"] if pt["target_cur"] > 0 else 0.0
    passed = {
        "target": bool(len(bk.target) > 0 and pt["target_cur"] > 0 and pt["target_fall_frac"] >= cfg.target_fall
                       and fall_lo > 0),
        "progress": bool(bounds["dD_ub"] < 0),
        "drift": bool(bounds["collateral_ub"] <= cfg.drift_max),
        "other_forgotten": bool(bounds["other_rise_ub"] <= cfg.other_rise_max),
        "between": bool(bounds["between_rise_ub"] <= cfg.between_rise_max),
    }
    return passed, pt, bounds, M


# ------------------------------------------------------------------ the gate
_REF_CACHE: Dict = {}


def _decode(version, xi, c) -> torch.Tensor:
    from . import policy
    return policy.decode(version, xi, c).action


def gate_check(cell, base, current, new, target_uids: Sequence[int], check_index: int,
               cfg: GateConfig = GateConfig(), tr=None) -> GateResult:
    """Label-free gate check, cluster scope. Same arguments as ``toyrepair.gate.gate_check``
    (target_uids: the cluster's member negative uids). Same batch: gate pool, ``check_index``."""
    t0 = time.time()
    tr = tr if tr is not None else tracks.get(cell)
    uids = [int(u) for u in getattr(target_uids, "uids", target_uids)]
    bk = buckets_for(tr, cell, uids)
    c, xi = pools.draw(cell, pools.GATE, cfg.n_g, index=check_index)
    key = (cell.key, id(base.net), check_index, cfg.n_g)
    if key not in _REF_CACHE:
        if len(_REF_CACHE) > 64:
            _REF_CACHE.clear()
        _REF_CACHE[key] = _decode(base, xi, c)
    a_b = _REF_CACHE[key]
    n_dec = 2 * cfg.n_g
    if current is base or current.net is base.net:
        a_c = a_b
    else:
        a_c = _decode(current, xi, c)
        n_dec += cfg.n_g
    a_n = _decode(new, xi, c)
    bb, bc, bn = (bk.of(tr.assign(a, c)) for a in (a_b, a_c, a_n))
    passed, pt, bounds, M = tests_from_buckets(bb, bc, bn, bk, cfg,
                                               pools.seed_sequence(pools.GATE, cell.index, check_index))
    from . import edits
    u = edits.unit_tests(new, current)
    unit = {"passed": bool(u.passed), "checks": {k: bool(v) for k, v in u.checks.items()},
            "detail": dict(u.detail)}
    passed["unit"] = unit["passed"]
    accepted = bool(cfg.accept_all or all(passed.values()))
    numbers = {**pt, **bounds, "mass_ref": M[0].tolist(), "mass_cur": M[1].tolist(), "mass_new": M[2].tolist(),
               "buckets": {"target": list(bk.target), "other": list(bk.other), "good": list(bk.good),
                           "n_tracks": int(tr.n_tracks)},
               "unit": unit, "all_passed": all(passed.values()), "accept_all": cfg.accept_all,
               "thresholds": {"target_fall": cfg.target_fall, "drift_max": cfg.drift_max,
                              "other_rise_max": cfg.other_rise_max, "between_rise_max": cfg.between_rise_max},
               "n_boot": cfg.n_boot, "decodes": n_dec,
               "versions": {"base": getattr(base, "id", None), "current": getattr(current, "id", None),
                            "new": getattr(new, "id", None)}}
    return GateResult(accepted=accepted, passed=passed, target_fall=pt["target_fall"],
                      target_fall_ci=(bounds["target_fall_lo"], bounds["target_fall_hi"]), numbers=numbers,
                      check_index=int(check_index), n_g=cfg.n_g, scope="cluster", cpu_s=time.time() - t0)
