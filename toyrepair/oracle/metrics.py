"""Removed, Collateral, D, and the toypref ``compute`` wrapper (gate and final evaluation only).

Definitions (method map; toypref ``experiment.compute``), over the 9 buckets 8 modes + OFF:
  Removed     = 1 - neg_mass_ratio = 1 - m[F].sum() / m_ref[F].sum()
  D           = tv_to_ideal = 1/2 sum |m - ideal| over 8 modes + OFF, ideal uniform on R.
  retain_tv   = toypref's collateral: TV over the retained modes only, each renormalised. This is
                what the method map reports as "collateral" (verified in W0 SURVEY.md).
  collateral  = the PLAN.md Section 2 definition: TV between the policy and pi_ref over the
                retained modes PLUS the between-modes bucket (R + OFF), each renormalised over
                those buckets. The gate's drift test (test 3) uses this one. Final evaluation
                reports both; ``FinalResult.collateral`` is the method-map ``retain_tv``.

Owner: W1-A2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from . import _adapt
from .labels import N_MODES, OFF, label, masses


@dataclass(frozen=True)
class Scores:
    """D, removed, collateral (PLAN definition), mass_off, retain_tv (method-map collateral);
    shares in [0, 1]."""
    D: float
    removed: float
    collateral: float
    mass_off: float
    retain_tv: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"D": self.D, "removed": self.removed, "collateral": self.collateral,
                "mass_off": self.mass_off, "retain_tv": self.retain_tv}


def ideal(retain: Sequence[int], n_modes: int = N_MODES) -> np.ndarray:
    """(n_modes + 1,) ideal mass: uniform on the retained modes, 0 elsewhere (incl. OFF)."""
    out = np.zeros(n_modes + 1)
    out[list(retain)] = 1.0 / len(retain)
    return out


def d_ideal(m: np.ndarray, retain: Sequence[int]) -> np.ndarray:
    """D of mass vectors m (..., 9): 1/2 sum |m - ideal|."""
    m = np.asarray(m, dtype=np.float64)
    return 0.5 * np.abs(m - ideal(retain, m.shape[-1] - 1)).sum(-1)


def _renorm_tv(m: np.ndarray, m_ref: np.ndarray, idx) -> np.ndarray:
    p = m[..., idx]
    q = m_ref[..., idx]
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-12)
    return 0.5 * np.abs(p - q).sum(-1)


def collateral(m: np.ndarray, m_ref: np.ndarray, retain: Sequence[int]) -> np.ndarray:
    """PLAN collateral of (..., 9) mass vectors: renormalised TV over retained modes + OFF."""
    m = np.asarray(m, dtype=np.float64)
    return _renorm_tv(m, np.asarray(m_ref, dtype=np.float64), list(retain) + [m.shape[-1] - 1])


def retain_tv(m: np.ndarray, m_ref: np.ndarray, retain: Sequence[int]) -> np.ndarray:
    """toypref ``retain_tv`` of (..., 9) mass vectors: renormalised TV over retained modes only."""
    return _renorm_tv(np.asarray(m, dtype=np.float64), np.asarray(m_ref, dtype=np.float64), list(retain))


def scores_from_masses(m: np.ndarray, m_ref: np.ndarray, forget, retain) -> Scores:
    """Scores from (9,) mass vectors of the policy and pi_ref."""
    m = np.asarray(m, dtype=np.float64)
    m_ref = np.asarray(m_ref, dtype=np.float64)
    F = list(forget)
    removed = 1.0 - float(m[F].sum() / max(m_ref[F].sum(), 1e-12))
    return Scores(D=float(d_ideal(m, retain)), removed=removed, collateral=float(collateral(m, m_ref, retain)),
                  mass_off=float(m[-1]), retain_tv=float(retain_tv(m, m_ref, retain)))


class _Sampler:
    """``sample(xi, c)`` interface over a version, for toypref's ``compute``."""

    def __init__(self, version):
        self.version = version

    @torch.no_grad()
    def sample(self, xi, c, batch=65536):
        return _adapt.actions(self.version, xi, c)


_BANKS: Dict = {}


def _cond_bank(cell):
    key = ("cond", cell.key, id(cell.ref))
    if key not in _BANKS:
        from toypref.cond import experiment as E
        _BANKS[key] = E.Bank(cell.ref, cell.seed, cell._oracle.toypref_edit)
    return _BANKS[key]


def _uncond_bank(cell):
    """toypref's fixed unconditional eval noises: ``randn(n_eval, 2, gen(seed, EVAL_NOISE))``
    (``scripts/run_cell.py``), with pi_ref's labels."""
    key = ("uncond", cell.key, id(cell.ref))
    if key not in _BANKS:
        from toypref import seeds
        from toypref.config import DEFAULT
        Z = torch.randn(DEFAULT.n_eval, 2, generator=seeds.gen(cell.seed, seeds.EVAL_NOISE))
        a_ref = cell.ref.sample(Z)
        c = torch.zeros(Z.shape[0])
        _BANKS[key] = {"Z": Z, "c": c, "a_ref": a_ref, "lab_ref": label(cell, a_ref, c)}
    return _BANKS[key]


def toypref_compute(cell, version) -> Dict[str, float]:
    """toypref ``experiment.compute(Bank(ref, seed, ed), sampler)`` on the method-map eval bank
    (20,000 fixed (c, xi), stream P_EVAL); returns toypref's dict (tv_to_ideal, neg_mass_ratio,
    retain_tv, mass_off, ...). ``version``: a ``policy.Version`` (or any object with ``.net``).

    Unconditional cells: toypref's unconditional ``metrics.compute`` needs log-probs and test
    samples that the gate does not; this returns the same mass-based keys (``mass_m*``,
    ``mass_off``, ``neg_mass_ratio``, ``retain_tv``, ``tv_to_ideal``, ``tv_to_ideal_ref``) on
    toypref's fixed unconditional eval noises with the same formulas."""
    from . import forget_modes, retain_modes
    if cell.grid == "cond":
        from toypref.cond import experiment as E
        return {k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
                for k, v in E.compute(_cond_bank(cell), _Sampler(version)).items()}
    b = _uncond_bank(cell)
    a = _adapt.actions(version, b["Z"], b["c"])
    lab = label(cell, a, b["c"])
    m, mr = masses(lab), masses(b["lab_ref"])
    F, R = forget_modes(cell), retain_modes(cell)
    out = {f"mass_m{j}": float(m[j]) for j in range(N_MODES)}
    out["mass_off"] = float(m[N_MODES])
    out["neg_mass_ratio"] = float(m[F].sum() / max(mr[F].sum(), 1e-12))
    out["retain_tv"] = float(retain_tv(m, mr, R))
    out["tv_to_ideal"] = float(d_ideal(m, R))
    out["tv_to_ideal_ref"] = float(d_ideal(mr, R))
    return out


__all__ = ["Scores", "scores_from_masses", "toypref_compute", "ideal", "d_ideal", "collateral",
           "retain_tv", "OFF"]
