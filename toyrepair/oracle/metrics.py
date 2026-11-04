"""Removed, Collateral, D, and the toypref ``compute`` wrapper (gate and final evaluation only).

Definitions (method map; toypref ``experiment.compute``):
  Removed     = 1 - neg_mass_ratio = 1 - m[F].sum() / m_ref[F].sum()
  Collateral  = retain_tv (TV over retained modes, renormalised). The plan says "retained modes
                plus the between-modes region, each renormalised"; W1 verifies which one the
                method map reports (``retain_tv`` in toypref) and keeps that.
  D           = tv_to_ideal = 1/2 sum |m - ideal| over 8 modes + OFF, ideal uniform on R.

Owner: W1-A2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Scores:
    """D, removed, collateral, mass_off (shares in [0, 1])."""
    D: float
    removed: float
    collateral: float
    mass_off: float


def scores_from_masses(m: np.ndarray, m_ref: np.ndarray, forget, retain) -> Scores:
    """Scores from (9,) mass vectors of the policy and pi_ref."""
    raise NotImplementedError


def toypref_compute(cell, version) -> Dict[str, float]:
    """toypref ``experiment.compute(Bank(ref, seed, ed), sampler)`` on the method-map eval bank
    (20,000 fixed (c, xi), stream P_EVAL); returns toypref's dict (tv_to_ideal, neg_mass_ratio,
    retain_tv, mass_off, ...). ``version`` must expose ``sample(xi, c)``."""
    raise NotImplementedError
