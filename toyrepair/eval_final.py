"""Final evaluation of a version on a cell (docs/PLAN.md 4.8, 9.5).

One of only two modules allowed to import ``toyrepair.oracle``. Final D, Removed and Collateral
come from the toypref code (``toyrepair.oracle.metrics.toypref_compute``: toypref's fixed eval
bank, 20,000 (c, xi) on stream (seed, P_EVAL)), so the numbers match the method map. Results
also include toyrepair FINAL-pool numbers (``pools.FINAL``) and the target cluster's removed
share. Final numbers never go back into any loop.

Owner: W1-A2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

N_FINAL = 65_536


@dataclass
class FinalResult:
    """cell key; arm; version id; D, removed, collateral, mass_off (toypref eval bank);
    toypref: full toypref compute dict; target_removed: removed share of the target clusters'
    oracle modes (None for global arms); final_pool: same scores on the FINAL pool."""
    cell: str
    arm: str
    version: str
    D: float
    removed: float
    collateral: float
    mass_off: float
    toypref: Dict[str, float] = field(default_factory=dict)
    target_removed: Optional[float] = None
    final_pool: Dict[str, float] = field(default_factory=dict)


def evaluate(cell, version, arm: str, target_clusters: Optional[List] = None) -> FinalResult:
    raise NotImplementedError
