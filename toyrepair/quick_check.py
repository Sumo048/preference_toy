"""quick_check: oracle-free check of a candidate version (docs/PLAN.md 6.8).

Owner: W1-A1.
"""

from __future__ import annotations

from dataclasses import dataclass

N_DRIFT = 4096


@dataclass(frozen=True)
class QuickCheckResult:
    """still_bad: share of the cluster's logged noises (xi_i, c_i) that still decode into B_i;
    residual: the cluster's residual basin mass (mean r_i, 256 mining noises per member);
    drift: share of 4096 mining-pool pairs whose decode moves by more than the median basin
    radius from the current version's decode. All in [0, 1]."""
    still_bad: float
    residual: float
    drift: float
    decodes: int = 0


def quick_check(cell, current, candidate, cluster) -> QuickCheckResult:
    raise NotImplementedError
