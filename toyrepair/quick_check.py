"""quick_check: oracle-free check of a candidate version (docs/PLAN.md 6.8).

Owner: W1-A1.

Three numbers, no oracle:
  still_bad  share of the cluster's logged noises (xi_i, c_i) that the candidate still decodes into B_i
  residual   the cluster's residual basin mass under the candidate (mean r_i, 256 mining noises per member)
  drift      share of the 4096 mining-pool pairs ``pools.draw(cell, MINING, 4096)`` whose candidate
             decode moves by more than the median basin radius (over all of the cell's negatives)
             from the current version's decode
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

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

    def render(self) -> str:
        return (f"quick_check: still_bad={self.still_bad:.3f} residual={self.residual:.3f} "
                f"drift={self.drift:.4f} (decodes {self.decodes})")


def quick_check(cell, current, candidate, cluster) -> QuickCheckResult:
    """``cluster``: ``toyrepair.miner.Cluster``; current / candidate: ``policy.Version``."""
    from . import flowstate, miner, policy, pools
    t0 = policy.counter.total
    rows = np.asarray(cluster.members, dtype=int)
    idx = torch.as_tensor(rows)
    act = policy.decode(candidate, cell.neg.xi[idx], cell.neg.c[idx]).action
    still = float(np.mean([cell.in_bad_basin(int(i), act[j:j + 1])[0] for j, i in enumerate(rows)]))
    resid = float(miner.residual_mass(cell, candidate, rows).mean())
    c, xi = pools.draw(cell, pools.MINING, N_DRIFT)
    a_cur = policy.decode(current, xi, c).action
    a_new = policy.decode(candidate, xi, c).action
    thr = float(np.median(flowstate.basin_radii(cell)))
    drift = float((torch.linalg.norm(a_new - a_cur, dim=1) > thr).float().mean())
    return QuickCheckResult(still_bad=still, residual=resid, drift=drift,
                            decodes=int(round(policy.counter.total - t0)))
