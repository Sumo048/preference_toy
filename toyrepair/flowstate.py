"""Flow states, renoise, primitives and the branch tool (docs/PLAN.md 2, 5.4, 6.3).

Flow state: (x_k, t_k, c) at step k of one decode. A branch restarts the decode from a saved
flow state. Renoise at step k >= 1: x_hat = x_k - t_k * v(x_k, t_k, c);
x' = t_k * eps + (1 - t_k) * x_hat, eps ~ N(0, I) fresh. At step 0 x' is fresh noise.

Rescue: the branch ends outside B_i at a typical action (5th-NN distance to pi_ref samples at
c_i at or below their 90th percentile, as ``toypref.cond.targets``).
p(k): share of M renoise branches from step k that end in a rescue, averaged over members.
k_commit: earliest sampled step after which every later p(k) < 0.2; if none, the last step.
Class R: a primitive at or before k_commit rescues >= 50% of members; else U.

Owner: W1-A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_STEPS: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16)
M_BRANCH = 32                 # renoise branches per member and step
M_EXTRA = 64                  # extra branches when the 95% Wilson interval contains P_COMMIT
P_COMMIT = 0.2                # k_commit threshold on p(k)
RESCUE_CLASS_R = 0.5          # class R if a primitive rescues >= 50% of members
PUSH_DIRS = 8
PUSH_ALPHAS = (0.5, 1.0, 2.0)  # x basin radius (90th pct distance of B_i samples to their mean)
PUSH_H = 3
TABLE_ROW_CAP = 40
TOP_PRIMITIVES = 5


@dataclass(frozen=True)
class FlowState:
    """x: (B, 2) float32 state x_k; k: step 0..20; c: (B,) float32. t_k = 1 - k/20."""
    x: torch.Tensor
    k: int
    c: torch.Tensor

    @property
    def t(self) -> float:
        return 1.0 - self.k / 20.0


@dataclass(frozen=True)
class Primitive:
    """A scripted change to a decode.

    kind "push": add total displacement alpha * u over steps k .. k+h-1 (alpha/h per step,
                 added to x after each Euler update); u (2,) unit vector; alpha in action units.
    kind "hop":  xi <- xi + tau * u at step 0 (k = 0, h = 0, alpha = tau, noise units).
    name: short id for tables, e.g. ``push(d3,1.0r,k6,h3)`` or ``hop(0.45,d7)``.
    """
    kind: str
    u: Tuple[float, float]
    alpha: float
    k: int
    h: int = PUSH_H
    name: str = ""


@dataclass(frozen=True)
class BranchRow:
    """One row of the branch table.

    step k, t_k, p (rescue share), ci (95% Wilson lo, hi), n (branches used per member),
    best: up to 5 (primitive name, rescue rate in [0, 1]) at this step (empty if none run)."""
    k: int
    t: float
    p: float
    ci: Tuple[float, float]
    n: int
    best: Tuple[Tuple[str, float], ...] = ()


@dataclass
class BranchTable:
    """cluster: cluster id; rows: <= 40 rows in step order; k_commit; failure_class "R" | "U";
    best_primitive: the best primitive overall (None if class U); n_decodes: decodes used."""
    cluster: int
    rows: List[BranchRow]
    k_commit: int
    failure_class: str
    best_primitive: Optional[Primitive] = None
    n_decodes: int = 0

    def render(self) -> str:
        """Plain-text table for the dossier (<= 40 rows)."""
        raise NotImplementedError


def save_states(version, xi: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(21, B, 2) states x_0..x_20 of the uninterrupted decode."""
    raise NotImplementedError


def restore_and_decode(version, state: FlowState, primitive: Optional[Primitive] = None) -> torch.Tensor:
    """Continue the decode from ``state`` to step 20, optionally applying ``primitive``.
    With no primitive, equals the uninterrupted decode bit for bit (5.4). Returns (B, 2)."""
    raise NotImplementedError


def renoise(version, state: FlowState, g: torch.Generator) -> FlowState:
    """Renoise ``state`` (see module doc) with one fresh eps (B, 2) from ``g``; same k and c."""
    raise NotImplementedError


def push(u: Sequence[float], alpha: float, k: int, h: int = PUSH_H) -> Primitive:
    """Build a push primitive; u is normalised."""
    raise NotImplementedError


def hop(tau: float, u: Sequence[float]) -> Primitive:
    """Build a hop primitive (k = 0); u is normalised."""
    raise NotImplementedError


def default_primitives(cell, cluster_members: np.ndarray, k: int) -> List[Primitive]:
    """8 directions x alpha in (0.5, 1, 2) x basin radius pushes at step k with h = 3, plus the
    toypref hop search exits (``toypref.cond.targets`` hop) for each member."""
    raise NotImplementedError


def wilson(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial share."""
    raise NotImplementedError


def k_commit(steps: Sequence[int], p: Sequence[float], thr: float = P_COMMIT) -> int:
    """Earliest sampled step after which every later p(k) < thr; else the last step."""
    raise NotImplementedError


def branch(cell, version, cluster, steps: Sequence[int] = DEFAULT_STEPS, M: int = M_BRANCH,
           primitives: Optional[List[Primitive]] = None, seed: int = 0) -> BranchTable:
    """The branch tool on all members of ``cluster`` (``toyrepair.miner.Cluster``).

    For each step: p(k) with M renoise branches per member (+64 where the Wilson interval
    contains 0.2). Then k_commit, then each primitive at k_commit and the two sampled steps
    before it; class R/U. Fresh noise comes from the mining pool stream. Counts decodes."""
    raise NotImplementedError
