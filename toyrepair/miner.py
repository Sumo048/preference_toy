"""Failure miner: clusters of logged negatives and the atlas (docs/PLAN.md 6.2).

Affinity (label-free): A_ij = 1/2 (1[f_ref(xi_j, c_i) in B_i] + 1[f_ref(xi_i, c_j) in B_j]).
Clustering on 1 - A with HDBSCAN (sklearn.cluster.HDBSCAN, metric "precomputed"); if it gives
fewer than 2 clusters, agglomerative clustering with average linkage cut at distance 0.7.
Residual basin mass r_i(pi): share of 256 fresh noises at c_i that pi decodes into B_i.
A cluster is open while its mean r_i >= 0.02.
Purity (oracle) goes to ``runs/<run>/purity.csv`` and the ledger only, never to a card, via
``toyrepair.oracle.purity(cell, cluster.uids)`` (``log_purity`` below).

Owner: W1-A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

AGGLO_CUT = 0.7
N_RESIDUAL = 256
OPEN_THRESHOLD = 0.02
CARD_TOKEN_CAP = 150
ATLAS_TOKEN_CAP = 1500
N_EXEMPLARS = 3


@dataclass
class Cluster:
    """id: cluster id (stable within one atlas); members: (m,) int row positions into
    ``cell.neg``; uids: (m,) int64 negative uids."""
    id: int
    members: np.ndarray
    uids: np.ndarray


@dataclass
class FailureCard:
    """One open cluster, <= 150 tokens when rendered.

    n_members; r_ref / r_cur: mean residual basin mass under pi_ref / current version (share
    in [0, 1]); c_range: (5th, 95th) percentile of member conditions; cohesion: mean A within
    the cluster; k_commit and failure_class ("R" | "U" | "?" before the branch tool ran);
    exemplars: up to 3 negative uids."""
    cluster: int
    n_members: int
    r_ref: float
    r_cur: float
    c_range: Tuple[float, float]
    cohesion: float
    k_commit: Optional[int]
    failure_class: str
    exemplars: Tuple[int, ...]

    def render(self) -> str:
        raise NotImplementedError


@dataclass
class Atlas:
    """The failure cards of one version.

    version_id; clusters: all clusters (open and closed); cards: one per open cluster, sorted
    by r_cur descending; affinity: (N, N) float A; residual: (N,) r_i under the version."""
    version_id: str
    clusters: List[Cluster]
    cards: List[FailureCard]
    affinity: np.ndarray
    residual: np.ndarray
    meta: Dict = field(default_factory=dict)

    def open_ids(self) -> List[int]:
        return [c.cluster for c in self.cards]

    def render(self) -> str:
        """Plain text, <= 1500 tokens."""
        raise NotImplementedError


def affinity(cell) -> np.ndarray:
    """(N, N) float64 label-free affinity A under pi_ref (symmetric, diagonal 1)."""
    raise NotImplementedError


def cluster(A: np.ndarray, uids: np.ndarray) -> List[Cluster]:
    """HDBSCAN on 1 - A, fallback agglomerative (average, cut 0.7)."""
    raise NotImplementedError


def residual_mass(cell, version, rows: Optional[np.ndarray] = None, n: int = N_RESIDUAL) -> np.ndarray:
    """(len(rows),) r_i(version): share of n mining-pool noises at c_i decoded into B_i."""
    raise NotImplementedError


def log_purity(cell, atlas: Atlas, path: str) -> None:
    """Append one CSV row per cluster (version, cluster id, n, purity) using
    ``toyrepair.oracle.purity`` (lazy import, never on an LLM-facing path)."""
    raise NotImplementedError


def mine(cell, version, branch_tables: Optional[Dict[int, object]] = None) -> Atlas:
    """Mine the cell's logged negatives under ``version``. Clusters come from pi_ref's affinity
    (fixed per cell); r_cur is recomputed per version. k_commit / class come from
    ``branch_tables`` when given."""
    raise NotImplementedError
