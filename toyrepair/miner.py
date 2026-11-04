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

import math

import numpy as np

AGGLO_CUT = 0.7
N_RESIDUAL = 256
OPEN_THRESHOLD = 0.02
CARD_TOKEN_CAP = 150
ATLAS_TOKEN_CAP = 1500
N_EXEMPLARS = 3
HDBSCAN_MIN_CLUSTER_SIZE = 2     # smallest legal value: npm 1-4 cells have only 1-16 negatives
HDBSCAN_MIN_SAMPLES = 1
#: "hdbscan" = PLAN 6.2 (HDBSCAN, agglomerative fallback); "agglomerative" = the fallback rule only.
#: W1-A1 found HDBSCAN merges forgotten modes on k4 npm 16 (largest-cluster purity 0.33-0.52) while
#: average linkage at 0.7 keeps them apart (see docs/DONE_W1-A1.md). Change only before Phase 5.
CLUSTER_RULE = "hdbscan"
CHARS_PER_TOKEN = 3.5            # conservative estimate for the caps (W1-D measures the real ratio)


def est_tokens(text: str) -> int:
    """Token estimate used for the card and atlas caps."""
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


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
        """One line, <= 150 tokens."""
        kc = "?" if self.k_commit is None else str(self.k_commit)
        ex = ",".join(str(u) for u in self.exemplars)
        txt = (f"cluster {self.cluster}: n={self.n_members} r_ref={self.r_ref:.3f} r_cur={self.r_cur:.3f} "
               f"c=[{self.c_range[0]:.3f},{self.c_range[1]:.3f}] cohesion={self.cohesion:.2f} "
               f"k_commit={kc} class={self.failure_class} exemplars=[{ex}]")
        while est_tokens(txt) > CARD_TOKEN_CAP:       # cannot happen with 3 exemplars; kept as a guard
            txt = txt[: int(CARD_TOKEN_CAP * CHARS_PER_TOKEN) - 3] + "..."
        return txt


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

    def cluster_by_id(self, cid: int) -> Cluster:
        for c in self.clusters:
            if c.id == int(cid):
                return c
        raise KeyError(f"no cluster {cid}")

    def render(self) -> str:
        """Plain text, <= 1500 tokens. Cards beyond the cap are cut, and the cut count is stated."""
        n_closed = len(self.clusters) - len(self.cards)
        head = (f"atlas of version {self.version_id}: {len(self.clusters)} clusters of "
                f"{len(self.residual)} logged negatives; {len(self.cards)} open (residual basin mass >= "
                f"{OPEN_THRESHOLD}), {n_closed} closed")
        lines = [head]
        budget = ATLAS_TOKEN_CAP - est_tokens(head) - 12
        cut = 0
        for card in self.cards:
            t = card.render()
            if est_tokens(t) + 1 > budget:
                cut += 1
                continue
            lines.append(t)
            budget -= est_tokens(t) + 1
        if cut:
            lines.append(f"({cut} cards cut by the {ATLAS_TOKEN_CAP}-token cap; the smallest r_cur are cut)")
        return "\n".join(lines)


# ---------------------------------------------------------------- helpers
def _cache(cell) -> Dict:
    return cell.__dict__.setdefault("_w1a1_cache", {})


def _in_bad_many(cell, i: int, x) -> np.ndarray:
    return cell.in_bad_basin(i, x)


def affinity(cell) -> np.ndarray:
    """(N, N) float64 label-free affinity A under pi_ref (symmetric, diagonal 1).

    H[i, j] = 1[pi_ref(xi_j, c_i) in B_i]; A = (H + H^T) / 2 with the diagonal set to 1. Costs
    N^2 pi_ref decodes, counted under "v0"; cached on the cell."""
    import torch
    from . import policy
    cache = _cache(cell)
    if "affinity" in cache:
        return cache["affinity"]
    n = len(cell.neg)
    v0 = cache.setdefault("v0", policy.base_version(cell))
    H = np.zeros((n, n))
    xi = cell.neg.xi
    chunk = max(1, 65536 // max(n, 1))
    for s in range(0, n, chunk):
        rows = list(range(s, min(n, s + chunk)))
        X = xi.repeat(len(rows), 1)
        C = torch.cat([torch.full((n,), float(cell.neg.c[i])) for i in rows])
        act = policy.decode(v0, X, C).action
        for r, i in enumerate(rows):
            H[i] = _in_bad_many(cell, i, act[r * n:(r + 1) * n])
    A = 0.5 * (H + H.T)
    np.fill_diagonal(A, 1.0)
    cache["affinity"] = A
    cache["affinity_H"] = H
    return A


def cluster(A: np.ndarray, uids: np.ndarray) -> List[Cluster]:
    """HDBSCAN on 1 - A, fallback agglomerative (average, cut 0.7).

    HDBSCAN: ``sklearn.cluster.HDBSCAN(metric="precomputed", min_cluster_size=2,
    min_samples=1)``. Points HDBSCAN labels noise become singleton clusters (they are still
    failures to repair). "Fewer than 2 clusters" counts HDBSCAN's non-noise clusters. With
    N <= 2 negatives the agglomerative rule decides directly. Cluster ids are 0.. in order of
    each cluster's smallest uid (stable, label-free). ``Cluster.meta`` is not a field, so the
    method used is returned through ``cluster_method(A)``."""
    labels, _ = _cluster_labels(A)
    return _clusters_from_labels(labels, uids)


def cluster_method(A: np.ndarray) -> str:
    return _cluster_labels(A)[1]


def _cluster_labels(A: np.ndarray):
    n = A.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int), "single"
    D = np.clip(1.0 - A, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    if n >= 3 and CLUSTER_RULE == "hdbscan":
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(metric="precomputed", min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                      min_samples=HDBSCAN_MIN_SAMPLES, copy=True).fit_predict(D)
        if len(set(lab[lab >= 0].tolist())) >= 2:
            nxt = lab.max() + 1
            lab = lab.copy()
            for j in np.flatnonzero(lab < 0):
                lab[j] = nxt
                nxt += 1
            return lab, "hdbscan"
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                  distance_threshold=AGGLO_CUT).fit_predict(D)
    return lab, "agglomerative"


def _clusters_from_labels(labels: np.ndarray, uids: np.ndarray) -> List[Cluster]:
    uids = np.asarray(uids)
    groups = {}
    for row, l in enumerate(labels):
        groups.setdefault(int(l), []).append(row)
    ordered = sorted(groups.values(), key=lambda rows: int(uids[rows].min()))
    out = []
    for cid, rows in enumerate(ordered):
        rows = np.array(sorted(rows, key=lambda r: int(uids[r])), dtype=int)
        out.append(Cluster(id=cid, members=rows, uids=uids[rows].astype(np.int64)))
    return out


def residual_mass(cell, version, rows: Optional[np.ndarray] = None, n: int = N_RESIDUAL) -> np.ndarray:
    """(len(rows),) r_i(version): share of n mining-pool noises at c_i decoded into B_i.

    The n noises are ``pools.noise(cell, MINING, n, stream=1)``, the same for every negative and
    every version (common random numbers). Counts len(rows) * n decodes."""
    import torch
    from . import policy, pools
    rows = np.arange(len(cell.neg)) if rows is None else np.asarray(rows, dtype=int)
    z = pools.noise(cell, pools.MINING, n, pools.STREAM_RESIDUAL)
    out = np.zeros(len(rows))
    chunk = max(1, 65536 // n)
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        X = z.repeat(len(part), 1)
        C = torch.cat([torch.full((n,), float(cell.neg.c[i])) for i in part])
        act = policy.decode(version, X, C).action
        for r, i in enumerate(part):
            out[s + r] = float(cell.in_bad_basin(int(i), act[r * n:(r + 1) * n]).mean())
    return out


def _purity_adapter(cell, uids) -> float:
    """LOCAL ADAPTER until W1-A2's ``oracle.purity`` is merged (docs/INTERFACE_REQUESTS_W1-A1.md):
    the same definition, majority-mode share of the members' oracle modes. Used only for the
    purity CSV."""
    rows = [int(np.flatnonzero(cell.neg.uid == int(u))[0]) for u in uids]
    modes = np.asarray(cell._oracle.neg_modes)[rows]       # noqa: SLF001 (adapter; replaced by oracle.purity)
    _, cnt = np.unique(modes, return_counts=True)
    return float(cnt.max() / len(modes))


def purity_of(cell, uids) -> float:
    """``toyrepair.oracle.purity`` (lazy import), or the local adapter while it is a stub."""
    from . import oracle                                    # lazy: never on an LLM-facing path
    try:
        return float(oracle.purity(cell, list(map(int, uids))))
    except NotImplementedError:
        return _purity_adapter(cell, uids)


def log_purity(cell, atlas: Atlas, path: str) -> None:
    """Append one CSV row per cluster (version, cluster id, n, purity) using
    ``toyrepair.oracle.purity`` (lazy import, never on an LLM-facing path).

    Columns: cell, version, cluster, n, purity, open (1/0)."""
    import csv
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    open_ids = set(atlas.open_ids())
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["cell", "version", "cluster", "n", "purity", "open"])
        for cl in atlas.clusters:
            w.writerow([cell.key, atlas.version_id, cl.id, len(cl.members), f"{purity_of(cell, cl.uids):.4f}",
                        int(cl.id in open_ids)])


def clusters_of(cell) -> List[Cluster]:
    """pi_ref clusters of the cell (cached): affinity, then ``cluster``."""
    cache = _cache(cell)
    if "clusters" not in cache:
        A = affinity(cell)
        labels, method = _cluster_labels(A)
        cache["clusters"] = _clusters_from_labels(labels, cell.neg.uid)
        cache["cluster_method"] = method
    return cache["clusters"]


def residual_ref(cell) -> np.ndarray:
    """(N,) r_i under pi_ref (cached)."""
    from . import policy
    cache = _cache(cell)
    if "r_ref" not in cache:
        v0 = cache.setdefault("v0", policy.base_version(cell))
        cache["r_ref"] = residual_mass(cell, v0)
    return cache["r_ref"]


def mine(cell, version, branch_tables: Optional[Dict[int, object]] = None) -> Atlas:
    """Mine the cell's logged negatives under ``version``. Clusters come from pi_ref's affinity
    (fixed per cell); r_cur is recomputed per version. k_commit / class come from
    ``branch_tables`` when given.

    Exemplars: the 3 members with the highest mean affinity to their cluster (ties: lower uid).
    ``meta``: cluster_method, n_clusters, n_open, decodes (this call)."""
    from . import policy
    t0 = policy.counter.total
    A = affinity(cell)
    cls = clusters_of(cell)
    r_ref = residual_ref(cell)
    r_cur = r_ref if version.id == "v0" and not version.patches else residual_mass(cell, version)
    cards = []
    c = cell.neg.c.numpy()
    for cl in cls:
        m = cl.members
        rc = float(r_cur[m].mean())
        if rc < OPEN_THRESHOLD:
            continue
        sub = A[np.ix_(m, m)]
        coh = float((sub.sum() - len(m)) / (len(m) * (len(m) - 1))) if len(m) > 1 else 1.0
        cent = sub.mean(1)
        order = sorted(range(len(m)), key=lambda j: (-cent[j], int(cl.uids[j])))
        bt = (branch_tables or {}).get(cl.id)
        cards.append(FailureCard(
            cluster=cl.id, n_members=len(m), r_ref=float(r_ref[m].mean()), r_cur=rc,
            c_range=(float(np.percentile(c[m], 5)), float(np.percentile(c[m], 95))), cohesion=coh,
            k_commit=None if bt is None else int(bt.k_commit),
            failure_class="?" if bt is None else str(bt.failure_class),
            exemplars=tuple(int(cl.uids[j]) for j in order[:N_EXEMPLARS])))
    cards.sort(key=lambda cd: (-cd.r_cur, cd.cluster))
    return Atlas(version_id=version.id, clusters=cls, cards=cards, affinity=A, residual=np.asarray(r_cur),
                 meta={"cluster_method": _cache(cell).get("cluster_method"), "n_clusters": len(cls),
                       "n_open": len(cards), "decodes": policy.counter.total - t0})
