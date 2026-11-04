"""Label-free behaviour tracks across the condition (Phase 4, W2b). Used by ``contrast`` and ``unit_card``.

Adapted from the exploration sessions (E3 ``explore/E3/tracks.py``, E1 ``common.build_tracker``): a
behaviour of pi_ref moves continuously in action space as c changes. So:

1. decode a fixed set of ``N_DEC`` noises with pi_ref at each of ``N_GRID`` conditions in [0, 1];
2. at each condition, cluster the typical decodes (5th-NN distance at or below its 90th percentile)
   by single linkage at 3x the median 1-NN spacing (toypref's basin rule); drop clusters under 2%;
3. link each cluster to its mutual nearest cluster (by centroid) at the next grid condition;
4. the connected components of the links are the tracks.

A logged negative (c_i, a_i) marks the track of a_i at c_i. A cluster's tracks are the tracks its
members' logged actions fall in. ``assign(a, c)`` gives the track of any (action, c): the track of
the nearest typical pi_ref decode at the nearest grid condition, if it is within ``eps_mult`` x eps.

Inputs: pi_ref decodes (network inputs and outputs) and the logged negatives only. No oracle, no
labels, no ring geometry, no mode count. The surgeon may compute the same thing in the sandbox
(PLAN 7.3: discovering behaviours by sampling pi_ref is allowed). Build cost: N_GRID x N_DEC
decodes of pi_ref (51,712), about 20 s on one thread; the result is cached per cell.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np
import torch

TYP_K, TYP_Q, EPS_MULT, MIN_SHARE = 5, 0.9, 3.0, 0.02
N_GRID, N_DEC = 101, 512
#: radius multipliers: members mark tracks at 3x eps (E3); samples are assigned at 2x eps (E1)
MARK_EPS, ASSIGN_EPS = 3.0, 2.0
#: noise stream of the track decodes: mining pool (1), sub-stream 9 (documented here, PLAN 5.3)
TRACK_STREAM = 9
VERSION = 1


def _components(adj: np.ndarray) -> (int, np.ndarray):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    return connected_components(csr_matrix(adj), directed=False)


def _cluster_level(dec: torch.Tensor):
    d = torch.cdist(dec, dec)
    d.fill_diagonal_(float("inf"))
    kth = d.kthvalue(TYP_K, dim=1).values
    typ = (kth <= torch.quantile(kth, TYP_Q)).numpy()
    pts = dec[typ]
    dp = torch.cdist(pts, pts)
    dp.fill_diagonal_(float("inf"))
    eps = EPS_MULT * float(dp.min(1).values.median())
    n, lab = _components((dp <= eps).numpy())
    sizes = np.bincount(lab, minlength=n)
    keep = sizes >= max(3, int(MIN_SHARE * len(pts)))
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(int(keep.sum()))
    lab = remap[lab]
    ok = lab >= 0
    pts, lab = pts[ok], lab[ok]
    cents = torch.stack([pts[lab == j].mean(0) for j in range(int(keep.sum()))])
    return pts, lab, cents, eps


@dataclass
class Tracks:
    grid: np.ndarray                 # (N_GRID,) conditions
    levels: list                     # per grid point: (pts (P,2), lab (P,), centroids (m,2), eps)
    offs: np.ndarray                 # cluster offsets per level
    track: np.ndarray                # global cluster index -> track id
    n_tracks: int
    version: int = VERSION

    def assign(self, a: torch.Tensor, c: torch.Tensor, eps_mult: float = ASSIGN_EPS) -> np.ndarray:
        """(B,) track id of each (a, c), or -1 if a is not near a typical pi_ref decode at the nearest grid c."""
        a = torch.as_tensor(a, dtype=torch.float32).reshape(-1, 2)
        cn = torch.as_tensor(c, dtype=torch.float32).reshape(-1).numpy()
        gi = np.clip(np.rint(cn * (len(self.grid) - 1)).astype(int), 0, len(self.grid) - 1)
        out = -np.ones(a.shape[0], dtype=int)
        for k in np.unique(gi):
            sel = np.flatnonzero(gi == k)
            pts, lab, _, eps = self.levels[k]
            d = torch.cdist(a[sel], pts)
            dmin, j = d.min(1)
            hit = (dmin <= eps_mult * eps).numpy()
            out[sel[hit]] = self.track[self.offs[k] + lab[j.numpy()[hit]]]
        return out

    def marked(self, cell, uids: Sequence[int]) -> Set[int]:
        """Tracks marked by the logged actions of the negatives ``uids``."""
        pos = {int(u): i for i, u in enumerate(np.asarray(cell.neg.uid))}
        rows = [pos[int(u)] for u in uids if int(u) in pos]
        if not rows:
            return set()
        t = self.assign(cell.neg.a[rows], cell.neg.c[rows], eps_mult=MARK_EPS)
        return set(int(x) for x in t if x >= 0)


def build(cell) -> Tracks:
    from . import policy, pools
    base = policy.base_version(cell)
    g = pools.torch_generator(cell, pools.MINING, TRACK_STREAM)
    z = torch.randn(N_DEC, 2, generator=g)
    grid = np.linspace(0.0, 1.0, N_GRID)
    zz = z.repeat(N_GRID, 1)
    cc = torch.from_numpy(np.repeat(grid, N_DEC).astype(np.float32))
    with torch.no_grad():
        act = policy.decode(base, zz, cc, count=False).action
    levels = [_cluster_level(act[i * N_DEC:(i + 1) * N_DEC]) for i in range(N_GRID)]
    offs = np.cumsum([0] + [len(L[2]) for L in levels])
    rows, cols = [], []
    for gi in range(N_GRID - 1):
        d = torch.cdist(levels[gi][2], levels[gi + 1][2])
        ab, ba = d.argmin(1).numpy(), d.argmin(0).numpy()
        for i, j in enumerate(ab):
            if ba[j] == i:
                rows.append(offs[gi] + i)
                cols.append(offs[gi + 1] + j)
    N = int(offs[-1])
    adj = np.zeros((N, N), dtype=bool)
    adj[rows, cols] = True
    n, track = _components(adj)
    return Tracks(grid=grid, levels=levels, offs=offs, track=track, n_tracks=int(n))


def get(cell, cache_dir: Optional[str] = None) -> Tracks:
    """``build(cell)``, cached in ``$TOYPREF_RUNS/cells/<cell key>/tracks.pkl`` (mode-700 repo)."""
    if cache_dir is None:
        cache_dir = os.path.join(os.environ["TOYPREF_RUNS"], "cells", cell.key)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tracks.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                tr = pickle.load(fh)
            if getattr(tr, "version", 0) == VERSION:
                return tr
        except Exception:
            pass
    tr = build(cell)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(tr, fh)
    os.replace(tmp, path)
    return tr


def target_labels(tr: Tracks, cell, uids: Sequence[int], a: torch.Tensor, c: torch.Tensor) -> np.ndarray:
    """(B,) bool: action a at condition c lies in one of the tracks the cluster's members mark."""
    marked = tr.marked(cell, uids)
    if not marked:
        return np.zeros(len(a), dtype=bool)
    return np.isin(tr.assign(a, c), list(marked))


def track_shares(tr: Tracks, a: torch.Tensor, c: torch.Tensor) -> List[float]:
    """Share of each track among (a, c) (index = track id), plus the unassigned share last."""
    t = tr.assign(a, c)
    h = np.bincount(t[t >= 0], minlength=tr.n_tracks).astype(float)
    return list(h / max(len(t), 1)) + [float((t < 0).mean())]
