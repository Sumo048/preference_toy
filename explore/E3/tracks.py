"""Label-free behaviour tracks across the condition (E3).

Idea: a behaviour moves continuously in action space as c changes. So cluster pi_ref's own decodes
on a fine grid of c (single linkage at a self-calibrated radius, as toypref's basins do), link each
cluster to the nearest cluster at the next grid point, and union the links into TRACKS. A logged
negative (c_i, a_i) marks the track of a_i's cluster at c_i as bad. A sample (c, a) is flagged bad
if a falls in a bad track's cluster at the grid point nearest c.

Inputs used: pi_ref decodes (network inputs and outputs only), the logged negatives. No labels,
no ring geometry, no mode count. The oracle is used only for the purity diagnostic.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

TYP_K, TYP_Q, EPS_MULT = 5, 0.9, 3.0


def cluster_at(ref, c, z):
    dec = ref.sample(z, torch.full((z.shape[0],), float(c)))
    d = torch.cdist(dec, dec)
    d.fill_diagonal_(float("inf"))
    kth = d.kthvalue(TYP_K, dim=1).values
    thr = float(torch.quantile(kth, TYP_Q))
    typ = (kth <= thr).numpy()
    pts = dec[typ]
    dp = torch.cdist(pts, pts)
    dp.fill_diagonal_(float("inf"))
    eps = EPS_MULT * float(dp.min(1).values.median())
    n, lab = connected_components(csr_matrix((dp <= eps).numpy()), directed=False)
    # drop tiny clusters (< 2% of points): stray links, not behaviours
    sizes = np.bincount(lab, minlength=n)
    keep = sizes >= max(3, int(0.02 * len(pts)))
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(keep.sum())
    lab = remap[lab]
    ok = lab >= 0
    pts, lab = pts[ok], lab[ok]
    cents = torch.stack([pts[lab == j].mean(0) for j in range(keep.sum())])
    return pts, lab, cents, eps


class Tracks:
    def __init__(self, ref, seed, n_grid=101, n_dec=512):
        g = torch.Generator().manual_seed(int(seed) * 7_777 + 4242)
        z = torch.randn(n_dec, 2, generator=g)
        self.grid = np.linspace(0.0, 1.0, n_grid)
        self.levels = [cluster_at(ref, c, z) for c in self.grid]
        offs = np.cumsum([0] + [len(L[2]) for L in self.levels])
        self.offs = offs
        rows, cols = [], []
        for gi in range(n_grid - 1):
            ca, cb = self.levels[gi][2], self.levels[gi + 1][2]
            d = torch.cdist(ca, cb)
            # mutual nearest neighbours between adjacent grid levels
            ab, ba = d.argmin(1).numpy(), d.argmin(0).numpy()
            for i, j in enumerate(ab):
                if ba[j] == i:
                    rows.append(offs[gi] + i)
                    cols.append(offs[gi + 1] + j)
        N = offs[-1]
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        self.n_tracks, self.track = connected_components(A, directed=False)
        self.n_clusters = [len(L[2]) for L in self.levels]

    def assign(self, a: torch.Tensor, c: torch.Tensor, eps_mult: float = 1.0) -> np.ndarray:
        """track id per (a, c), or -1 if a is not within eps_mult*eps of a typical decode at the nearest grid c."""
        gi = np.clip(np.rint(c.numpy() * (len(self.grid) - 1)).astype(int), 0, len(self.grid) - 1)
        out = -np.ones(a.shape[0], dtype=int)
        for k in np.unique(gi):
            sel = np.flatnonzero(gi == k)
            pts, lab, _, eps = self.levels[k]
            d = torch.cdist(a[sel], pts)
            j = d.argmin(1).numpy()
            hit = (d.min(1).values <= eps_mult * eps).numpy()
            out[sel[hit]] = self.track[self.offs[k] + lab[j[hit]]]
        return out


def bad_tracks(tr: Tracks, c_neg, a_neg):
    t = tr.assign(a_neg, c_neg, eps_mult=3.0)
    return sorted(set(int(x) for x in t if x >= 0)), float((t < 0).mean())
