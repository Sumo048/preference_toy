"""TRKNG: no-grid mask builders from pi_ref's own activations.

B6 (explore/E3/tracks.py) clusters pi_ref decodes at 101 grid values of the condition c and links the
clusters across neighbouring c. A real robot condition cannot be gridded. The builders here never use c
as a coordinate, a grid or a feature, and never use the ring geometry or CondMixture. They see only:
  - pi_ref decodes: for a noise xi and a condition c drawn by pi_ref's sampler, the hidden activations of
    pi_ref (output of blocks.2, the residual stream before out_norm) at chosen Euler steps. c enters only
    as the network's own input, so the activations encode it implicitly.
  - the logged negatives (c_i, xi_i, a_i), decoded from their own (xi_i, c_i) the same way.

Builders (each returns, for every sample in the B6 training pool, a cluster label and a flag):
  TA  k-means on step-0 activations of N decodes, k in {8,..,64} chosen by silhouette; a negative flags
      its nearest centroid.
  TB  activations at steps 0, 4, 8 concatenated, PCA to 32 dims, HDBSCAN; a negative flags the cluster of
      its nearest clustered fit point. HDBSCAN noise points are unassigned (dropped from the kept set,
      as B6 drops samples outside every track).
  TC  kNN graph over the TB features of the N fit decodes; edges kept if shorter than a threshold tau
      set from the negatives' own nearest-neighbour distances; connected components that contain a
      fit point within tau of a negative are bad. Pool samples farther than tau from every fit point
      are unassigned (dropped).

The oracle (CondMixture.assign, via pool.mode) is used only by score() for evaluation.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

LAYER = 2


@torch.no_grad()
def activations(ref, xi, c, steps=(0, 4, 8), batch=65536):
    """pi_ref's Euler decode of (xi, c), recording blocks[LAYER]'s output at the given steps.
    -> (dict step -> (n, hidden) float32 numpy, final actions (n, 2) tensor). Same op order as CondFlow.sample."""
    net = ref.net
    K = ref.n_euler_steps
    dt = 1.0 / K
    rec = {s: [] for s in steps}
    outs = []
    store = {}

    def hook(_m, _i, o):
        store["h"] = o

    hd = net.blocks[LAYER].register_forward_hook(hook)
    try:
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float()
            cc = c[i:i + batch].float()
            for k in range(K):
                t = torch.full((x.shape[0],), 1.0 - k * dt)
                x = x - dt * net(x, t, cc)
                if k in rec:
                    rec[k].append(store["h"].numpy().astype(np.float32))
            outs.append(x)
    finally:
        hd.remove()
    return {s: np.concatenate(v) for s, v in rec.items()}, torch.cat(outs)


def feat(acts, steps):
    return np.concatenate([acts[s] for s in steps], 1)


# ------------------------------------------------------------------ builders
def build_TA(F_fit, F_pool, F_neg, seed, ks=(8, 12, 16, 24, 32, 48, 64), n_sil=4000, n_init=3):
    rng = np.random.default_rng(seed)
    sub = rng.choice(F_fit.shape[0], min(n_sil, F_fit.shape[0]), replace=False)
    best = None
    sils = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(F_fit)
        s = float(silhouette_score(F_fit[sub], km.labels_[sub]))
        sils[k] = s
        if best is None or s > best[0]:
            best = (s, k, km)
    _, k, km = best
    lab_pool = km.predict(F_pool)
    lab_neg = km.predict(F_neg)
    bad = sorted(set(int(x) for x in lab_neg))
    return {"lab": lab_pool, "flag": np.isin(lab_pool, bad), "assigned": np.ones(len(lab_pool), bool),
            "info": {"k": int(k), "sil": best[0], "sils": sils, "n_clusters": int(k), "n_bad": len(bad)}}


def pca_fit(F_fit, seed, dim=32):
    return PCA(n_components=dim, random_state=seed).fit(F_fit)


def build_TB(Z_fit, Z_pool, Z_neg, seed, mcs_frac=0.01, min_samples=10):
    mcs = max(10, int(mcs_frac * Z_fit.shape[0]))
    hb = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, copy=True).fit(Z_fit)
    lab_fit = hb.labels_
    nn_all = NearestNeighbors(n_neighbors=1).fit(Z_fit)
    j = nn_all.kneighbors(Z_pool, return_distance=False)[:, 0]
    lab_pool = lab_fit[j]
    ok = lab_fit >= 0
    if ok.any():
        nn_ok = NearestNeighbors(n_neighbors=1).fit(Z_fit[ok])
        lab_neg = lab_fit[ok][nn_ok.kneighbors(Z_neg, return_distance=False)[:, 0]]
    else:
        lab_neg = np.array([], int)
    bad = sorted(set(int(x) for x in lab_neg))
    assigned = lab_pool >= 0
    return {"lab": lab_pool, "flag": np.isin(lab_pool, bad) & assigned, "assigned": assigned,
            "info": {"mcs": mcs, "n_clusters": int(lab_fit.max() + 1), "n_bad": len(bad),
                     "noise_fit": float((lab_fit < 0).mean())}}


def build_TC(Z_fit, Z_pool, Z_neg, seed, knn=10, tau_mult=1.5):
    nn = NearestNeighbors(n_neighbors=knn + 1).fit(Z_fit)
    d, j = nn.kneighbors(Z_fit)
    d, j = d[:, 1:], j[:, 1:]
    # threshold from the negatives' own neighbour distances: median distance from a negative to its
    # knn-th nearest fit decode, times tau_mult
    dn, jn = nn.kneighbors(Z_neg, n_neighbors=knn)
    tau = tau_mult * float(np.median(dn[:, -1]))
    n = Z_fit.shape[0]
    m = d <= tau
    rows = np.repeat(np.arange(n), knn)[m.ravel()]
    cols = j.ravel()[m.ravel()]
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncomp, comp = connected_components(A, directed=False)
    bad = sorted(set(int(comp[x]) for x, dd in zip(jn.ravel(), dn.ravel()) if dd <= tau))
    dp, jp = nn.kneighbors(Z_pool, n_neighbors=1)
    assigned = dp[:, 0] <= tau
    lab_pool = np.where(assigned, comp[jp[:, 0]], -1)
    sizes = np.bincount(comp, minlength=ncomp)
    return {"lab": lab_pool, "flag": np.isin(lab_pool, bad) & assigned, "assigned": assigned,
            "info": {"tau": tau, "n_clusters": int((sizes >= max(3, 0.01 * n)).sum()), "n_comp": int(ncomp),
                     "n_bad": len(bad), "bad_mass_fit": float(np.isin(comp, bad).mean())}}


# ------------------------------------------------------------------ checks and scoring
def keep_of(m):
    return m["assigned"] & ~m["flag"]


def ari(l1, l2):
    return float(adjusted_rand_score(l1, l2))


def jacc(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def balance(keep, ref_cells, n_cells):
    """Label-free: kept-set share over a fixed fine partition (k-means 64 on the pool's TB features) vs
    pi_ref's share renormalised over the cells the mask mostly keeps. 1 = whole cells kept or dropped,
    lower = the mask cuts through cells and keeps them unevenly."""
    tot = np.bincount(ref_cells, minlength=n_cells).astype(float)
    kept = np.bincount(ref_cells[keep], minlength=n_cells).astype(float)
    frac = np.where(tot > 0, kept / np.maximum(tot, 1), 0)
    cells = frac > 0.5
    if kept.sum() == 0 or not cells.any():
        return 0.0
    p = kept / kept.sum()
    q = np.where(cells, tot, 0) / tot[cells].sum()
    return float(1 - 0.5 * np.abs(p - q).sum())


def score(m, bad_oracle):
    """ORACLE (evaluation only): flag precision/recall against forget-mode membership of pool samples."""
    flag, keep = m["flag"], keep_of(m)
    return {"prec": float(bad_oracle[flag].mean()) if flag.any() else float("nan"),
            "rec": float(flag[bad_oracle].mean()) if bad_oracle.any() else float("nan"),
            "keep": int(keep.sum()), "keep_frac_bad": float(bad_oracle[keep].mean()) if keep.any() else float("nan"),
            "unassigned": float((~m["assigned"]).mean()), "flag_frac": float(flag.mean())}


def typical(Z_fit, Z_pool, q=0.9, k=5):
    """Label-free typicality (B6's rule without the c-grid): a pool sample is typical if its distance to
    its k-th nearest fit decode is at most the q-quantile of the fit decodes' own k-th-neighbour distances.
    A pool sample that is itself a fit decode (distance 0) skips itself."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z_fit)
    df, _ = nn.kneighbors(Z_fit)
    thr = float(np.quantile(df[:, k], q))
    dp, _ = nn.kneighbors(Z_pool)
    dk = np.where(dp[:, 0] == 0, dp[:, k], dp[:, k - 1])
    return dk <= thr
