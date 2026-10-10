"""Behaviour tracking across logged conditions (round 4, docs/NOVEL_PREREG.md).

All methods here use ONLY: the logged negatives (c_i, a_i [, xi_i]), forward decodes of pi_ref at
logged conditions (the 4096 unlabelled anchor-pool conditions + this cell's negative conditions),
and the unlabelled pool samples. The oracle labeller is used ONLY in `diagnostics`.

Tracker (TRACK, judge's frozen spec #1):
  1. nodes V = pool conditions + negative conditions; one shared noise bank Z (M draws);
     D_u[m] = f_ref(z_m, c_u).
  2. per node: typical = 5-NN distance <= node q90; single linkage on typical points, cut at the
     largest log-gap among the top-20 merge heights; negatives' actions are inserted before clustering.
  3. graph: delta(u,v) = median_m ||D_u[m] - D_v[m]||; edges = mutual-kNN(k=8) U MST (or the
     sorted-c chain for the ablation).
  4. link (u,q)-(v,r) iff mutual argmax of the bank-overlap O_uv[q,r] and O >= half the typical size
     on both sides; union-find by decreasing O, refusing merges that would put two clusters of one
     node in one component (refusals counted as a label-free alarm).
  5. seeds K_bad = components of the negatives' clusters (OR), or the enrichment rule (RTRACK).
  6. a pool sample is good iff its cluster (nearest typical bank decode within the cut) is in a
     component outside K_bad; atypical/unassigned samples are excluded from both sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

K_TYP, Q_TYP, TOP_MERGES, K_GRAPH = 5, 0.9, 20, 8


def gen(seed, purpose):
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + purpose)


P_BANK1, P_BANK2, P_JOINT, P_PRA = 80_000, 80_001, 80_002, 80_003


def cluster_node(dec: np.ndarray):
    """(labels for every row, -1 = atypical/unassigned; n_clusters; cut height; typical mask)."""
    d = np.linalg.norm(dec[:, None] - dec[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    kth = np.sort(d, axis=1)[:, K_TYP - 1]
    typ = kth <= np.quantile(kth, Q_TYP)
    pts = dec[typ]
    if len(pts) < 3:
        return -np.ones(len(dec), int), 0, 0.0, typ
    Zl = linkage(pts, "single")
    h = np.sort(Zl[:, 2])[-TOP_MERGES:]
    g = np.diff(np.log(h + 1e-9))
    k = int(np.argmax(g))
    cut = float(np.sqrt(h[k] * h[k + 1]))
    lab = fcluster(Zl, cut, criterion="distance") - 1
    full = -np.ones(len(dec), int)
    full[typ] = lab                      # spec #1 step 2: atypical points get -1 (review fix 2)
    return full, int(lab.max() + 1), cut, typ


@dataclass
class Tracker:
    V: int
    node_c: np.ndarray            # (V,)
    labels: np.ndarray            # (V, M[+1]) cluster per bank decode (and inserted negative)
    pts: List[np.ndarray]         # typical bank decodes per node (for assignment)
    pts_lab: List[np.ndarray]
    cut: np.ndarray               # (V,)
    comp: Dict                    # (node, cluster) -> component id
    delta: Optional[np.ndarray]
    diag: Dict = field(default_factory=dict)
    bank: Optional[List[np.ndarray]] = None   # full cloud per node (bank decodes [+ inserted negatives])
    thr: Optional[np.ndarray] = None          # per-node q90 of the 5-NN distance (typicality)

    def typical(self, u: int, a: np.ndarray) -> np.ndarray:
        """spec #1 step 6 (review fix 1): 5-NN distance to the node's cloud <= the node's q90."""
        d = np.sort(np.linalg.norm(a[:, None] - self.bank[u][None], axis=-1), axis=1)[:, K_TYP - 1]
        return d <= self.thr[u]

    def assign(self, u: int, a: np.ndarray) -> np.ndarray:
        """component of actions `a` (n,2) at node u; -1 if atypical or unassigned."""
        P = self.pts[u]
        if len(P) == 0:
            return -np.ones(len(a), int)
        dd = np.linalg.norm(a[:, None] - P[None], axis=-1)
        j = dd.argmin(1)
        out = np.array([self.comp.get((u, int(self.pts_lab[u][jj])), -1) for jj in j])
        out[dd.min(1) > self.cut[u]] = -1
        out[~self.typical(u, a)] = -1
        return out


def _decode_bank(ref, node_c, Z, chunk=262_144):
    V, M = len(node_c), Z.shape[0]
    zz = Z.repeat(V, 1)
    cc = torch.from_numpy(np.repeat(node_c, M).astype(np.float32))
    out = []
    for s in range(0, zz.shape[0], chunk):
        out.append(ref.sample(zz[s:s + chunk], cc[s:s + chunk]))
    return torch.cat(out).reshape(V, M, 2).numpy()


def _delta(D: np.ndarray, chunk=24):
    V = D.shape[0]
    Dt = torch.from_numpy(D)
    out = np.empty((V, V), np.float32)
    for s in range(0, V, chunk):
        d = torch.linalg.norm(Dt[s:s + chunk, None] - Dt[None], dim=-1)    # (c, V, M)
        out[s:s + chunk] = d.median(dim=-1).values.numpy()
    np.fill_diagonal(out, np.inf)
    return out


def build_tracker(ref, seed, node_c: np.ndarray, neg_node: np.ndarray, neg_a: np.ndarray, M=128,
                  bank=P_BANK1, graph="knn") -> (Tracker, np.ndarray):
    """Returns the tracker and the component of every negative (-1 if unseeded)."""
    V = len(node_c)
    Z = torch.randn(M, 2, generator=gen(seed, bank))
    D = _decode_bank(ref, node_c, Z)
    labels = -np.ones((V, M + 1), int)
    pts, pts_lab, cuts, sizes = [], [], np.zeros(V), []
    thrs, clouds = np.zeros(V), []
    neg_at = {}
    for i, u in enumerate(neg_node):
        neg_at.setdefault(int(u), []).append(i)
    neg_lab = -np.ones(len(neg_node), int)
    for u in range(V):
        dec = D[u]
        extra = neg_at.get(u, [])
        if extra:
            dec = np.concatenate([dec, neg_a[extra]])
        lab, ncl, cut, typ = cluster_node(dec)
        dfull = np.linalg.norm(dec[:, None] - dec[None], axis=-1)
        np.fill_diagonal(dfull, np.inf)
        thrs[u] = float(np.quantile(np.sort(dfull, axis=1)[:, K_TYP - 1], Q_TYP))
        clouds.append(dec)
        labels[u, :M] = lab[:M]
        for t, i in enumerate(extra):
            neg_lab[i] = lab[M + t]
        P = dec[:M][typ[:M]]
        pts.append(P)
        pts_lab.append(lab[:M][typ[:M]])
        cuts[u] = cut
        sizes.append(np.bincount(lab[:M][lab[:M] >= 0], minlength=max(ncl, 1)))
    # ---- graph
    delta = _delta(D)
    if graph == "sortc":
        o = np.argsort(node_c)
        edges = set((int(min(a, b)), int(max(a, b))) for a, b in zip(o[:-1], o[1:]))
    else:
        nn = np.argsort(delta, axis=1)[:, :K_GRAPH]
        nnset = [set(r) for r in nn]
        edges = set()
        for u in range(V):
            for v in nn[u]:
                if u in nnset[v]:
                    edges.add((min(u, int(v)), max(u, int(v))))
        dm = delta.copy()
        dm[~np.isfinite(dm)] = 0
        mst = minimum_spanning_tree(dm).tocoo()
        for a, b in zip(mst.row, mst.col):
            edges.add((int(min(a, b)), int(max(a, b))))
    # ---- candidate links
    cands = []
    for u, v in edges:
        lu, lv = labels[u, :M], labels[v, :M]
        ok = (lu >= 0) & (lv >= 0)
        if not ok.any():
            continue
        nq, nr = lu[ok].max() + 1, lv[ok].max() + 1
        O = np.zeros((nq, nr), int)
        np.add.at(O, (lu[ok], lv[ok]), 1)
        for q in range(nq):
            r = int(O[q].argmax())
            if O[q, r] == 0 or int(O[:, r].argmax()) != q:
                continue
            if O[q, r] >= 0.5 * sizes[u][q] and O[q, r] >= 0.5 * sizes[v][r]:
                cands.append((int(O[q, r]), u, q, v, r))
    cands.sort(key=lambda x: -x[0])
    # ---- union-find with node-disjointness
    parent, nodes_of = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u in range(V):
        for q in range(len(sizes[u])):
            parent[(u, q)] = (u, q)
            nodes_of[(u, q)] = {u}
    refusals = 0
    for _, u, q, v, r in cands:
        a, b = find((u, q)), find((v, r))
        if a == b:
            continue
        if nodes_of[a] & nodes_of[b]:
            refusals += 1
            continue
        if len(nodes_of[a]) < len(nodes_of[b]):
            a, b = b, a
        parent[b] = a
        nodes_of[a] |= nodes_of.pop(b)
    roots = {}
    comp = {}
    for key in parent:
        rt = find(key)
        comp[key] = roots.setdefault(rt, len(roots))
    neg_comp = np.array([comp.get((int(u), int(l)), -1) if l >= 0 else -1 for u, l in zip(neg_node, neg_lab)])
    tr = Tracker(V=V, node_c=node_c, labels=labels, pts=pts, pts_lab=pts_lab, cut=cuts, comp=comp, delta=delta,
                 bank=clouds, thr=thrs,
                 diag={"n_edges": len(edges), "n_links_cand": len(cands), "refusals": refusals,
                       "n_components": len(roots), "neg_unseeded": int((neg_comp < 0).sum())})
    tr.sizes = sizes
    return tr, neg_comp


def comp_mass_at(tr: Tracker, u: int) -> Dict[int, float]:
    """typical-decode share of each component at node u."""
    lab = tr.labels[u, :-1]
    lab = lab[lab >= 0]
    out = {}
    for q in lab:
        g = tr.comp[(u, int(q))]
        out[g] = out.get(g, 0) + 1.0 / max(len(lab), 1)
    return out


def flag_or(neg_comp):
    return set(int(g) for g in neg_comp if g >= 0)


def flag_enrichment(tr: Tracker, neg_comp, neg_node):
    """RTRACK: flag g iff X_g >= 1, X_g > N m_g, and (X_g >= 2 or N m_g < 0.5); drop any flagged g
    holding more than half of a negative node's mass (pre-decision alarm)."""
    N = len(neg_comp)
    X = {}
    for g in neg_comp:
        if g >= 0:
            X[int(g)] = X.get(int(g), 0) + 1
    masses = [comp_mass_at(tr, int(u)) for u in neg_node]
    flagged, alarms = set(), 0
    for g, x in X.items():
        m = float(np.mean([ms.get(g, 0.0) for ms in masses]))
        if x >= 1 and x > N * m and (x >= 2 or N * m < 0.5):
            if any(ms.get(g, 0.0) > 0.5 for ms in masses):
                alarms += 1
                continue
            flagged.add(g)
    return flagged, alarms


def pool_bad_mask(tr: Tracker, flagged, pool_a: np.ndarray, pool_node: np.ndarray):
    """(bad, assigned) over pool samples; pool sample j sits at node pool_node[j]."""
    comp = np.array([tr.assign(int(u), pool_a[j:j + 1])[0] for j, u in enumerate(pool_node)])
    assigned = comp >= 0
    bad = np.isin(comp, list(flagged)) if flagged else np.zeros(len(comp), bool)
    return bad, assigned


# ------------------------------------------------------------------ JOINT-plateau (spec #3)
def joint_labeller(ref, seed, node_c, neg_c, neg_a, pool_c, pool_a, Ms=16, kcs=(4, 8, 16, 32, 64, 128)):
    V = len(node_c)
    Z = torch.randn(V * Ms, 2, generator=gen(seed, P_JOINT))
    cc = torch.from_numpy(np.repeat(node_c, Ms).astype(np.float32))
    A = ref.sample(Z, cc).numpy()
    C = np.repeat(node_c, Ms)
    sc = np.sort(node_c)
    gap = float(np.median(np.diff(sc)))
    res = {}
    for kc in kcs:
        # eps from the pooled cloud at the provisional scale lam0 = 1/(kc gap) (pilot definition),
        # then lam = eps / (kc gap): adjacent nodes sit eps/kc apart along the condition axis
        lam0 = 1.0 / (kc * gap)
        X0 = np.c_[lam0 * C, A]
        d0, _ = cKDTree(X0).query(X0, k=2)
        eps = 3.0 * float(np.median(d0[:, 1]))
        lam = eps / (kc * gap)
        X = np.c_[lam * C, A]
        tree = cKDTree(X)
        kd, _ = tree.query(X, k=K_TYP + 1)
        typ = kd[:, -1] <= np.quantile(kd[:, -1], Q_TYP)
        Xt = X[typ]
        t2 = cKDTree(Xt)
        pairs = t2.query_pairs(eps, output_type="ndarray")
        n = len(Xt)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
        ncomp, lab = connected_components(g, directed=False)
        cnt = np.bincount(lab)
        big = cnt >= 0.01 * n
        lab = np.where(big[lab], lab, -1)
        res[kc] = (int(big.sum()), lam, Xt, lab, t2, eps, tree, float(np.quantile(kd[:, -1], Q_TYP)))
    counts = [res[k][0] for k in kcs]
    pick = kcs[-1]
    for i in range(len(kcs) - 2):
        if counts[i] >= 2 and counts[i] == counts[i + 1] == counts[i + 2]:
            pick = kcs[i]
            break
    nc, lam, Xt, lab, t2, eps, tree_full, thrJ = res[pick]

    def vote(c, a):
        q = np.c_[lam * np.asarray(c), np.asarray(a)]
        dq, _ = tree_full.query(q, k=K_TYP)
        atyp = np.atleast_2d(dq)[:, -1] > thrJ          # review fix 1: typicality gate
        _, idx = t2.query(q, k=8)
        out = []
        for row in np.atleast_2d(idx):
            l = lab[row]
            l = l[l >= 0]
            out.append(int(np.bincount(l).argmax()) if len(l) else -1)
        out = np.array(out)
        out[atyp] = -1
        return out
    neg_comp = vote(neg_c, neg_a)
    flagged = set(int(g) for g in neg_comp if g >= 0)
    pc = vote(pool_c, pool_a)
    bad = np.isin(pc, list(flagged)) if flagged else np.zeros(len(pc), bool)
    return bad, pc >= 0, {"joint_kc": pick, "joint_counts": counts, "joint_ncomp": nc, "joint_eps": eps,
                          "neg_unseeded": int((neg_comp < 0).sum())}


# ------------------------------------------------------------------ PRA (spec #4)
def pra_rows(ref, seed, tr: Tracker, flagged, train_nodes, S=64, max_tries=8):
    """Pinned rows: good noises keep their own decode; bad noises are Hungarian-assigned (action
    cost) to fresh typical good decodes at the same node."""
    cs, ns, ts = [], [], []
    stats = {"pra_bad": 0, "pra_rows": 0, "pra_short": 0}
    for u in train_nodes:
        g = gen(seed, P_PRA + 7 * int(u))
        c = float(tr.node_c[u])
        z = torch.randn(S, 2, generator=g)
        a = ref.sample(z, torch.full((S,), c)).numpy()
        comp = tr.assign(int(u), a)
        bad = np.isin(comp, list(flagged)) if flagged else np.zeros(S, bool)
        good = (comp >= 0) & ~bad
        tgt = a.copy()
        nb = int(bad.sum())
        if nb:
            pool = np.empty((0, a.shape[1]), dtype=a.dtype)    # spec #4: fresh rejection samples only
            tries = 0
            while len(pool) < nb and tries < max_tries:
                zz = torch.randn(S, 2, generator=g)
                aa = ref.sample(zz, torch.full((S,), c)).numpy()
                cc = tr.assign(int(u), aa)
                ok = (cc >= 0) & ~(np.isin(cc, list(flagged)))
                pool = np.concatenate([pool, aa[ok]])
                tries += 1
            if len(pool) == 0:
                stats["pra_short"] += 1
                keep = ~bad
                z, tgt = z[torch.from_numpy(keep)], tgt[keep]
            else:
                E = pool[:nb] if len(pool) >= nb else pool[np.arange(nb) % len(pool)]
                cost = np.linalg.norm(a[bad][:, None] - E[None], axis=-1) ** 2
                ri, ci = linear_sum_assignment(cost)
                bi = np.flatnonzero(bad)
                tgt[bi[ri]] = E[ci]
        stats["pra_bad"] += nb
        cs += [c] * len(z)
        ns.append(z)
        ts.append(torch.from_numpy(np.asarray(tgt, dtype=np.float32)))
    stats["pra_rows"] = len(cs)
    return torch.tensor(cs, dtype=torch.float32), torch.cat(ns), torch.cat(ts), stats


# ------------------------------------------------------------------ REACH vote (spec #5)
class ReachVote:
    """Vote weighted by a reach curve w(delta) estimated from pairs of negatives.

    h_ij = 1[f_ref(xi_i, c_j) in B_j] - m_j; w = Gaussian-kernel regression of h on delta(c_i, c_j)
    (bandwidth = median pair delta); v(z, c) = sum_j w(delta(c, c_j)) [1[f_ref(z, c_j) in B_j] - m_j]
    / sum_j |w(delta(c, c_j))|; theta = 0.5 x median leave-one-out self-vote; abstain where
    sum_j |w| is below its 10th percentile over pool queries (excluded from the anchor)."""

    def __init__(self, ref, basins, neg_c, neg_xi, delta_neg_neg):
        self.ref, self.basins, self.c = ref, basins, np.asarray(neg_c)
        self.N = len(basins)
        self.m = np.array([float(np.mean(b.lab == b.bad)) for b in basins])
        H = np.zeros((self.N, self.N))
        if self.N > 1:
            dec = ref.sample(neg_xi.repeat(self.N, 1), torch.from_numpy(np.repeat(self.c, self.N).astype(np.float32)))
            dec = dec.reshape(self.N, self.N, 2)     # [j, i] = f_ref(xi_i, c_j)
            for j in range(self.N):
                H[:, j] = (basins[j].assign(dec[j]) == basins[j].bad).astype(float) - self.m[j]
        self.H = H
        dd = delta_neg_neg.copy()
        iu = ~np.eye(self.N, dtype=bool)
        self.pair_d, self.pair_h = dd[iu], H[iu]
        self.bw = float(np.median(self.pair_d)) if self.N > 1 else 1.0
        self.theta = None
        if self.N > 1:
            sv = [self._vote_from(np.delete(H[i], i), np.delete(delta_neg_neg[i], i))[0] for i in range(self.N)]
            self.theta = 0.5 * float(np.median(sv))

    def w(self, d):
        d = np.asarray(d, dtype=np.float64)
        if self.N < 2:
            return np.zeros_like(d)
        flat = d.reshape(-1)
        out = np.empty_like(flat)
        step = max(1, int(2e7 // max(len(self.pair_d), 1)))      # review fix 3: chunked, same values
        for s0 in range(0, len(flat), step):
            K = np.exp(-0.5 * ((flat[s0:s0 + step, None] - self.pair_d[None]) / self.bw) ** 2)
            out[s0:s0 + step] = (K * self.pair_h).sum(-1) / np.maximum(K.sum(-1), 1e-9)
        return out.reshape(d.shape)

    def _vote_from(self, hits_minus_m, d):
        w = self.w(d)
        return float((w * hits_minus_m).sum() / max(np.abs(w).sum(), 1e-9)), float(np.abs(w).sum())

    @torch.no_grad()
    def vote(self, z, d_to_negs):
        """z (P,2) candidate noises; d_to_negs (P, N) deltas of each query's condition to the negatives."""
        P = z.shape[0]
        dec = self.ref.sample(z.repeat(self.N, 1), torch.from_numpy(np.repeat(self.c, P).astype(np.float32)))
        dec = dec.reshape(self.N, P, 2)
        hm = np.stack([(self.basins[j].assign(dec[j]) == self.basins[j].bad).astype(float) - self.m[j]
                       for j in range(self.N)], 1)                      # (P, N)
        w = self.w(d_to_negs)                                          # (P, N)
        return (w * hm).sum(1) / np.maximum(np.abs(w).sum(1), 1e-9), np.abs(w).sum(1)
