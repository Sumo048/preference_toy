"""Label-free clustering of logged negatives (PLAN 6.2 affinity, average linkage cut 0.7)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import lib


@torch.no_grad()
def affinity(cell, basins):
    ed = cell.ed
    N = len(ed.uid)
    # f_ref(xi_j, c_i) for all i, j
    xi = ed.xi.repeat(N, 1)                                  # j fastest
    c = ed.c.repeat_interleave(N)                            # i slow
    a = cell.ref.sample(xi, c).reshape(N, N, 2)
    hit = np.zeros((N, N))
    for i in range(N):
        hit[i] = basins[i].assign(a[i]) == basins[i].bad
    return 0.5 * (hit + hit.T)


def cluster(cell, basins, cut=0.7):
    A = affinity(cell, basins)
    D = 1 - A
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    return fcluster(Z, t=cut, criterion="distance") - 1, A
