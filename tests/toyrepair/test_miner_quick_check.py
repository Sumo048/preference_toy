import csv
import os

import numpy as np
import pytest
import torch

import toyrepair
from toyrepair import miner, policy
from toyrepair.cell import load_cell
from toyrepair.quick_check import quick_check

RUNS = os.environ["TOYPREF_RUNS"]
have_cond = os.path.exists(os.path.join(RUNS, "cond_base", "s0", "policy.pt"))


def _block(sizes, inside=0.95, outside=0.05):
    n = sum(sizes)
    A = np.full((n, n), outside)
    s = 0
    for k in sizes:
        A[s:s + k, s:s + k] = inside
        s += k
    np.fill_diagonal(A, 1.0)
    return A


def test_cluster_blocks_hdbscan(monkeypatch):
    monkeypatch.setattr(miner, "CLUSTER_RULE", "hdbscan")      # W2a set the default to agglomerative
    A = _block([4, 5, 3])
    uids = np.arange(100, 112)[::-1].copy()
    cls = miner.cluster(A, uids)
    assert sorted(len(c.members) for c in cls) == [3, 4, 5]
    assert miner.cluster_method(A) == "hdbscan"
    assert sorted(np.concatenate([c.members for c in cls]).tolist()) == list(range(12))
    # ids ordered by smallest uid
    mins = [int(c.uids.min()) for c in cls]
    assert mins == sorted(mins)


def test_cluster_rule_switch(monkeypatch):
    A = _block([4, 5, 3])
    monkeypatch.setattr(miner, "CLUSTER_RULE", "agglomerative")
    assert miner.cluster_method(A) == "agglomerative" and len(miner.cluster(A, np.arange(12))) == 3


def test_cluster_fallback_and_tiny():
    A = _block([6])
    assert miner.cluster_method(A) == "agglomerative" and len(miner.cluster(A, np.arange(6))) == 1
    assert len(miner.cluster(np.ones((1, 1)), np.array([7]))) == 1
    A2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert len(miner.cluster(A2, np.array([1, 2]))) == 2
    A3 = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert len(miner.cluster(A3, np.array([1, 2]))) == 1


@pytest.fixture(scope="module")
def cell():
    if not have_cond:
        pytest.skip("no runs/cond_base/s0")
    return load_cell("cond", "k4", 0, 4)


def test_mine_and_caps(cell, tmp_path):
    v0 = policy.base_version(cell)
    at = miner.mine(cell, v0)
    N = len(cell.neg)
    A = at.affinity
    assert A.shape == (N, N) and np.allclose(A, A.T) and np.all(np.diag(A) == 1)
    assert sum(len(c.members) for c in at.clusters) == N
    assert at.residual.shape == (N,) and np.all((at.residual >= 0) & (at.residual <= 1))
    # logged negatives sit in their own basins: pi_ref residual mass is clearly positive
    assert at.residual.mean() > 0.1
    for card in at.cards:
        assert miner.est_tokens(card.render()) <= miner.CARD_TOKEN_CAP
        assert card.r_cur >= miner.OPEN_THRESHOLD and card.failure_class == "?"
    assert miner.est_tokens(at.render()) <= miner.ATLAS_TOKEN_CAP
    assert [c.r_cur for c in at.cards] == sorted([c.r_cur for c in at.cards], reverse=True)
    p = tmp_path / "purity.csv"
    miner.log_purity(cell, at, str(p))
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == len(at.clusters) and all(0 < float(r["purity"]) <= 1 for r in rows)
    # the LLM-facing render never mentions purity or modes
    assert "purity" not in at.render() and "mode" not in at.render()


def test_quick_check_identity(cell):
    v0 = policy.base_version(cell)
    cl = miner.clusters_of(cell)[0]
    q = quick_check(cell, v0, v0, cl)
    assert q.drift == 0.0
    assert q.residual == pytest.approx(float(miner.residual_ref(cell)[cl.members].mean()))
    assert 0 <= q.still_bad <= 1 and q.decodes > 4096
