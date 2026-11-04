import numpy as np
import pytest
import torch

from toyrepair import pools
from toyrepair.cell import cell_index


class _C:
    def __init__(self, index):
        self.index = index


def test_cell_index_formula():
    assert cell_index("cond", "k1", 0, 1) == 0
    assert cell_index("cond", "k4", 0, 4) == 201
    assert cell_index("cond", "alt", 7, 64) == 100 * 4 + 70 + 3
    assert cell_index("uncond", "base_k2", 3, 16) == 1000 + 100 + 30 + 2
    idx = {cell_index(g, s, seed, n)
           for g, ss in (("cond", ("k1", "k2", "k4", "contig", "alt", "pairs", "b3p1")),
                         ("uncond", ("base_k1", "base_k2", "base_k4")))
           for s in ss for seed in range(10) for n in (1, 4, 16, 64)}
    assert len(idx) == 10 * 10 * 4
    with pytest.raises(ValueError):
        cell_index("cond", "base_k2", 0, 4)
    with pytest.raises(ValueError):
        cell_index("cond", "k4", 0, 8)


def test_seed_sequence_matches_spec():
    ss = pools.seed_sequence(pools.GATE, 201, 5)
    assert list(ss.entropy) == [2, 201, 5] or ss.entropy == [2, 201, 5]
    a = np.random.default_rng(np.random.SeedSequence([2, 201, 5])).random(4, dtype=np.float32)
    c, _ = pools.draw(_C(201), pools.GATE, 4, index=5)
    assert np.array_equal(c.numpy(), a)


def test_draw_deterministic_and_shapes():
    c1, x1 = pools.draw(_C(201), pools.MINING, 1000)
    c2, x2 = pools.draw(_C(201), pools.MINING, 1000)
    assert torch.equal(c1, c2) and torch.equal(x1, x2)
    assert c1.shape == (1000,) and x1.shape == (1000, 2)
    assert c1.dtype == torch.float32 and x1.dtype == torch.float32
    assert float(c1.min()) >= 0.0 and float(c1.max()) < 1.0
    assert abs(float(x1.mean())) < 0.1 and abs(float(x1.std()) - 1) < 0.1


def test_streams_independent():
    base = pools.draw(_C(201), pools.MINING, 512)[1]
    others = [pools.draw(_C(201), pools.GATE, 512)[1], pools.draw(_C(201), pools.FINAL, 512)[1],
              pools.draw(_C(201), pools.GATE, 512, index=1)[1], pools.draw(_C(202), pools.MINING, 512)[1]]
    for o in others:
        assert not torch.equal(base, o)
        r = np.corrcoef(base[:, 0].numpy(), o[:, 0].numpy())[0, 1]
        assert abs(r) < 0.15
    assert not torch.equal(pools.draw(_C(201), pools.GATE, 64, 0)[1], pools.draw(_C(201), pools.GATE, 64, 1)[1])


def test_bad_pool():
    with pytest.raises(ValueError):
        pools.seed_sequence(4, 0, 0)
