"""Independent (c, xi) pools per cell (docs/PLAN.md Section 5.3).

Each cell keeps three random streams, made with
``numpy.random.SeedSequence([pool, cell_index, check_index])``:

    pool 1 = MINING   (miner, quick_check drift, branch-tool fresh noises), check index 0
    pool 2 = GATE     (one fresh batch per gate check), check index = gate-check number 0, 1, ...
    pool 3 = FINAL    (final evaluation of toyrepair metrics), check index 0

The cell index is ``1000*g + 100*s + 10*seed + n`` (see ``toyrepair.cell.cell_index``).

The logged negatives and pi_ref come from toypref's own seed rule, not from these pools. The
toypref method-map metrics (``toyrepair.oracle.metrics``) use toypref's own fixed eval bank.

Final-evaluation draws never feed back into any loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

MINING, GATE, FINAL = 1, 2, 3
POOLS = {"mining": MINING, "gate": GATE, "final": FINAL}


def seed_sequence(pool: int, cell_index: int, check_index: int = 0) -> np.random.SeedSequence:
    """The SeedSequence of one stream: ``SeedSequence([pool, cell_index, check_index])``."""
    if pool not in (MINING, GATE, FINAL):
        raise ValueError(f"pool must be 1 (mining), 2 (gate) or 3 (final), got {pool}")
    if cell_index < 0 or check_index < 0:
        raise ValueError("cell_index and check_index must be >= 0")
    return np.random.SeedSequence([int(pool), int(cell_index), int(check_index)])


def rng(pool: int, cell_index: int, check_index: int = 0) -> np.random.Generator:
    """A fresh ``numpy.random.Generator`` (PCG64) on one stream."""
    return np.random.default_rng(seed_sequence(pool, cell_index, check_index))


def draw_index(pool: int, cell_index: int, n: int, index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """``n`` pairs (c, xi) from stream (pool, cell_index, index).

    Returns
        c:  (n,) float32 tensor, uniform in [0, 1)  (condition; ignored by the unconditional toy)
        xi: (n, 2) float32 tensor, standard normal  (decode noise at step 0)

    Draw order: all n conditions first, then all n noises. Prefix property: the first m < n
    conditions equal ``draw_index(pool, cell_index, m, index)[0]`` (noises are not prefix-stable
    across different n, because they are drawn after all n conditions).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    g = rng(pool, cell_index, index)
    c = g.random(n, dtype=np.float32)
    xi = g.standard_normal((n, 2), dtype=np.float32)
    return torch.from_numpy(c), torch.from_numpy(xi)


def draw(cell, pool: int, n: int, index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """``n`` pairs (c, xi) for ``cell`` (any object with an int ``index`` attribute, e.g.
    ``toyrepair.cell.Cell``) from pool ``pool`` (MINING, GATE, FINAL) and check index ``index``.

    Mining and final evaluation use index 0; gate check j uses index j. Same arguments give
    bit-identical tensors. See ``draw_index`` for shapes and dtypes. (The plan fixes index 0 for
    mining and final; other indices are allowed for sub-streams but must be documented by the
    caller.)"""
    return draw_index(pool, int(cell.index), n, index)
