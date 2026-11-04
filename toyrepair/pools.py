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

Sub-streams (W1-A1). Tools that need more than one kind of draw from the mining pool use child
streams of the same SeedSequence, ``SeedSequence([pool, cell_index, check_index],
spawn_key=(stream,))`` (= child ``stream`` of the root). The root stream itself is ``draw``.
Mining-pool use, all at check index 0:

    root      ``draw(cell, MINING, 4096)``           quick_check drift pairs (c, xi)
    stream 1  ``noise(cell, MINING, 256, stream=1)``  residual-basin-mass noises (same 256 for every
                                                     negative and every version: common random numbers)
    stream 2  ``torch_generator(cell, MINING, stream=2, salt=seed)``  branch-tool renoise eps
    stream 3  ``torch_generator(cell, MINING, stream=3)``             hop-search directions
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


# ---------------------------------------------------------------- sub-streams (W1-A1)
STREAM_RESIDUAL, STREAM_BRANCH, STREAM_HOP = 1, 2, 3
N_DRIFT_PAIRS = 4096


def sub_sequence(pool: int, cell_index: int, check_index: int = 0, stream: int = 0) -> np.random.SeedSequence:
    """Child ``stream`` of ``seed_sequence(pool, cell_index, check_index)`` (stream 0 here is a
    child too, not the root; the root is ``seed_sequence``)."""
    root = seed_sequence(pool, cell_index, check_index)
    return np.random.SeedSequence(root.entropy, spawn_key=(int(stream),))


def noise(cell, pool: int, n: int, stream: int, index: int = 0) -> torch.Tensor:
    """(n, 2) float32 standard normals from child ``stream`` of (pool, cell.index, index)."""
    g = np.random.default_rng(sub_sequence(pool, int(cell.index), index, stream))
    return torch.from_numpy(g.standard_normal((n, 2), dtype=np.float32))


def torch_generator(cell, pool: int, stream: int, index: int = 0, salt: int = 0) -> torch.Generator:
    """A ``torch.Generator`` seeded from child ``stream`` of (pool, cell.index, index); ``salt``
    (e.g. a tool call's seed argument) selects an independent generator inside that stream."""
    ss = sub_sequence(pool, int(cell.index), index, stream)
    st = np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (int(salt),)).generate_state(2, np.uint32)
    return torch.Generator().manual_seed(int(st[0]) << 31 ^ int(st[1]))
