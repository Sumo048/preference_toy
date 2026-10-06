"""Every random stream in the toy, in one place.

Fairness rule: anything an arm consumes is drawn from a stream keyed by
(experiment seed, purpose[, pool row]) and NEVER by the arm or the sweep cell.
Consequences:

* every arm in a seed edits the same base policy, is evaluated on the same eval
  noises (common random numbers, so per-noise flips are paired) and trains with
  the same minibatch RNG;
* a given logged negative gets the same candidates, the same target and the same
  jitters in every cell it appears in, so the n_neg and n_forget sweeps are
  NESTED: the 4-sample edit set contains the 2-sample one, row for row;
* two arms that share a construction step (e.g. R_spread_log and R_spread_nhood,
  or every spread_* support variant) share its output bit for bit, as the
  preference repo requires.
"""

from __future__ import annotations

import numpy as np
import torch

# purpose offsets
BASE_TRAIN = 0
EVAL_NOISE = 10_000
POOL = 20_000
FORGET_PERM = 30_000
TRAIN = 50_000
RM_TRAIN = 60_000
BON = 70_000
ONLINE_POOL = 80_000
TEST_SAMPLES = 90_000
ORACLE_POOL = 100_000

# per-row streams (target candidates vs support noises are separate streams so
# that S-nhood draws the same target as S-log, cf. pref_pillars.reroute)
ROW_CAND = 1
ROW_SUPPORT = 2
ROW_RESAMPLE = 3


def gen(seed: int, purpose: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + purpose)


def np_rng(seed: int, purpose: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) * 1_000_003 + purpose)


def row_gen(seed: int, pool_index: int, stream: int, round_: int = 0) -> torch.Generator:
    """Stream for one logged negative, identified by its index in the rollout pool."""
    return torch.Generator().manual_seed(
        (int(seed) * 7_919 + int(round_) * 104_729) * 10_000_019 + int(pool_index) * 16 + stream
    )
