"""Every random stream in robo2d.

Rule (as in the other toys): anything an arm consumes is drawn from a stream keyed by
(experiment seed, purpose[, id]) and NEVER by the arm, so all arms of a seed edit the same base,
see the same logged negatives, are evaluated on the same noises (common random numbers) and
share each negative row's candidate / target / jitter draws.
"""

from __future__ import annotations

import numpy as np
import torch

# purposes
P_BASE_SETUPS = 1_000      # setups the base policy's demos come from (disjoint from the experiment setups)
P_BASE_DEMOS = 2_000
P_BASE_TRAIN = 3_000       # minibatch / noise / time stream of base training
P_SETUPS = 10_000          # the 160 training + 40 held-out experiment setups
P_NONEG = 11_000           # which training setups carry no negatives
P_KEEP = 12_000            # how many negatives (1-3) a setup keeps
P_LOG = 20_000             # + setup id: logged rollout noises
P_EVAL = 40_000            # + setup id: eval noise sequences (CRN)
P_TRAIN = 60_000           # fine-tune minibatch stream (shared by all arms)
P_PROBE = 70_000

# per-row streams
ROW_CAND = 1               # basins decodes, hop directions, R_ours targets
ROW_SUPPORT = 2            # bsup jitters
ROW_LOCAL = 4              # local-anchor decodes
ROW_SDPO = 5               # SDPO positives


def gen(seed: int, purpose: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + int(purpose))


def nprng(seed: int, purpose: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) * 1_000_003 + int(purpose))


def row_gen(seed: int, uid: int, stream: int) -> torch.Generator:
    """Stream for one negative row, identified by its uid (setup, episode, chunk)."""
    return torch.Generator().manual_seed((int(seed) * 7919 + 17) * 10_000_019 + int(uid) * 32 + int(stream))
