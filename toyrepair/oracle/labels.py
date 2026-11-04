"""Oracle mode labels (owner W1-A2; gate, final evaluation and miner purity only)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

OFF = -1  # "between modes"


def label(cell, a: torch.Tensor, c: torch.Tensor) -> np.ndarray:
    """(B,) int64 mode index in 0..7, or OFF = -1 between modes, for actions a (B, 2) at
    conditions c (B,). cond: ``toypref.cond.task.CondMixture().assign(a, c)`` (assign radius
    3 x mode_std = 0.75); uncond: ``toypref.mixture.assign_modes(DEFAULT.mixture, a)``."""
    raise NotImplementedError


def masses(lab: np.ndarray, n_modes: int = 8) -> np.ndarray:
    """(n_modes + 1,) float64 shares per mode; last entry = OFF. Same as toypref ``masses``."""
    raise NotImplementedError


def target_modes(cell, uids: Sequence[int]) -> List[int]:
    """Majority oracle modes of a cluster's members (by negative uid): every mode that holds
    the largest member count (ties keep all), sorted."""
    raise NotImplementedError
