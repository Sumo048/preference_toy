"""Oracle mode labels (owner W1-A2; gate, final evaluation and miner purity only)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

OFF = -1  # "between modes"
N_MODES = 8


def _finite(a: torch.Tensor) -> torch.Tensor:
    """Non-finite actions are moved far away, so they label OFF (as toypref does)."""
    a = a.detach().float().cpu()
    ok = torch.isfinite(a).all(1, keepdim=True)
    return torch.where(ok, a, torch.full_like(a, 1e6))


def label(cell, a: torch.Tensor, c: torch.Tensor) -> np.ndarray:
    """(B,) int64 mode index in 0..7, or OFF = -1 between modes, for actions a (B, 2) at
    conditions c (B,). cond: ``toypref.cond.task.CondMixture().assign(a, c)`` (assign radius
    3 x mode_std = 0.75); uncond: ``toypref.mixture.assign_modes(DEFAULT.mixture, a)``."""
    a = _finite(torch.as_tensor(a))
    if cell.grid == "cond":
        from toypref.cond.task import CondMixture
        c = torch.as_tensor(c).float().reshape(-1)
        lab = CondMixture().assign(a, c)
    else:
        from toypref.config import DEFAULT
        from toypref.mixture import assign_modes
        lab = assign_modes(DEFAULT.mixture, a)
    return np.asarray(lab, dtype=np.int64)


def masses(lab: np.ndarray, n_modes: int = N_MODES) -> np.ndarray:
    """(n_modes + 1,) float64 shares per mode; last entry = OFF. Same as toypref ``masses``."""
    lab = np.asarray(lab)
    return np.array([np.mean(lab == j) for j in range(n_modes)] + [np.mean(lab == OFF)], dtype=np.float64)


def counts(lab: np.ndarray, n_modes: int = N_MODES) -> np.ndarray:
    """(n_modes + 1,) int64 counts per mode; last entry = OFF."""
    lab = np.asarray(lab, dtype=np.int64)
    return np.bincount(np.where(lab == OFF, n_modes, lab), minlength=n_modes + 1)


def negative_modes(cell, uids: Sequence[int]) -> np.ndarray:
    """(m,) int oracle mode of each logged negative, by uid (``cell.neg.uid``)."""
    uids = np.asarray(list(uids), dtype=np.int64)
    pos = {int(u): i for i, u in enumerate(cell.neg.uid)}
    missing = [int(u) for u in uids if int(u) not in pos]
    if missing:
        raise KeyError(f"uids not among the cell's logged negatives: {missing[:5]}")
    modes = np.asarray(cell._oracle.neg_modes)
    return modes[[pos[int(u)] for u in uids]].astype(np.int64)


def target_modes(cell, uids: Sequence[int]) -> List[int]:
    """Majority oracle modes of a cluster's members (by negative uid): every mode that holds
    the largest member count (ties keep all), sorted."""
    m = negative_modes(cell, uids)
    if m.size == 0:
        return []
    vals, cnt = np.unique(m, return_counts=True)
    return sorted(int(v) for v in vals[cnt == cnt.max()])
