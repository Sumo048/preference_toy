"""The toy target: an equal-weight mixture of isotropic Gaussians on a ring in R^2.

Everything downstream labels a point by its nearest mode centre, with an "off-mode"
bucket for points farther than ``assign_radius`` from every centre. The off bucket
is what catches the bridges a flow draws between modes, and it is the cheapest
signal of a damaged policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

OFF = -1


@dataclass(frozen=True)
class MixtureConfig:
    n_modes: int = 8
    dim: int = 2
    radius: float = 3.0
    mode_std: float = 0.25
    #: points farther than this from every centre are labelled OFF
    assign_radius: float = 0.75   # = 3 * mode_std

    def centers(self) -> np.ndarray:
        if self.dim != 2:
            raise ValueError("the ring layout is defined for dim=2")
        ang = 2 * np.pi * np.arange(self.n_modes) / self.n_modes
        return np.stack([self.radius * np.cos(ang), self.radius * np.sin(ang)], 1).astype(np.float32)

    def adjacency(self) -> np.ndarray:
        """(M, M) bool: ring neighbours."""
        M = self.n_modes
        A = np.zeros((M, M), dtype=bool)
        for j in range(M):
            A[j, (j - 1) % M] = A[j, (j + 1) % M] = True
        return A


def sample_true(cfg: MixtureConfig, n: int, rng: np.random.Generator, modes=None):
    """(x, mode) i.i.d. from the true mixture, optionally restricted to ``modes``."""
    C = cfg.centers()
    pool = np.arange(cfg.n_modes) if modes is None else np.asarray(list(modes))
    m = rng.choice(pool, size=n)
    x = C[m] + cfg.mode_std * rng.standard_normal((n, cfg.dim)).astype(np.float32)
    return x.astype(np.float32), m


def assign_modes(cfg: MixtureConfig, x) -> np.ndarray:
    """Nearest-centre label, OFF beyond ``assign_radius``."""
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    C = cfg.centers()
    d = np.linalg.norm(x[:, None, :] - C[None, :, :], axis=-1)
    lab = d.argmin(1)
    lab[d.min(1) > cfg.assign_radius] = OFF
    return lab


def mode_masses(cfg: MixtureConfig, labels: np.ndarray) -> np.ndarray:
    """(M + 1,) fractions; the last entry is the OFF bucket."""
    M = cfg.n_modes
    out = np.zeros(M + 1)
    for j in range(M):
        out[j] = np.mean(labels == j)
    out[M] = np.mean(labels == OFF)
    return out
