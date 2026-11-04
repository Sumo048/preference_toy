"""Conditional toy: an 8-mode ring whose geometry depends on a condition c in [0, 1).

    centre_j(c) = r(c) * (cos(2 pi j / 8 + phi(c)), sin(...)),   r(c) = 2.5 + c,
    phi(c) = 1.5 * (2 pi / 8) * c

Across the condition range the ring rotates by 1.5 mode spacings and grows, so a location in
action space does not identify a mode without its condition: mode j at one condition sits
where another mode sits at a different condition. A "behaviour" is the mode INDEX, consistent
across conditions (as "placed on its side" is a behaviour at any state). Every logged rollout
has its own condition, so no two samples share one -- the practical regime.

The policy is a conditional rectified flow v(x, t, c), same convention as the unconditional
toy (t = 1 noise, t = 0 action, Euler 1 -> 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..flow import ResidualBlock, SinusoidalTimeEmbedding

OFF = -1


@dataclass(frozen=True)
class CondMixture:
    n_modes: int = 8
    mode_std: float = 0.25
    r0: float = 2.5
    r_slope: float = 1.0
    rot_spacings: float = 1.5
    assign_k_std: float = 3.0

    def centers(self, c) -> np.ndarray:
        """(N, M, 2) mode centres at each condition."""
        c = np.asarray(c, dtype=np.float64).reshape(-1, 1)
        ang = 2 * np.pi * np.arange(self.n_modes)[None, :] / self.n_modes \
            + self.rot_spacings * (2 * np.pi / self.n_modes) * c
        r = self.r0 + self.r_slope * c
        return np.stack([r * np.cos(ang), r * np.sin(ang)], -1).astype(np.float32)

    def sample(self, c, rng: np.random.Generator, modes=None):
        c = np.asarray(c).reshape(-1)
        pool = np.arange(self.n_modes) if modes is None else np.asarray(list(modes))
        m = rng.choice(pool, size=len(c))
        C = self.centers(c)[np.arange(len(c)), m]
        return (C + self.mode_std * rng.standard_normal(C.shape)).astype(np.float32), m

    def assign(self, x, c) -> np.ndarray:
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
        d = np.linalg.norm(x[:, None, :] - self.centers(c), axis=-1)
        lab = d.argmin(1)
        lab[d.min(1) > self.assign_k_std * self.mode_std] = OFF
        return lab


class CondVelocityNet(nn.Module):
    def __init__(self, hidden=128, n_layers=3, t_dim=64, c_freqs=8, dim=2):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(t_dim)
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.c_freqs = c_freqs
        self.c_mlp = nn.Sequential(nn.Linear(2 * c_freqs, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, 2 * t_dim) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def c_feat(self, c):
        k = torch.arange(1, self.c_freqs + 1, dtype=c.dtype)
        a = 2 * math.pi * c.reshape(-1, 1) * k.reshape(1, -1) / 2.0
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward(self, x, t, c):
        cond = torch.cat([self.time_mlp(self.time_embed(t)), self.c_mlp(self.c_feat(c))], -1)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, cond)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


class CondFlow(nn.Module):
    n_euler_steps = 20

    def __init__(self, **kw):
        super().__init__()
        self.kw = kw
        self.net = CondVelocityNet(**kw)

    def flow_loss(self, a, noise, t, c):
        tt = t.reshape(-1, 1)
        x_t = tt * noise + (1 - tt) * a
        return ((self.net(x_t, t, c) - (noise - a)) ** 2).mean(-1)

    @torch.no_grad()
    def sample(self, xi, c, batch=65536):
        out = []
        K = self.n_euler_steps
        dt = 1.0 / K
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float()
            cc = c[i:i + batch].float()
            for k in range(K):
                t = torch.full((x.shape[0],), 1.0 - k * dt)
                x = x - dt * self.net(x, t, cc)
            out.append(x)
        return torch.cat(out)


def frozen(p: CondFlow) -> CondFlow:
    import copy
    q = copy.deepcopy(p)
    q.eval()
    for x in q.parameters():
        x.requires_grad_(False)
    return q
