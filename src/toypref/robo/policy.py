"""Chunk flow policy v(x_t, t, obs): a pi0-style conditional rectified flow over 16-D chunks.

Same convention as the other toys: x_t = t xi + (1 - t) a (t = 1 noise, t = 0 action),
target velocity xi - a, Beta(1.5, 1) * 0.999 + 0.001 times, Euler 20 steps 1 -> 0.
The observation (13-D) enters through an MLP embedding concatenated to the time embedding.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..flow import ResidualBlock, SinusoidalTimeEmbedding
from .env import ACT_DIM, OBS_DIM


class ChunkVelocityNet(nn.Module):
    def __init__(self, hidden=256, n_layers=4, t_dim=64, o_dim=128, act_dim=ACT_DIM, obs_dim=OBS_DIM):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(t_dim)
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.obs_mlp = nn.Sequential(nn.Linear(obs_dim, o_dim), nn.SiLU(), nn.Linear(o_dim, o_dim), nn.SiLU(),
                                     nn.Linear(o_dim, o_dim))
        self.inp = nn.Linear(act_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, t_dim + o_dim) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, act_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, o):
        cond = torch.cat([self.time_mlp(self.time_embed(t)), self.obs_mlp(o)], -1)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, cond)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


class ChunkFlow(nn.Module):
    n_euler_steps = 20

    def __init__(self, **kw):
        super().__init__()
        self.kw = kw
        self.net = ChunkVelocityNet(**kw)

    def flow_loss(self, a, noise, t, o):
        """Per-row flow-matching loss (mean over the 16 action dims)."""
        tt = t.reshape(-1, 1)
        x_t = tt * noise + (1 - tt) * a
        return ((self.net(x_t, t, o) - (noise - a)) ** 2).mean(-1)

    @torch.no_grad()
    def sample(self, xi, o, batch=32768):
        out = []
        K = self.n_euler_steps
        dt = 1.0 / K
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float()
            oo = o[i:i + batch].float()
            for k in range(K):
                t = torch.full((x.shape[0],), 1.0 - k * dt)
                x = x - dt * self.net(x, t, oo)
            out.append(x)
        return torch.cat(out) if out else xi.new_zeros((0, xi.shape[1]))


def frozen(p: ChunkFlow) -> ChunkFlow:
    q = copy.deepcopy(p)
    q.eval()
    for x in q.parameters():
        x.requires_grad_(False)
    return q


def save(pol: ChunkFlow, path: str, **extra):
    torch.save({"kw": pol.kw, "state": pol.state_dict(), **extra}, path)


def load(path: str) -> ChunkFlow:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    p = ChunkFlow(**sd["kw"])
    p.load_state_dict(sd["state"])
    return frozen(p)
