"""Unconditional rectified-flow policy over R^2 with a deterministic Euler sampler.

Convention is the one the preference repo (pi0 / openpi) and pref_pillars use:

    t ~ Beta(1.5, 1) * 0.999 + 0.001
    x_t = t * noise + (1 - t) * a          t = 1 is noise, t = 0 is the action
    u_t = noise - a
    loss = || v_theta(x_t, t) - u_t ||^2

Sampling integrates t: 1 -> 0 with ``x <- x - dt * v``. A logged noise ``xi``
therefore fully determines its action, which is the property rerouting re-pairs.
The network has no conditioning input: the toy removes the state so that every
effect is about the noise -> action map alone.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicyConfig:
    dim: int = 2
    hidden_dim: int = 128
    n_layers: int = 3
    time_embed_dim: int = 64
    n_euler_steps: int = 20


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=t.dtype) / max(half - 1, 1))
        ang = t.reshape(-1, 1) * freqs.reshape(1, -1) * 1_000.0
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim + cond_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, h, cond):
        z = self.act(self.fc1(torch.cat([self.norm(h), cond], dim=-1)))
        return h + self.fc2(z)


class VelocityNet(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim), nn.SiLU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )
        self.inp = nn.Linear(cfg.dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.hidden_dim, cfg.time_embed_dim) for _ in range(cfg.n_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        cond = self.time_mlp(self.time_embed(t))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


class FlowPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        self.net = VelocityNet(cfg)

    # -- training ---------------------------------------------------------
    def flow_loss(self, a, noise, t, reduce="none"):
        tt = t.reshape(-1, 1)
        x_t = tt * noise + (1.0 - tt) * a
        per = ((self.net(x_t, t) - (noise - a)) ** 2).mean(-1)
        return per.mean() if reduce == "mean" else per

    @staticmethod
    def sample_time(n: int, gen: torch.Generator, a: float = 1.5, b: float = 1.0):
        """Beta(a, b) * 0.999 + 0.001, reproducible from ``gen``."""
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        beta = np.random.default_rng(seed).beta(a, b, size=n)
        return torch.from_numpy(beta * 0.999 + 0.001).float()

    # -- deterministic sampling -------------------------------------------
    @torch.no_grad()
    def sample(self, xi, n_steps=None, repel_from=None, repel_alpha=0.0, batch=65536):
        """Integrate t: 1 -> 0. ``repel_alpha > 0`` repels from ``repel_from``.

        Sign: the ablation plan writes ``v' = v + alpha (x - a_neg)`` for a 0 -> 1
        sampler; this one steps ``x <- x - dt v`` so the repulsive form is
        ``v' = v - alpha (x - a_neg)`` (pref_pillars measured the literal sign to
        attract).
        """
        K = int(n_steps or self.cfg.n_euler_steps)
        outs = []
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float()
            neg = None if repel_from is None else repel_from[i:i + batch]
            dt = 1.0 / K
            for k in range(K):
                t = torch.full((x.shape[0],), 1.0 - k * dt)
                v = self.net(x, t)
                if repel_alpha != 0.0:
                    v = v - repel_alpha * (x - neg)
                x = x - dt * v
            outs.append(x)
        return torch.cat(outs)

    @torch.no_grad()
    def invert(self, a, n_steps=None, t_start=0.0, time_offset=0.0):
        """Explicit-Euler inverse of the sampler, walking t: t_start -> 1.

        Each step is ``x <- x + dt * v(x, t + time_offset * dt)``.
        ``time_offset = 1`` reads the velocity at the forward-grid point whose
        sampler step this undoes (patch 0019's default, used by the inverted-ball
        arms); ``time_offset = 0`` reads it at the current time (the "FRS
        indexing" of patch 0020/0021, used by the noised partial inversion).
        With ``t_start > 0`` the caller has already noised ``a`` to ``t_start``
        and only the grid steps in [t_start, 1) are walked.
        """
        K = int(n_steps or self.cfg.n_euler_steps)
        dt = 1.0 / K
        x = a.clone().float()
        for k in range(K):
            tv = k * dt
            if tv + 1e-9 < t_start:
                continue
            t = torch.full((x.shape[0],), tv + time_offset * dt)
            x = x + dt * self.net(x, t)
        return x

    def log_prob(self, x0, n_steps: int = 100, batch: int = 4096):
        """Continuous-time log density via the instantaneous change of variables.

        d log p_t(x_t)/dt = -div v, integrated from the data end (t=0) to the prior
        (t=1) with Euler. This is the density of the continuous ODE, not of the
        20-step sampler; it is a secondary metric and is reported as such.
        """
        outs = []
        dt = 1.0 / n_steps
        for i in range(0, x0.shape[0], batch):
            x = x0[i:i + batch].clone().float()
            div_int = torch.zeros(x.shape[0])
            for k in range(n_steps):
                tv = k * dt
                t = torch.full((x.shape[0],), max(tv, 1e-3))
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    v = self.net(xg, t)
                    div = torch.zeros(x.shape[0])
                    for d in range(x.shape[1]):
                        g = torch.autograd.grad(v[:, d].sum(), xg, retain_graph=d < x.shape[1] - 1)[0]
                        div = div + g[:, d]
                x = (xg + dt * v).detach()
                div_int = div_int + dt * div.detach()
            logn = -0.5 * (x ** 2).sum(1) - 0.5 * x.shape[1] * math.log(2 * math.pi)
            outs.append(logn + div_int)
        return torch.cat(outs)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)


def frozen_copy(policy: FlowPolicy) -> FlowPolicy:
    ref = copy.deepcopy(policy)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def load_policy(path: str) -> FlowPolicy:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    pol = FlowPolicy(PolicyConfig(**sd["policy_cfg"]))
    pol.load_state_dict(sd["ema"])
    return frozen_copy(pol)
