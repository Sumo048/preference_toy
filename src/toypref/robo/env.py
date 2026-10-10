"""robo2d environment: setups, observations, chunk execution and the episode labeller.

A SETUP is (start, goal, 1-3 disc obstacles with radii) drawn from a seeded generator in a
canonical frame (start on the left, goal on the right, obstacles in between), then rotated by a
uniform angle and translated, so absolute positions and headings vary across setups.

Episodes are batched: every array has a leading episode axis E. Absent obstacles (setups with
fewer than 3) are padded with radius 0 and a far-away centre; in the observation they are
zero-padded.

The LABELLER (``label``) is the source of the trajectory-level labels (collision, jerk). Methods
never call it: they only see the logged negative rows. Metrics and diagnostics may use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

H = 8                  # steps per action chunk
N_CHUNKS = 6           # chunks per episode
T_MAX = H * N_CHUNKS
V_SCALE = 0.05         # action units: normalised velocity * V_SCALE = displacement per step
ACT_DIM = 2 * H
MAX_OBS = 3
OBS_DIM = 4 + 3 * MAX_OBS
GOAL_TOL = 0.10
MAX_STEP = 0.20        # per-step displacement clip (4x nominal); edited policies cannot teleport
FAR = 1.0e3

# jerk metric: max over windows of W second differences of mean ||p_{k+1} - 2 p_k + p_{k-1}||^2
JERK_W = 8
JERK_THR = 1.2e-3      # fixed: clean demo chunks max 5.8e-4, injected zig-zag min 2.7e-3 


@dataclass
class Setups:
    """A batch of setups (S of them)."""
    sid: np.ndarray        # (S,) global setup ids
    start: np.ndarray      # (S, 2)
    goal: np.ndarray       # (S, 2)
    centre: np.ndarray     # (S, 3, 2), padded with FAR
    radius: np.ndarray     # (S, 3), padded with 0
    n_obs: np.ndarray      # (S,)

    def __len__(self):
        return len(self.sid)

    def take(self, idx) -> "Setups":
        idx = np.asarray(idx)
        return Setups(self.sid[idx], self.start[idx], self.goal[idx], self.centre[idx],
                      self.radius[idx], self.n_obs[idx])

    def axis(self):
        d = self.goal - self.start
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def sample_setups(rng: np.random.Generator, n: int, sid0: int = 0) -> Setups:
    """Canonical frame: start (-D/2, 0), goal (D/2, 0); obstacles between them with a passable gap."""
    S = np.zeros((n, 2)); G = np.zeros((n, 2))
    C = np.full((n, MAX_OBS, 2), FAR); R = np.zeros((n, MAX_OBS)); K = np.zeros(n, dtype=int)
    for i in range(n):
        D = rng.uniform(1.5, 1.9)
        k = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        for _ in range(1000):
            xs = np.sort(rng.uniform(-0.32 * D, 0.32 * D, k))
            ys = rng.uniform(-0.10, 0.10, k)
            rs = rng.uniform(0.12, 0.24, k)
            cc = np.stack([xs, ys], -1)
            ok = True
            for a in range(k):
                for b in range(a + 1, k):
                    if np.linalg.norm(cc[a] - cc[b]) < rs[a] + rs[b] + 0.30:
                        ok = False
                for p in (np.array([-D / 2, 0.0]), np.array([D / 2, 0.0])):
                    if np.linalg.norm(cc[a] - p) < rs[a] + 0.30:
                        ok = False
            if ok:
                break
        th = rng.uniform(0, 2 * np.pi)
        Rm = _rot(th)
        off = rng.uniform(-0.3, 0.3, 2)
        S[i] = Rm @ np.array([-D / 2, 0.0]) + off
        G[i] = Rm @ np.array([D / 2, 0.0]) + off
        C[i, :k] = cc @ Rm.T + off
        R[i, :k] = rs
        K[i] = k
    return Setups(np.arange(sid0, sid0 + n), S, G, C, R, K)


def observe(p, goal, centre, radius) -> np.ndarray:
    """(E, 13): position, goal - position, 3 nearest obstacles (centre - position, radius) by
    surface distance; absent obstacles are zero-padded."""
    rel = centre - p[:, None, :]
    present = radius > 0
    surf = np.where(present, np.linalg.norm(rel, axis=-1) - radius, np.inf)
    order = np.argsort(surf, axis=1, kind="stable")
    rel = np.take_along_axis(rel, order[..., None], 1)
    rad = np.take_along_axis(radius, order, 1)
    pres = np.take_along_axis(present, order, 1)
    rel = np.where(pres[..., None], rel, 0.0)
    rad = np.where(pres, rad, 0.0)
    obst = np.concatenate([rel, rad[..., None]], -1).reshape(len(p), -1)
    return np.concatenate([p, goal - p, obst], -1).astype(np.float32)


def seg_disc_hit(a, b, centre, radius) -> np.ndarray:
    """(E,) whether segment a->b (E,2) touches any disc (E,3,2)/(E,3)."""
    ab = (b - a)[:, None, :]
    ac = centre - a[:, None, :]
    den = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip((ac * ab).sum(-1) / den, 0.0, 1.0)
    closest = a[:, None, :] + t[..., None] * ab
    d = np.linalg.norm(closest - centre, axis=-1)
    return ((d < radius) & (radius > 0)).any(-1)


class Episodes:
    """Batched episode state; positions are (E, T_MAX + 1, 2), NaN after termination."""

    def __init__(self, st: Setups, idx):
        self.idx = np.asarray(idx)                    # setup index per episode
        E = len(self.idx)
        self.start, self.goal = st.start[self.idx], st.goal[self.idx]
        self.centre, self.radius = st.centre[self.idx], st.radius[self.idx]
        self.pos = np.full((E, T_MAX + 1, 2), np.nan)
        self.pos[:, 0] = self.start
        self.p = self.start.copy()
        self.t = np.zeros(E, dtype=int)
        self.alive = np.ones(E, dtype=bool)
        self.collided = np.zeros(E, dtype=bool)
        self.success = np.zeros(E, dtype=bool)
        self.col_chunk = np.full(E, -1)
        self.n_chunks = np.zeros(E, dtype=int)        # chunks started
        self.clipped = np.zeros(E, dtype=int)

    def obs(self):
        return observe(self.p, self.goal, self.centre, self.radius)

    def execute(self, act: np.ndarray, chunk: int):
        """Run one normalised chunk (E, 16) open loop for the alive episodes."""
        vel = act.reshape(-1, H, 2).astype(np.float64) * V_SCALE
        live = self.alive.copy()
        self.n_chunks[live] += 1
        for h in range(H):
            a = self.alive
            if not a.any():
                break
            st = vel[:, h]
            n = np.linalg.norm(st, axis=-1, keepdims=True)
            over = n[:, 0] > MAX_STEP
            self.clipped += (over & a)
            st = np.where(over[:, None], st * MAX_STEP / np.maximum(n, 1e-12), st)
            new = self.p + st
            hit = seg_disc_hit(self.p, new, self.centre, self.radius) & a
            self.collided |= hit
            self.col_chunk[hit] = chunk
            mv = a & ~hit
            self.p[mv] = new[mv]
            self.t[mv] += 1
            self.pos[mv, self.t[mv]] = self.p[mv]
            reach = mv & (np.linalg.norm(self.p - self.goal, axis=-1) < GOAL_TOL)
            self.success |= reach
            self.alive &= ~(hit | reach)


def jerk_score(pos: np.ndarray) -> np.ndarray:
    """(E,) max over windows of JERK_W second differences of mean squared second difference.

    ``pos`` is (E, T, 2) with NaN after termination."""
    d2 = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]
    q = (d2 ** 2).sum(-1)                                  # (E, T-2), NaN where undefined
    valid = ~np.isnan(q)
    q0 = np.where(valid, q, 0.0)
    cs = np.concatenate([np.zeros((len(q), 1)), np.cumsum(q0, 1)], 1)
    cv = np.concatenate([np.zeros((len(q), 1)), np.cumsum(valid, 1)], 1)
    W = JERK_W
    if q.shape[1] < W:
        n = np.maximum(cv[:, -1], 1)
        return cs[:, -1] / n
    s = cs[:, W:] - cs[:, :-W]
    n = cv[:, W:] - cv[:, :-W]
    full = n == W
    m = np.where(full, s / W, 0.0)
    out = m.max(1)
    short = ~full.any(1)                                   # episodes with < W second differences
    if short.any():
        out[short] = cs[short, -1] / np.maximum(cv[short, -1], 1)
    return out


def chunk_jerk(act: np.ndarray) -> np.ndarray:
    """(N,) jerk score of a single normalised chunk executed in free space."""
    vel = act.reshape(-1, H, 2) * V_SCALE
    pos = np.concatenate([np.zeros((len(vel), 1, 2)), np.cumsum(vel, 1)], 1)
    return jerk_score(pos)


def homotopy(st: Setups, idx, pos) -> np.ndarray:
    """(E,) int code: bit j set when the path passes obstacle j (canonical order along the
    start->goal axis) on the LEFT of the axis. -1 for paths with no positions."""
    idx = np.asarray(idx)
    ax = st.axis()[idx]
    out = np.zeros(len(idx), dtype=int)
    for e in range(len(idx)):
        s = idx[e]
        k = st.n_obs[s]
        cen = st.centre[s, :k]
        order = np.argsort((cen - st.start[s]) @ ax[e])
        P = pos[e][~np.isnan(pos[e, :, 0])]
        code = 0
        for bit, j in enumerate(order):
            d = np.linalg.norm(P - cen[j], axis=-1)
            q = P[d.argmin()] - cen[j]
            if ax[e, 0] * q[1] - ax[e, 1] * q[0] > 0:
                code |= 1 << bit
        out[e] = code
    return out


def label(ep: Episodes):
    """Trajectory labeller: (negative, collided, jerky, jerk score). THE ONLY LABEL SOURCE."""
    js = jerk_score(ep.pos)
    jerky = js > JERK_THR
    return ep.collided | jerky, ep.collided.copy(), jerky, js


def path_length(pos) -> np.ndarray:
    d = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
    return np.nansum(d, 1)
