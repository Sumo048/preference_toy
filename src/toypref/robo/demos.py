"""Expert demonstrations with low-rate weird-behaviour injection.

A demo passes each obstacle on a chosen side (homotopy class): waypoints at lateral offset
r + clearance around every obstacle in the canonical frame, Chaikin corner cutting, then
arc-length resampling with a per-demo speed and a slow speed modulation. Training samples are
taken at EVERY step k (obs at p_k, chunk = the next H velocities, zero-padded after the goal),
as chunked VLA policies are trained.

Injection (the source of the base policy's tails), per training sample:
  collision  with prob ``q_col`` among samples whose nearest obstacle is ahead and within
             reach, the chunk is replaced by H steps at the demo speed heading INTO it
  jerk       with prob ``q_jerk`` the chunk gets an alternating perpendicular zig-zag of
             amplitude 0.5-0.8 x speed (net displacement unchanged)
The demo itself keeps following its clean path (the injected chunk is an alternative action at
that state), so p(bad | state) ~ the injection rate: a tail, not a mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import env as E

Q_COL = 0.03          # calibrated (base_calib): collision ~7%, jerk ~8%, success ~93% of episodes
Q_JERK = 0.012
CLEAR_MIN = 0.03
SMOOTH_SIGMA = 0.12
MAX_TURN = 0.06        # rad per 0.01 of arc length (curvature <= 6)


@dataclass
class DemoSet:
    obs: np.ndarray        # (N, 13)
    act: np.ndarray        # (N, 16) normalised
    kind: np.ndarray       # (N,) 0 clean, 1 collision-injected, 2 jerk-injected (diagnostic only)
    setup: np.ndarray      # (N,)
    demo: np.ndarray       # (N,)
    hom: np.ndarray        # (N,) homotopy code of the demo


def _chaikin(P, it=5):
    for _ in range(it):
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R = 0.25 * P[:-1] + 0.75 * P[1:]
        mid = np.empty((2 * len(Q), 2))
        mid[0::2], mid[1::2] = Q, R
        P = np.concatenate([P[:1], mid, P[-1:]])
    return P


def _frame(st: E.Setups, s):
    u = st.goal[s] - st.start[s]
    D = np.linalg.norm(u)
    u = u / D
    n = np.array([-u[1], u[0]])
    mid = 0.5 * (st.start[s] + st.goal[s])
    return mid, u, n, D


def demo_path(st: E.Setups, s: int, rng: np.random.Generator, sides=None, max_tries=40):
    """Dense smooth path (P, 2) from start to goal and its side code, or (None, None)."""
    mid, u, n, D = _frame(st, s)
    k = int(st.n_obs[s])
    cen = st.centre[s, :k]
    rad = st.radius[s, :k]
    cx, cy = (cen - mid) @ u, (cen - mid) @ n
    order = np.argsort(cx)
    for _ in range(max_tries):
        sd = rng.choice([-1, 1], size=k) if sides is None else np.asarray(sides)
        wps = [np.array([-D / 2, 0.0])]
        for j in order:
            clr = rng.uniform(0.12, 0.24)
            y = cy[j] + sd[j] * (rad[j] + clr)
            jx = rng.uniform(-0.04, 0.04)
            for dx in (-rad[j] - 0.06, 0.0, rad[j] + 0.06):
                wps.append(np.array([cx[j] + dx + jx, y]))
        wps.append(np.array([D / 2, 0.0]))
        keep = [wps[0]]                       # waypoints strictly forward along the axis
        for w in wps[1:-1]:
            if keep[-1][0] + 0.05 < w[0] < D / 2 - 0.10:
                keep.append(w)
        wps = keep + [wps[-1]]
        P = _smooth(_densify(_chaikin(np.stack(wps)), 0.01), SMOOTH_SIGMA / 0.01)
        W = mid + P[:, :1] * u + P[:, 1:] * n
        dense = W
        d = np.linalg.norm(dense[:, None, :] - cen[None], axis=-1) - rad[None]
        v = np.diff(dense, axis=0)
        cosang = (v[1:] * v[:-1]).sum(-1) / np.maximum(
            np.linalg.norm(v[1:], axis=-1) * np.linalg.norm(v[:-1], axis=-1), 1e-12)
        if d.min() >= CLEAR_MIN and cosang.min() > np.cos(MAX_TURN):
            code = sum(1 << b for b, j in enumerate(order) if sd[j] > 0)
            return W, code
    return None, None


def _smooth(P, sig):
    """Gaussian low-pass of a densely sampled path (endpoints held), bounding its curvature."""
    r = int(3 * sig)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(P[:1], r, 0), P, np.repeat(P[-1:], r, 0)])
    out = np.stack([np.convolve(pad[:, d], k, mode="valid") for d in range(2)], -1)
    out[0], out[-1] = P[0], P[-1]
    return out


def _densify(W, h):
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.arange(0, s[-1], h)
    return np.stack([np.interp(q, s, W[:, 0]), np.interp(q, s, W[:, 1])], -1)


def resample(W, rng: np.random.Generator):
    """Positions p_0..p_T at a varying speed; p_T = goal."""
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    v0 = rng.uniform(0.055, 0.075)
    per, ph = rng.uniform(25, 50), rng.uniform(0, 2 * np.pi)
    ks = np.arange(E.T_MAX + 1)
    v = v0 * (1 + 0.12 * np.sin(2 * np.pi * ks / per + ph))
    cum = np.concatenate([[0], np.cumsum(v)])
    T = int(np.searchsorted(cum, L))
    q = np.minimum(cum[:T + 1], L)
    P = np.stack([np.interp(q, s, W[:, 0]), np.interp(q, s, W[:, 1])], -1)
    return P, v0


def chunks_from(P):
    """(T, 16) chunk at every step k < T: velocities k..k+H-1, zero after the goal."""
    T = len(P) - 1
    vel = np.diff(P, axis=0)
    vel = np.concatenate([vel, np.zeros((E.H, 2))])
    idx = np.arange(T)[:, None] + np.arange(E.H)[None]
    return vel[idx].reshape(T, -1) / E.V_SCALE


def make_demos(st: E.Setups, n_per_setup: int, rng: np.random.Generator, q_col=Q_COL, q_jerk=Q_JERK) -> DemoSet:
    obs, act, kind, sup, dem, hom = [], [], [], [], [], []
    did = 0
    for s in range(len(st)):
        for _ in range(n_per_setup):
            W, code = demo_path(st, s, rng)
            if W is None:
                continue
            P, v0 = resample(W, rng)
            T = len(P) - 1
            if T > E.T_MAX - 2:
                continue
            p = P[:T]
            o = E.observe(p, np.repeat(st.goal[s:s + 1], T, 0), np.repeat(st.centre[s:s + 1], T, 0),
                          np.repeat(st.radius[s:s + 1], T, 0))
            a = chunks_from(P)
            kd = np.zeros(T, dtype=int)
            # collision injection: nearest obstacle ahead and within one chunk's reach
            rel = st.centre[s][None] - p[:, None, :]
            dist = np.linalg.norm(rel, axis=-1)
            surf = np.where(st.radius[s][None] > 0, dist - st.radius[s][None], np.inf)
            j = surf.argmin(1)
            dsurf = surf[np.arange(T), j]
            to_c = rel[np.arange(T), j] / dist[np.arange(T), j, None]
            heading = (a.reshape(T, E.H, 2)[:, :4].sum(1))
            heading = heading / np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-9)
            ahead = (heading * to_c).sum(-1) > 0.3
            elig = (dsurf > 0.03) & (dsurf < 0.40) & ahead
            u = rng.random(T)
            col = elig & (u < q_col)
            for k in np.flatnonzero(col):
                c, r = st.centre[s, j[k]], st.radius[s, j[k]]
                d = np.linalg.norm(c - p[k])
                half = np.arcsin(min(r / d, 1.0))
                ang = np.arctan2(*(c - p[k])[::-1]) + rng.uniform(-0.5, 0.5) * half
                spd = np.linalg.norm(np.diff(P, axis=0)[k]) if k < T else v0
                spd = max(spd, 0.8 * v0)
                a[k] = np.tile(np.array([np.cos(ang), np.sin(ang)]) * spd / E.V_SCALE, E.H)
                kd[k] = 1
            # jerk injection: any mid-path sample not already injected
            u2 = rng.random(T)
            jer = (u2 < q_jerk) & (kd == 0) & (np.arange(T) < T - E.H)
            for k in np.flatnonzero(jer):
                v = a[k].reshape(E.H, 2)
                sp = np.linalg.norm(v, axis=-1, keepdims=True)
                nrm = np.stack([-v[:, 1], v[:, 0]], -1) / np.maximum(sp, 1e-9)
                A = rng.uniform(0.5, 0.8) * sp
                sign = rng.choice([-1, 1]) * (-1.0) ** np.arange(E.H)[:, None]
                a[k] = (v + A * sign * nrm).reshape(-1)
                kd[k] = 2
            obs.append(o); act.append(a.astype(np.float32)); kind.append(kd)
            sup.append(np.full(T, st.sid[s])); dem.append(np.full(T, did)); hom.append(np.full(T, code))
            did += 1
    return DemoSet(np.concatenate(obs), np.concatenate(act), np.concatenate(kind), np.concatenate(sup),
                   np.concatenate(dem), np.concatenate(hom))
