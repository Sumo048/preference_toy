"""Expert demonstrations on long courses with low-rate weird-behaviour injection.

A demo passes each obstacle of each segment on a chosen side: per segment, waypoints at lateral
offset r + clearance around every obstacle in the segment frame; the segments are concatenated,
Chaikin corner-cut, low-passed (which rounds the turns at the waypoints) and resampled at a
per-demo speed. A demo is kept when it clears every obstacle by CLEAR_MIN, its curvature is
bounded and it passes every waypoint within WP_TOL (< GOAL_TOL, so the waypoint switch fires).
Training samples are taken at every step (obs with the waypoint switching rule of the env).

Injection (the source of the base's tails), per training sample:
  F1 collision  prob q_col where the nearest obstacle is ahead and within reach: H steps into it
  F2 jerk       prob q_jerk: alternating perpendicular zig-zag (net displacement unchanged)
  F3 momentum   prob q_mom where the demo turns away from an obstacle it is heading at: the chunk
                continues straight at the last velocity. The contact follows a chunk or more later,
                so the cause precedes the symptom.
  F4 route      (knob, default 0) per demo with prob q_route one obstacle is passed with a NEGATIVE
                clearance (a grazing route): a separate, colliding mode rather than a tail.
  marker        (knob rho, default 0) every injection rate is multiplied by clip(1 + rho z, 0, 3),
                z = the demo's standardised speed: a harmless behaviour (speed) correlated with
                failures (P2b).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import env as E

Q_COL = 0.004
Q_JERK = 0.006
Q_MOM = 0.02
Q_ROUTE = 0.0
RHO = 0.0
CLEAR_MIN = 0.03
SMOOTH_SIGMA = 0.12
MAX_TURN = 0.06
WP_TOL = 0.07
V_LO, V_HI = 0.055, 0.075
V_SD = (V_HI - V_LO) / np.sqrt(12)


@dataclass
class DemoSet:
    obs: np.ndarray
    act: np.ndarray
    kind: np.ndarray       # 0 clean, 1 collision, 2 jerk, 3 momentum (diagnostic only)
    setup: np.ndarray
    demo: np.ndarray
    route: np.ndarray      # demo is a grazing-route demo (diagnostic only)


def _chaikin(P, it=5):
    for _ in range(it):
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R = 0.25 * P[:-1] + 0.75 * P[1:]
        mid = np.empty((2 * len(Q), 2))
        mid[0::2], mid[1::2] = Q, R
        P = np.concatenate([P[:1], mid, P[-1:]])
    return P


def _smooth(P, sig):
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
    q = np.append(q, s[-1])
    return np.stack([np.interp(q, s, W[:, 0]), np.interp(q, s, W[:, 1])], -1)


def demo_path(st: E.Setups, s: int, rng: np.random.Generator, route=False, max_tries=40):
    """Dense smooth path through the course, its grazing obstacle slot (-1 if none), or (None, -1)."""
    K = st.K
    for _ in range(max_tries):
        graze = -1
        if route:
            pres = np.flatnonzero(st.radius[s] > 0)
            graze = int(rng.choice(pres))
        pts = [st.wps[s, 0]]
        for k in range(K):
            a, b = st.wps[s, k], st.wps[s, k + 1]
            D = np.linalg.norm(b - a)
            u = (b - a) / D
            n = np.array([-u[1], u[0]])
            sl = np.arange(k * E.SEG_OBS, (k + 1) * E.SEG_OBS)
            sl = sl[st.radius[s, sl] > 0]
            cx, cy = (st.centre[s, sl] - a) @ u, (st.centre[s, sl] - a) @ n
            loc = []
            for j, x, y in sorted(zip(sl, cx, cy), key=lambda z: z[1]):
                r = st.radius[s, j]
                side = rng.choice([-1, 1])
                clr = -0.35 * r if j == graze else rng.uniform(0.12, 0.24)
                yy = y + side * (r + clr)
                jx = rng.uniform(-0.04, 0.04)
                for dx in (-r - 0.06, 0.0, r + 0.06):
                    loc.append((x + dx + jx, yy))
            keep = []
            last = 0.0
            for x, y in loc:
                if last + 0.05 < x < D - 0.10:
                    keep.append(a + x * u + y * n)
                    last = x
            pts += keep + [b]
        P = _smooth(_densify(_chaikin(np.stack(pts)), 0.01), SMOOTH_SIGMA / 0.01)
        d = np.linalg.norm(P[:, None, :] - st.centre[s][None], axis=-1) - st.radius[s][None]
        d = np.where(st.radius[s][None] > 0, d, np.inf)
        if graze >= 0:
            d[:, graze] = np.inf
        v = np.diff(P, axis=0)
        cosang = (v[1:] * v[:-1]).sum(-1) / np.maximum(np.linalg.norm(v[1:], axis=-1) * np.linalg.norm(v[:-1], axis=-1), 1e-12)
        wp_d = np.array([np.linalg.norm(P - st.wps[s, k], axis=-1).min() for k in range(1, K + 1)])
        if d.min() >= CLEAR_MIN and cosang.min() > np.cos(MAX_TURN) and (wp_d < WP_TOL).all():
            return P, graze
    return None, -1


def resample(W, rng: np.random.Generator):
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    v0 = rng.uniform(V_LO, V_HI)
    per, ph = rng.uniform(25, 50), rng.uniform(0, 2 * np.pi)
    ks = np.arange(4 * E.T_MAX)
    v = v0 * (1 + 0.12 * np.sin(2 * np.pi * ks / per + ph))
    cum = np.concatenate([[0], np.cumsum(v)])
    T = int(np.searchsorted(cum, L))
    q = np.minimum(cum[:T + 1], L)
    P = np.stack([np.interp(q, s, W[:, 0]), np.interp(q, s, W[:, 1])], -1)
    return P, v0


def waypoint_index(st: E.Setups, s: int, P: np.ndarray) -> np.ndarray:
    """Waypoint index (1..K) the observation at each position P[t] sees, with the env's rule: the
    index advances AFTER arriving at a position within GOAL_TOL of the current waypoint."""
    w = 1
    out = np.ones(len(P), dtype=int)
    for t in range(1, len(P)):
        if w < st.K and np.linalg.norm(P[t] - st.wps[s, w]) < E.GOAL_TOL:
            w += 1
        out[t] = w
    return out


def chunks_from(P):
    T = len(P) - 1
    vel = np.diff(P, axis=0)
    vel = np.concatenate([vel, np.zeros((E.H, 2))])
    idx = np.arange(T)[:, None] + np.arange(E.H)[None]
    return vel[idx].reshape(T, -1) / E.V_SCALE


def make_demos(st: E.Setups, n_per_setup: int, rng: np.random.Generator, q_col=Q_COL, q_jerk=Q_JERK, q_mom=Q_MOM,
               q_route=Q_ROUTE, rho=RHO) -> DemoSet:
    obs, act, kind, sup, dem, rte = [], [], [], [], [], []
    did = 0
    K = st.K
    for s in range(len(st)):
        for _ in range(n_per_setup):
            route = rng.random() < q_route
            W, graze = demo_path(st, s, rng, route=route)
            if W is None:
                continue
            P, v0 = resample(W, rng)
            T = len(P) - 1
            if T > E.T_MAX - 2:
                continue
            mult = float(np.clip(1 + rho * (v0 - 0.5 * (V_LO + V_HI)) / V_SD, 0.0, 3.0))
            p = P[:T]
            wi = waypoint_index(st, s, P)[:T]
            wp = st.wps[s, wi]
            wpn = st.wps[s, np.minimum(wi + 1, K)]
            wpn = np.where((wi < K)[:, None], wpn, wp)
            pv = np.concatenate([np.zeros((1, 2)), np.diff(P[:T], axis=0)]) / E.V_SCALE
            o = E.observe(p, wp, wpn, np.repeat(st.centre[s:s + 1], T, 0), np.repeat(st.radius[s:s + 1], T, 0), pv)
            a = chunks_from(P)
            kd = np.zeros(T, dtype=int)
            cen, rad = st.centre[s], st.radius[s]
            rel = cen[None] - p[:, None, :]
            dist = np.linalg.norm(rel, axis=-1)
            surf = np.where(rad[None] > 0, dist - rad[None], np.inf)
            j = surf.argmin(1)
            dsurf = surf[np.arange(T), j]
            to_c = rel[np.arange(T), j] / dist[np.arange(T), j, None]
            heading = a.reshape(T, E.H, 2)[:, :4].sum(1)
            heading = heading / np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-9)
            spd_step = np.concatenate([np.linalg.norm(np.diff(P, axis=0), axis=-1), [v0]])[:T]
            # F1: tail collision
            ahead = (heading * to_c).sum(-1) > 0.3
            col = (dsurf > 0.03) & (dsurf < 0.40) & ahead & (rng.random(T) < q_col * mult)
            for k in np.flatnonzero(col):
                c, r = cen[j[k]], rad[j[k]]
                d = np.linalg.norm(c - p[k])
                half = np.arcsin(min(r / d, 1.0))
                ang = np.arctan2(*(c - p[k])[::-1]) + rng.uniform(-0.5, 0.5) * half
                spd = max(spd_step[k], 0.8 * v0)
                a[k] = np.tile(np.array([np.cos(ang), np.sin(ang)]) * spd / E.V_SCALE, E.H)
                kd[k] = 1
            # F3: momentum -- the demo turns, the injected chunk goes straight and a later contact follows
            pvn = pv[:, :] / np.maximum(np.linalg.norm(pv, axis=-1, keepdims=True), 1e-9)
            turn = (pvn * heading).sum(-1) < np.cos(0.25)
            straight_hit = np.zeros(T, bool)
            for k in np.flatnonzero(turn & (kd == 0) & (np.linalg.norm(pv, axis=-1) > 0) & (dsurf > 0.15) & (dsurf < 0.70)):
                end = p[k] + pvn[k] * spd_step[k] * 3 * E.H
                straight_hit[k] = E.seg_disc_hit(p[k:k + 1] + pvn[k:k + 1] * spd_step[k] * E.H, end[None],
                                                 cen[None], rad[None])[0]
            mom = straight_hit & (rng.random(T) < q_mom * mult)
            for k in np.flatnonzero(mom):
                a[k] = np.tile(pvn[k] * spd_step[k] / E.V_SCALE, E.H)
                kd[k] = 3
            # F2: jerk
            jer = (rng.random(T) < q_jerk * mult) & (kd == 0) & (np.arange(T) < T - E.H)
            for k in np.flatnonzero(jer):
                v = a[k].reshape(E.H, 2)
                sp = np.linalg.norm(v, axis=-1, keepdims=True)
                nrm = np.stack([-v[:, 1], v[:, 0]], -1) / np.maximum(sp, 1e-9)
                A = rng.uniform(0.5, 0.8) * sp
                sign = rng.choice([-1, 1]) * (-1.0) ** np.arange(E.H)[:, None]
                a[k] = (v + A * sign * nrm).reshape(-1)
                kd[k] = 2
            obs.append(o); act.append(a.astype(np.float32)); kind.append(kd)
            sup.append(np.full(T, st.sid[s])); dem.append(np.full(T, did)); rte.append(np.full(T, graze >= 0))
            did += 1
    return DemoSet(np.concatenate(obs), np.concatenate(act), np.concatenate(kind), np.concatenate(sup),
                   np.concatenate(dem), np.concatenate(rte))
