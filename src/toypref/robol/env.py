"""robol environment: long courses of K waypoint segments; contact does NOT end an episode.

A COURSE is K segments. Segment k runs from waypoint k to waypoint k+1 (waypoint 0 is the start)
past 1-3 disc obstacles placed as in robo2d (canonical frame, passable gap). Segment k+1 starts at
segment k's goal with its heading turned by U(-TURN, TURN). A course is accepted only when no
obstacle comes within (radius + CORRIDOR) of another segment's straight line and non-adjacent
segments stay SEG_SEP apart, so each obstacle belongs to one segment.

Observation (OBS_DIM = 17): position, current waypoint - position, next waypoint - current
waypoint (zero on the last segment), the 3 nearest obstacles of the whole course (centre -
position, radius) and the last executed step velocity / V_SCALE. The current waypoint advances
when the agent is within GOAL_TOL of it. Success = reaching the last waypoint.

Contact: a step that would enter a disc stops at the contact point and slides along the tangent
for the rest of the step (if the slide hits again it stays at the contact point). The episode goes
on. ``contact`` (E, T_MAX) marks every step that touched; an EVENT is a contact onset (a contact
with no contact in the previous H steps).

The LABELLER (``label``) is the source of trajectory labels (contact, jerk). Methods never call it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = 8
N_CHUNKS = 18
T_MAX = H * N_CHUNKS
V_SCALE = 0.05
ACT_DIM = 2 * H
MAX_OBS = 3
OBS_OBST = 6                       # obstacles start after position, goal offset, next-waypoint offset
OBS_VEL = OBS_OBST + 3 * MAX_OBS
OBS_DIM = OBS_VEL + 2
GOAL_TOL = 0.10
MAX_STEP = 0.20
FAR = 1.0e3
K_SEG = 4
SEG_OBS = 3                        # obstacle slots per segment
TURN = np.pi / 3
CORRIDOR = 0.30
SEG_SEP = 0.50
D_RANGE = (1.3, 1.7)

JERK_W = 8
JERK_THR = 1.2e-3


@dataclass
class Setups:
    """A batch of courses (S of them). ``centre``/``radius`` hold K * SEG_OBS slots (padded)."""
    sid: np.ndarray        # (S,)
    wps: np.ndarray        # (S, K + 1, 2): start, then the K waypoints
    centre: np.ndarray     # (S, K * SEG_OBS, 2), FAR padded
    radius: np.ndarray     # (S, K * SEG_OBS), 0 padded
    seg: np.ndarray        # (S, K * SEG_OBS) segment of each obstacle slot

    def __len__(self):
        return len(self.sid)

    @property
    def K(self):
        return self.wps.shape[1] - 1

    @property
    def start(self):
        return self.wps[:, 0]

    @property
    def goal(self):
        return self.wps[:, -1]

    def take(self, idx) -> "Setups":
        idx = np.asarray(idx)
        return Setups(self.sid[idx], self.wps[idx], self.centre[idx], self.radius[idx], self.seg[idx])


def concat(a: Setups, b: Setups) -> Setups:
    return Setups(*[np.concatenate([getattr(a, f), getattr(b, f)]) for f in ("sid", "wps", "centre", "radius", "seg")])


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _pt_seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + t * ab - p)


def _seg_seg_dist(a, b, c, d):
    return min(_pt_seg_dist(a, c, d), _pt_seg_dist(b, c, d), _pt_seg_dist(c, a, b), _pt_seg_dist(d, a, b))


def _segment_obstacles(rng, D):
    """Canonical-frame obstacles of one segment (start (-D/2, 0), goal (D/2, 0)), as robo2d."""
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
            return cc, rs
    return cc[:1], rs[:1]


def sample_setups(rng: np.random.Generator, n: int, sid0: int = 0, K: int = K_SEG) -> Setups:
    W = np.zeros((n, K + 1, 2))
    C = np.full((n, K * SEG_OBS, 2), FAR)
    R = np.zeros((n, K * SEG_OBS))
    G = np.repeat(np.arange(K), SEG_OBS)[None].repeat(n, 0)
    for i in range(n):
        for _ in range(1000):
            p = rng.uniform(-0.3, 0.3, 2)
            th = rng.uniform(0, 2 * np.pi)
            wps, cens, rads, segs = [p.copy()], [], [], []
            for k in range(K):
                D = rng.uniform(*D_RANGE)
                cc, rs = _segment_obstacles(rng, D)
                Rm = _rot(th)
                mid = p + Rm @ np.array([D / 2, 0.0])
                q = p + Rm @ np.array([D, 0.0])
                cens.append(cc @ Rm.T + mid)
                rads.append(rs)
                segs.append(np.full(len(rs), k))
                wps.append(q.copy())
                p = q
                th = th + rng.uniform(-TURN, TURN)
            wps = np.stack(wps)
            cen, rad, sg = np.concatenate(cens), np.concatenate(rads), np.concatenate(segs)
            ok = True
            for j in range(len(rad)):                 # each obstacle clear of other segments' corridors
                for k in range(K):
                    if k != sg[j] and _pt_seg_dist(cen[j], wps[k], wps[k + 1]) < rad[j] + CORRIDOR:
                        ok = False
            for a in range(K):                        # non-adjacent segments apart
                for b in range(a + 2, K):
                    if _seg_seg_dist(wps[a], wps[a + 1], wps[b], wps[b + 1]) < SEG_SEP:
                        ok = False
            if ok:
                break
        W[i] = wps
        for k in range(K):
            m = sg == k
            C[i, k * SEG_OBS:k * SEG_OBS + m.sum()] = cen[m]
            R[i, k * SEG_OBS:k * SEG_OBS + m.sum()] = rad[m]
    return Setups(np.arange(sid0, sid0 + n), W, C, R, G)


def observe(p, wp, wp_next, centre, radius, vel) -> np.ndarray:
    rel = centre - p[:, None, :]
    present = radius > 0
    surf = np.where(present, np.linalg.norm(rel, axis=-1) - radius, np.inf)
    order = np.argsort(surf, axis=1, kind="stable")[:, :MAX_OBS]
    rel = np.take_along_axis(rel, order[..., None], 1)
    rad = np.take_along_axis(radius, order, 1)
    pres = np.take_along_axis(present, order, 1)
    rel = np.where(pres[..., None], rel, 0.0)
    rad = np.where(pres, rad, 0.0)
    obst = np.concatenate([rel, rad[..., None]], -1).reshape(len(p), -1)
    return np.concatenate([p, wp - p, wp_next - wp, obst, vel], -1).astype(np.float32)


def seg_disc_hit(a, b, centre, radius) -> np.ndarray:
    """(E,) whether segment a->b touches any disc."""
    ab = (b - a)[:, None, :]
    ac = centre - a[:, None, :]
    den = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip((ac * ab).sum(-1) / den, 0.0, 1.0)
    closest = a[:, None, :] + t[..., None] * ab
    d = np.linalg.norm(closest - centre, axis=-1)
    return ((d < radius) & (radius > 0)).any(-1)


def first_hit(a, st, centre, radius):
    """Smallest t in [0, 1] with |a + t st - c| = r over the discs (points outside), and the disc."""
    f = a[:, None, :] - centre                                     # (E, M, 2)
    A = np.maximum((st ** 2).sum(-1), 1e-12)[:, None]
    B = 2 * (f * st[:, None, :]).sum(-1)
    Cq = (f ** 2).sum(-1) - radius ** 2
    disc = B ** 2 - 4 * A * Cq
    ok = (disc >= 0) & (radius > 0) & (Cq > 0)
    t0 = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
    t0 = np.where((t0 >= 0) & (t0 <= 1), t0, np.inf)
    j = t0.argmin(1)
    return t0[np.arange(len(a)), j], j


class Episodes:
    """Batched episode state; ``pos`` (E, T_MAX + 1, 2) NaN after termination, ``widx`` (E, T_MAX + 1)
    current waypoint index (1..K) at every step, ``contact`` (E, T_MAX) step touched a disc."""

    def __init__(self, st: Setups, idx):
        self.idx = np.asarray(idx)
        E = len(self.idx)
        self.wps = st.wps[self.idx]
        self.K = st.K
        self.centre, self.radius = st.centre[self.idx], st.radius[self.idx]
        self.pos = np.full((E, T_MAX + 1, 2), np.nan)
        self.pos[:, 0] = self.wps[:, 0]
        self.p = self.wps[:, 0].copy()
        self.v = np.zeros_like(self.p)
        self.w = np.ones(E, dtype=int)
        self.widx = np.zeros((E, T_MAX + 1), dtype=int)
        self.widx[:, 0] = 1
        self.contact = np.zeros((E, T_MAX), dtype=bool)
        self.t = np.zeros(E, dtype=int)
        self.alive = np.ones(E, dtype=bool)
        self.success = np.zeros(E, dtype=bool)
        self.clipped = np.zeros(E, dtype=int)

    def _wp(self):
        r = np.arange(len(self.p))
        cur = self.wps[r, self.w]
        nxt = np.where((self.w < self.K)[:, None], self.wps[r, np.minimum(self.w + 1, self.K)], cur)
        return cur, nxt

    def obs(self):
        cur, nxt = self._wp()
        return observe(self.p, cur, nxt, self.centre, self.radius, self.v)

    @property
    def collided(self):
        return self.contact.any(1)

    def execute(self, act: np.ndarray, chunk: int):
        vel = act.reshape(-1, H, 2).astype(np.float64) * V_SCALE
        for h in range(H):
            a = self.alive.copy()
            if not a.any():
                break
            st = vel[:, h]
            n = np.linalg.norm(st, axis=-1, keepdims=True)
            over = n[:, 0] > MAX_STEP
            self.clipped += (over & a)
            st = np.where(over[:, None], st * MAX_STEP / np.maximum(n, 1e-12), st)
            th, j = first_hit(self.p, st, self.centre, self.radius)
            hit = np.isfinite(th) & a
            new = self.p + st
            if hit.any():
                e = np.flatnonzero(hit)
                c = self.centre[e, j[e]]
                r = self.radius[e, j[e]]
                pc = self.p[e] + th[e, None] * st[e]
                nrm = pc - c
                nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
                pc = c + nrm * (r[:, None] + 1e-4)
                rem = (1 - th[e, None]) * st[e]
                tan = rem - (rem * nrm).sum(-1, keepdims=True) * nrm
                cand = pc + tan
                again = seg_disc_hit(pc, cand, self.centre[e], self.radius[e])
                new[e] = np.where(again[:, None], pc, cand)
            ae = np.flatnonzero(a)
            self.contact[ae[hit[ae]], self.t[ae[hit[ae]]]] = True
            step = new - self.p
            self.p[a] = new[a]
            self.v[a] = step[a] / V_SCALE
            self.t[a] += 1
            self.pos[ae, self.t[ae]] = self.p[ae]
            r = np.arange(len(self.p))
            near = a & (np.linalg.norm(self.p - self.wps[r, self.w], axis=-1) < GOAL_TOL)
            done = near & (self.w == self.K)
            self.success |= done
            adv = near & (self.w < self.K)
            self.w[adv] += 1
            self.widx[ae, self.t[ae]] = self.w[ae]
            self.alive &= ~done
            self.alive &= self.t < T_MAX


def _jerk_q(pos):
    d2 = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]
    return (d2 ** 2).sum(-1)                                       # (E, T-1), NaN where undefined


def step_jerk(pos: np.ndarray, contact: np.ndarray = None) -> np.ndarray:
    """(E, T+1-2) squared second difference at every interior position; positions next to a
    contact step are masked (NaN): a slide is a contact event, not jerk."""
    q = _jerk_q(pos)
    if contact is not None:
        c = np.zeros((pos.shape[0], pos.shape[1]), bool)          # position t+1 ends contact step t
        c[:, 1:contact.shape[1] + 1] |= contact
        c[:, :contact.shape[1]] |= contact
        near = c[:, 1:-1] | c[:, 2:] | c[:, :-2]
        q = np.where(near, np.nan, q)
    return q


def jerk_windows(pos, contact=None) -> np.ndarray:
    """(E, n_windows) mean squared second difference over full windows of JERK_W (else 0)."""
    q = step_jerk(pos, contact)
    valid = ~np.isnan(q)
    q0 = np.where(valid, q, 0.0)
    cs = np.concatenate([np.zeros((len(q), 1)), np.cumsum(q0, 1)], 1)
    cv = np.concatenate([np.zeros((len(q), 1)), np.cumsum(valid, 1)], 1)
    s = cs[:, JERK_W:] - cs[:, :-JERK_W]
    n = cv[:, JERK_W:] - cv[:, :-JERK_W]
    return np.where(n == JERK_W, s / JERK_W, 0.0)


def jerk_score(pos, contact=None) -> np.ndarray:
    w = jerk_windows(pos, contact)
    return w.max(1) if w.shape[1] else np.zeros(len(pos))


def chunk_jerk(act: np.ndarray, prev_vel: np.ndarray = None) -> np.ndarray:
    """(N,) jerk score of one normalised chunk in free space (as robo2d)."""
    vel = act.reshape(-1, H, 2) * V_SCALE
    pos = np.concatenate([np.zeros((len(vel), 1, 2)), np.cumsum(vel, 1)], 1)
    if prev_vel is not None:
        pv = np.asarray(prev_vel, np.float64) * V_SCALE
        before = np.where((np.abs(pv) > 0).any(-1, keepdims=True), -pv, np.nan)
        pos = np.concatenate([before[:, None, :], pos], 1)
    q = _jerk_q(pos)
    valid = ~np.isnan(q)
    q0 = np.where(valid, q, 0.0)
    W = min(JERK_W, q.shape[1])
    cs = np.concatenate([np.zeros((len(q), 1)), np.cumsum(q0, 1)], 1)
    return ((cs[:, W:] - cs[:, :-W]) / W).max(1)


def events(pos, contact):
    """Per-episode events [(type, onset step)]: contact onsets (no contact in the previous H steps)
    and jerk onsets (first step of a run of windows above JERK_THR; runs H apart)."""
    jw = jerk_windows(pos, contact) > JERK_THR
    out = []
    for e in range(len(pos)):
        ev = []
        last = -10 * H
        for t in np.flatnonzero(contact[e]):
            if t - last > H:
                ev.append(("contact", int(t)))
            last = t
        last = -10 * H
        for t in np.flatnonzero(jw[e]):
            if t - last > H:
                ev.append(("jerk", int(t)))               # window t spans second differences at positions t+1..t+W
            last = t
        out.append(sorted(ev, key=lambda x: x[1]))
    return out


def label(ep: Episodes):
    """Trajectory labeller: (negative, contact, jerky, jerk score)."""
    js = jerk_score(ep.pos, ep.contact)
    jerky = js > JERK_THR
    col = ep.collided
    return col | jerky, col, jerky, js


def homotopy_segments(st: Setups, idx, pos, widx, success) -> np.ndarray:
    """(E, K) per-segment side code (bit b: the b-th obstacle of the segment along its axis passed
    on the left); -1 where the segment was not completed."""
    idx = np.asarray(idx)
    K = st.K
    out = np.full((len(idx), K), -1, dtype=int)
    for e in range(len(idx)):
        s = idx[e]
        valid = ~np.isnan(pos[e, :, 0])
        for k in range(K):
            done = (widx[e][valid] > k + 1).any() if k < K - 1 else bool(success[e])
            if not done:
                continue
            m = valid & (widx[e] == k + 1)
            P = pos[e][m]
            if len(P) < 2:
                continue
            a, b = st.wps[s, k], st.wps[s, k + 1]
            ax = (b - a) / np.linalg.norm(b - a)
            sl = slice(k * SEG_OBS, (k + 1) * SEG_OBS)
            cen, rad = st.centre[s, sl], st.radius[s, sl]
            pres = np.flatnonzero(rad > 0)
            order = pres[np.argsort((cen[pres] - a) @ ax)]
            code = 0
            for bit, j in enumerate(order):
                d = np.linalg.norm(P - cen[j], axis=-1)
                q = P[d.argmin()] - cen[j]
                if ax[0] * q[1] - ax[1] * q[0] > 0:
                    code |= 1 << bit
            out[e, k] = code
    return out


def path_length(pos) -> np.ndarray:
    d = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
    return np.nansum(d, 1)
