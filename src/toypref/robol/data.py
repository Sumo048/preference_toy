"""Experiment courses, base training, rollouts, logged negatives, window oracles, unlabelled pool.

Training courses: N_LABELABLE courses that may carry negatives (ids 0..N_LABELABLE-1; the data curve
uses the first N_s of them, nested) and N_NONEG courses that never do. Held-out courses are new to
the base and to every method. The base is trained on N_BASE_SETUPS base-only courses plus every
training course (DEMOS_PER_SETUP demos each), never on held-out ones, so one base serves every N_s.

What a METHOD may read: ``NegRows`` (observation, logged noise, logged chunk, ids) of the rows its
label condition exposes, the unlabelled pool (``Pool``: observation, noise, chunk of base rollouts,
NO labels), and the frozen base. ``NegDiag`` and the pool's labels are PRIVILEGED (metrics,
diagnostics and the window oracles only).

Window oracles (per negative episode, per event):
  symptom  the chunk containing the event onset
  causal   chunks c in [onset chunk - CAUSAL_BACK, onset chunk] whose noise, resampled CAUSAL_R times
           with every other noise of the episode fixed, removes the event (no event of the same type
           with onset in [c * H, (onset chunk + 2) * H)) at least half the time; no such chunk -> the
           symptom chunk (counted as a fallback)
  human    the causal span shifted by U{-1, 0, 1} chunks with its width changed by U{-1, 0, 1} (>= 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from ..flow import FlowPolicy, EMA
from ..train import cosine
from . import demos as D
from . import env as E
from . import seeds as S
from .policy import ChunkFlow, frozen

N_LABELABLE = 512
N_NONEG = 128
N_HELDOUT = 128
N_PANEL = 32
N_BASE_SETUPS = 2000
DEMOS_PER_SETUP = 8
LOG_BATCH = 32
LOG_ROUNDS = 4
M_EVAL = 32
N_POOL = 8
CAUSAL_BACK = 3
CAUSAL_R = 4
CONTROL_BACK = (6, 7)       # placebo chunks: resampling these far-earlier chunks prices "any perturbation removes it"
NS_LEVELS = (16, 32, 64, 128, 256, 512)


def uid_of(sid, lid, c):
    return int(sid) * 100_000 + int(lid) * 100 + int(c)


# ------------------------------------------------------------------ courses
def feasible_setups(rng: np.random.Generator, n: int, sid0: int) -> E.Setups:
    out = []
    probe = np.random.default_rng(rng.integers(2**62))
    k = 0
    while len(out) < n:
        st = E.sample_setups(rng, 1, sid0 + k)
        k += 1
        if D.demo_path(st, 0, probe)[0] is not None:
            out.append(st)
    st = E.Setups(*[np.concatenate([getattr(s, f) for s in out]) for f in ("sid", "wps", "centre", "radius", "seg")])
    st.sid = np.arange(sid0, sid0 + n)
    return st


def experiment_setups(seed: int):
    """(train, held): train ids 0..N_LABELABLE+N_NONEG-1 (labelable first), held ids after."""
    rng = S.nprng(seed, S.P_SETUPS)
    n_tr = N_LABELABLE + N_NONEG
    st = feasible_setups(rng, n_tr + N_HELDOUT, 0)
    return st.take(np.arange(n_tr)), st.take(np.arange(n_tr, n_tr + N_HELDOUT))


def n_keep(seed: int) -> np.ndarray:
    k = S.nprng(seed, S.P_KEEP).integers(1, 4, size=N_LABELABLE + N_NONEG)
    k[N_LABELABLE:] = 0
    return k


def base_demos(seed: int, n_setups=N_BASE_SETUPS, **inj) -> D.DemoSet:
    st = feasible_setups(S.nprng(seed, S.P_BASE_SETUPS), n_setups, 1_000_000)
    ds = D.make_demos(st, DEMOS_PER_SETUP, S.nprng(seed, S.P_BASE_DEMOS), **inj)
    train, _ = experiment_setups(seed)
    dt = D.make_demos(train, DEMOS_PER_SETUP, S.nprng(seed, S.P_BASE_DEMOS_TRAIN), **inj)
    dt.demo = dt.demo + ds.demo.max() + 1
    return D.DemoSet(*[np.concatenate([getattr(ds, f), getattr(dt, f)]) for f in D.DemoSet.__dataclass_fields__])


def train_base(seed: int, ds: D.DemoSet, steps=30_000, batch=1024, lr=1e-3, log=print, **kw) -> ChunkFlow:
    torch.manual_seed(seed)
    pol = ChunkFlow(**kw)
    ema = EMA(pol, 0.999)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
    g = S.gen(seed, S.P_BASE_TRAIN)
    O = torch.from_numpy(ds.obs)
    A = torch.from_numpy(ds.act)
    N = O.shape[0]
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine(step, steps, 1000, lr, 1e-5)
        i = torch.randint(0, N, (batch,), generator=g)
        noise = torch.randn(batch, E.ACT_DIM, generator=g)
        t = FlowPolicy.sample_time(batch, g)
        loss = pol.flow_loss(A[i], noise, t, O[i]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        ema.update(pol)
        if step % 2000 == 0 or step == steps - 1:
            log(f"[robol base s{seed}] step {step} loss {float(loss.detach()):.4f}")
    pol.load_state_dict(ema.shadow)
    return frozen(pol)


# ------------------------------------------------------------------ rollouts
@dataclass
class Rollouts:
    idx: np.ndarray
    obs: np.ndarray          # (E, N_CHUNKS, OBS_DIM)
    noise: np.ndarray        # (E, N_CHUNKS, 16)
    act: np.ndarray
    ran: np.ndarray          # (E, N_CHUNKS)
    pos: np.ndarray          # (E, T_MAX + 1, 2)
    widx: np.ndarray         # (E, T_MAX + 1)
    contact: np.ndarray      # (E, T_MAX)
    negative: np.ndarray
    collided: np.ndarray
    jerky: np.ndarray
    jerk: np.ndarray
    success: np.ndarray
    steps: np.ndarray
    clipped: np.ndarray


@torch.no_grad()
def rollout(pol, st: E.Setups, idx, noise) -> Rollouts:
    idx = np.asarray(idx)
    ep = E.Episodes(st, idx)
    En = len(idx)
    noise = torch.as_tensor(noise, dtype=torch.float32)
    obs = np.zeros((En, E.N_CHUNKS, E.OBS_DIM), np.float32)
    act = np.zeros((En, E.N_CHUNKS, E.ACT_DIM), np.float32)
    ran = np.zeros((En, E.N_CHUNKS), bool)
    for c in range(E.N_CHUNKS):
        live = np.flatnonzero(ep.alive)
        if not len(live):
            break
        o = ep.obs()
        a = np.zeros((En, E.ACT_DIM), np.float32)
        a[live] = pol.sample(noise[live, c], torch.from_numpy(o[live])).numpy()
        obs[live, c], act[live, c] = o[live], a[live]
        ran[live, c] = True
        ep.execute(a, c)
    neg, col, jerky, js = E.label(ep)
    return Rollouts(idx, obs, noise.numpy(), act, ran, ep.pos, ep.widx, ep.contact, neg, col, jerky, js,
                    ep.success, ep.t.copy(), ep.clipped.copy())


def rollout_chunked(pol, st, idx, noise, batch=8192) -> Rollouts:
    """Same as ``rollout`` in batches (memory); results concatenated."""
    idx = np.asarray(idx)
    parts = [rollout(pol, st, idx[i:i + batch], noise[i:i + batch]) for i in range(0, len(idx), batch)]
    return Rollouts(*[np.concatenate([getattr(p, f) for p in parts]) for f in Rollouts.__dataclass_fields__])


# ------------------------------------------------------------------ logged negatives
@dataclass
class NegRows:
    obs: torch.Tensor
    xi: torch.Tensor
    a: torch.Tensor
    uid: np.ndarray
    setup: np.ndarray
    episode: np.ndarray
    chunk: np.ndarray

    def __len__(self):
        return int(self.obs.shape[0])

    def take(self, m) -> "NegRows":
        m = np.asarray(m)
        tm = torch.from_numpy(m) if m.dtype == bool else torch.as_tensor(m)
        return NegRows(self.obs[tm], self.xi[tm], self.a[tm], self.uid[m], self.setup[m], self.episode[m], self.chunk[m])


@dataclass
class NegDiag:
    """PRIVILEGED per-row facts."""
    sym: np.ndarray          # symptom-window row
    cause: np.ndarray        # causal-window row
    human: np.ndarray        # human-style window row
    ep_collided: np.ndarray
    ep_jerky: np.ndarray

    def take(self, m) -> "NegDiag":
        return NegDiag(*[getattr(self, f)[m] for f in self.__dataclass_fields__])


@dataclass
class LogResult:
    rows: NegRows
    diag: NegDiag
    episodes: Dict           # per kept negative episode: setup, noise, log id, events, causal stats
    per_setup: Dict


def log_negatives(ref, seed: int, train: E.Setups) -> LogResult:
    keep = n_keep(seed)
    need = np.flatnonzero(keep > 0)
    found: Dict[int, List] = {int(s): [] for s in need}
    noises = {int(s): torch.randn(LOG_BATCH * LOG_ROUNDS, E.N_CHUNKS, E.ACT_DIM,
                                  generator=S.gen(seed, S.P_LOG + int(train.sid[s]))) for s in need}
    stats = {"n_logged": np.zeros(len(train), int), "n_neg_seen": np.zeros(len(train), int), "wanted": keep.copy()}
    eps = []                                         # (setup, log id, Rollouts, j)
    for r in range(LOG_ROUNDS):
        todo = [int(s) for s in need if len(found[int(s)]) < keep[s]]
        if not todo:
            break
        idx = np.repeat(todo, LOG_BATCH)
        nz = torch.cat([noises[s][r * LOG_BATCH:(r + 1) * LOG_BATCH] for s in todo])
        ro = rollout_chunked(ref, train, idx, nz)
        for n, s in enumerate(todo):
            stats["n_logged"][s] += LOG_BATCH
            negs = np.flatnonzero(ro.negative[n * LOG_BATCH:(n + 1) * LOG_BATCH])
            stats["n_neg_seen"][s] += len(negs)
            for e in negs:
                if len(found[s]) >= keep[s]:
                    break
                found[s].append(r * LOG_BATCH + e)
                eps.append((s, r * LOG_BATCH + e, ro, n * LOG_BATCH + e))
    stats["n_kept"] = np.array([len(found.get(s, [])) for s in range(len(train))])
    ep_setup = np.array([e[0] for e in eps])
    ep_lid = np.array([e[1] for e in eps])
    ep_noise = np.stack([e[2].noise[e[3]] for e in eps])
    take = lambda f: np.stack([getattr(e[2], f)[e[3]] for e in eps])
    rec = {f: take(f) for f in ("obs", "act", "ran", "pos", "contact", "collided", "jerky")}
    ev = E.events(rec["pos"], rec["contact"])
    return _with_windows(ref, seed, train, ep_setup, ep_lid, ep_noise, rec, ev, stats)


def _with_windows(ref, seed, train, ep_setup, ep_lid, ep_noise, rec, ev, stats) -> LogResult:
    n_ep = len(ep_setup)
    # ---- causal oracle: resample one chunk's noise, replay, check the event
    jobs = []                                        # (episode, event idx, chunk, r)
    for e in range(n_ep):
        for k, (typ, t0) in enumerate(ev[e]):
            ce = t0 // E.H
            for c in list(range(max(0, ce - CAUSAL_BACK), ce + 1)) + [ce - b for b in CONTROL_BACK if ce - b >= 0]:
                if rec["ran"][e, c]:
                    for r in range(CAUSAL_R):
                        jobs.append((e, k, c, r))
    nz = np.repeat(ep_noise[[j[0] for j in jobs]][None], 1, 0)[0].copy() if jobs else np.zeros((0, E.N_CHUNKS, E.ACT_DIM), np.float32)
    for n, (e, k, c, r) in enumerate(jobs):
        uid = uid_of(train.sid[ep_setup[e]], ep_lid[e], c)
        nz[n, c] = torch.randn(E.ACT_DIM, generator=S.row_gen(seed, uid, S.ROW_CAUSAL + r)).numpy()
    removed = np.zeros(len(jobs), bool)
    if jobs:
        ro = rollout_chunked(ref, train, ep_setup[[j[0] for j in jobs]], nz)
        ev2 = E.events(ro.pos, ro.contact)
        for n, (e, k, c, r) in enumerate(jobs):
            typ, t0 = ev[e][k]
            hi = (t0 // E.H + 2) * E.H
            removed[n] = not any(tp == typ and c * E.H <= t < hi for tp, t in ev2[n])
    rate: Dict = {}
    for n, (e, k, c, r) in enumerate(jobs):
        rate.setdefault((e, k, c), []).append(removed[n])
    g = S.nprng(seed, S.P_WINDOW)
    rows = {f: [] for f in ("obs", "xi", "a", "uid", "setup", "episode", "chunk", "sym", "cause", "human", "col", "jer")}
    n_fallback = n_events = 0
    lags, ev_store = [], []
    for e in range(n_ep):
        sym = np.zeros(E.N_CHUNKS, bool)
        cause = np.zeros(E.N_CHUNKS, bool)
        human = np.zeros(E.N_CHUNKS, bool)
        ran = rec["ran"][e]
        evs = []
        for k, (typ, t0) in enumerate(ev[e]):
            n_events += 1
            ce = min(t0 // E.H, E.N_CHUNKS - 1)
            sym[ce] = True
            ctl_e = [np.mean(rate[(e, k, ce - b)]) for b in CONTROL_BACK if (e, k, ce - b) in rate]
            evs_ctl = float(np.mean(ctl_e)) if ctl_e else float("nan")
            cs = [c for c in range(max(0, ce - CAUSAL_BACK), ce + 1) if np.mean(rate.get((e, k, c), [0])) >= 0.5]
            if not cs:
                cs = [ce]
                n_fallback += 1
            cause[cs] = True
            lags.append(ce - min(cs))
            lo = min(cs) + int(g.integers(-1, 2))
            w = max(1, (max(cs) - min(cs) + 1) + int(g.integers(-1, 2)))
            lo = max(0, lo)
            human[lo:lo + w] = True
            evs.append({"type": typ, "onset": int(t0), "cause": [int(c) for c in cs], "control": evs_ctl,
                        "removal": {int(ce - c): float(np.mean(rate[(e, k, c)])) for c in range(max(0, ce - CAUSAL_BACK), ce + 1) if (e, k, c) in rate}})
        ev_store.append(evs)
        human &= ran
        sid = train.sid[ep_setup[e]]
        for c in np.flatnonzero(ran):
            rows["obs"].append(rec["obs"][e, c]); rows["xi"].append(ep_noise[e, c]); rows["a"].append(rec["act"][e, c])
            rows["uid"].append(uid_of(sid, ep_lid[e], c)); rows["setup"].append(int(ep_setup[e])); rows["episode"].append(e)
            rows["chunk"].append(int(c)); rows["sym"].append(bool(sym[c])); rows["cause"].append(bool(cause[c]))
            rows["human"].append(bool(human[c])); rows["col"].append(bool(rec["collided"][e])); rows["jer"].append(bool(rec["jerky"][e]))
    nr = NegRows(torch.from_numpy(np.stack(rows["obs"])), torch.from_numpy(np.stack(rows["xi"]).astype(np.float32)),
                 torch.from_numpy(np.stack(rows["a"])), np.array(rows["uid"]), np.array(rows["setup"]),
                 np.array(rows["episode"]), np.array(rows["chunk"]))
    assert len(set(nr.uid.tolist())) == len(nr.uid), "row uids must be unique"
    dg = NegDiag(np.array(rows["sym"]), np.array(rows["cause"]), np.array(rows["human"]), np.array(rows["col"]), np.array(rows["jer"]))
    cr = [np.mean(v) for (e, k, c), v in rate.items() if c in [x for x in ev_store[e][k]["cause"]]]
    ctl = [np.mean(v) for (e, k, c), v in rate.items() if ev[e][k][1] // E.H - c in CONTROL_BACK]
    by_back = {}
    for (e, k, c), v in rate.items():
        by_back.setdefault(int(ev[e][k][1] // E.H - c), []).append(np.mean(v))
    stats["removal_by_back"] = {b: float(np.mean(v)) for b, v in sorted(by_back.items())}
    stats["control_removal_rate"] = float(np.mean(ctl)) if ctl else float("nan")
    stats.update({"n_events": n_events, "causal_fallback": n_fallback / max(n_events, 1),
                  "lag_hist": np.bincount(lags, minlength=CAUSAL_BACK + 1).tolist(),
                  "causal_removal_rate": float(np.mean(cr)) if cr else float("nan")})
    return LogResult(nr, dg, {"setup": ep_setup, "noise": ep_noise, "log_id": ep_lid, "events": ev_store}, stats)


# ------------------------------------------------------------------ unlabelled pool
@dataclass
class Pool:
    """Unlabelled base rollouts on the training courses. obs/xi/a/setup/chunk are readable by methods;
    ``negative`` (per row: its episode is negative) is PRIVILEGED."""
    obs: torch.Tensor
    xi: torch.Tensor
    a: torch.Tensor
    setup: np.ndarray
    episode: np.ndarray
    chunk: np.ndarray
    negative: np.ndarray


def unlabelled_pool(ref, seed: int, train: E.Setups) -> Pool:
    idx = np.repeat(np.arange(len(train)), N_POOL)
    nz = torch.cat([torch.randn(N_POOL, E.N_CHUNKS, E.ACT_DIM, generator=S.gen(seed, S.P_UNLAB + int(train.sid[s])))
                    for s in range(len(train))])
    ro = rollout_chunked(ref, train, idx, nz)
    e, c = np.nonzero(ro.ran)
    return Pool(torch.from_numpy(ro.obs[e, c]), torch.from_numpy(ro.noise[e, c]), torch.from_numpy(ro.act[e, c]),
                idx[e], e, c, ro.negative[e])


# ------------------------------------------------------------------ re-collected negatives (online rounds)
P_RELOG = 200_000              # + round * 1000 + setup id


def collect_negatives(pol, seed: int, train: E.Setups, setups, round_id: int, labels: str = "traj") -> NegRows:
    """New negative rows from ``pol`` on the given training courses: up to n_keep(seed) negative episodes per
    course from LOG_ROUNDS x LOG_BATCH rollouts (fresh noise streams per round_id), labelled by the env labeller.
    labels: traj (every chunk) | approach (per event: the onset chunk and the 2 before it). Courses where the
    policy no longer fails contribute nothing. Row uids live in their own range (>= 1e9) so row streams are new."""
    keep = n_keep(seed)
    need = [int(s) for s in setups if keep[s] > 0]
    found: Dict[int, int] = {s: 0 for s in need}
    rows = {k: [] for k in ("obs", "xi", "a", "uid", "setup", "episode", "chunk")}
    n_ep = 0
    for r in range(LOG_ROUNDS):
        todo = [s for s in need if found[s] < keep[s]]
        if not todo:
            break
        nz = torch.cat([torch.randn(LOG_BATCH, E.N_CHUNKS, E.ACT_DIM,
                                    generator=S.gen(seed, P_RELOG + round_id * 1000 + int(train.sid[s]) + 100_000 * r)) for s in todo])
        ro = rollout_chunked(pol, train, np.repeat(todo, LOG_BATCH), nz)
        evs = E.events(ro.pos, ro.contact) if labels == "approach" else None
        for n, s in enumerate(todo):
            for e in np.flatnonzero(ro.negative[n * LOG_BATCH:(n + 1) * LOG_BATCH]):
                if found[s] >= keep[s]:
                    break
                j = n * LOG_BATCH + e
                ch = np.flatnonzero(ro.ran[j])
                if labels == "approach":
                    sel = np.zeros(E.N_CHUNKS, bool)
                    for _, t0 in evs[j]:
                        hi = min(t0 // E.H, E.N_CHUNKS - 1)
                        sel[max(0, hi - 2):hi + 1] = True
                    ch = [c for c in ch if sel[c]]
                for c in ch:
                    rows["obs"].append(ro.obs[j, c]); rows["xi"].append(ro.noise[j, c]); rows["a"].append(ro.act[j, c])
                    rows["uid"].append(10**9 + round_id * 10**8 + int(train.sid[s]) * 10**5 + (r * LOG_BATCH + int(e)) * 100 + int(c))
                    rows["setup"].append(s); rows["episode"].append(n_ep); rows["chunk"].append(int(c))
                found[s] += 1
                n_ep += 1
    if not rows["uid"]:
        z = torch.zeros(0, E.OBS_DIM), torch.zeros(0, E.ACT_DIM)
        return NegRows(z[0], z[1], z[1].clone(), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int))
    return NegRows(torch.from_numpy(np.stack(rows["obs"])), torch.from_numpy(np.stack(rows["xi"]).astype(np.float32)),
                   torch.from_numpy(np.stack(rows["a"])), np.array(rows["uid"]), np.array(rows["setup"]),
                   np.array(rows["episode"]), np.array(rows["chunk"]))
