"""Experiment setups, base training, rollouts and the logged negatives.

What a METHOD may read is exactly ``NegRows`` (observation, logged noise, logged action and ids
of every chunk of every logged negative episode) plus the frozen base policy, which it may query
at any observation. ``NegDiag`` holds privileged per-row facts (which chunk actually crashed or
jerked, failure type) and is used ONLY by diagnostics and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ..flow import FlowPolicy
from ..train import cosine
from ..flow import EMA
from . import demos as D
from . import env as E
from . import seeds as S
from .policy import ChunkFlow, frozen

N_TRAIN_SETUPS = 160
N_HELDOUT_SETUPS = 40
NONEG_FRAC = 0.2
N_BASE_SETUPS = 4000
DEMOS_PER_SETUP = 8
LOG_BATCH = 32          # logged rollouts per round per training setup
LOG_ROUNDS = 4          # at most 128 logged rollouts per setup
M_EVAL = 64             # CRN eval rollouts per setup


# ------------------------------------------------------------------ setups
def feasible_setups(rng: np.random.Generator, n: int, sid0: int) -> E.Setups:
    """n setups that admit at least one clean demo (others are resampled)."""
    out = []
    probe = np.random.default_rng(rng.integers(2**62))
    k = 0
    while len(out) < n:
        st = E.sample_setups(rng, 1, sid0 + k)
        k += 1
        if D.demo_path(st, 0, probe)[0] is not None:
            out.append(st)
    cat = lambda f: np.concatenate([getattr(s, f) for s in out])
    st = E.Setups(cat("sid"), cat("start"), cat("goal"), cat("centre"), cat("radius"), cat("n_obs"))
    st.sid = np.arange(sid0, sid0 + n)
    return st


def experiment_setups(seed: int):
    """(train, heldout) setups of an experiment seed; ids 0..159 train, 160..199 held out."""
    rng = S.nprng(seed, S.P_SETUPS)
    st = feasible_setups(rng, N_TRAIN_SETUPS + N_HELDOUT_SETUPS, 0)
    return st.take(np.arange(N_TRAIN_SETUPS)), st.take(np.arange(N_TRAIN_SETUPS, N_TRAIN_SETUPS + N_HELDOUT_SETUPS))


def noneg_setups(seed: int) -> np.ndarray:
    """Indices (into the training setups) that log NO negatives (20%, by seed)."""
    k = int(round(NONEG_FRAC * N_TRAIN_SETUPS))
    return np.sort(S.nprng(seed, S.P_NONEG).permutation(N_TRAIN_SETUPS)[:k])


def n_keep(seed: int) -> np.ndarray:
    """Negatives each training setup keeps (1-3; zero for the no-negative setups)."""
    k = S.nprng(seed, S.P_KEEP).integers(1, 4, size=N_TRAIN_SETUPS)
    k[noneg_setups(seed)] = 0
    return k


# ------------------------------------------------------------------ base
def base_demos(seed: int, q_col=D.Q_COL, q_jerk=D.Q_JERK, n_setups=N_BASE_SETUPS) -> D.DemoSet:
    st = feasible_setups(S.nprng(seed, S.P_BASE_SETUPS), n_setups, 1_000_000)
    return D.make_demos(st, DEMOS_PER_SETUP, S.nprng(seed, S.P_BASE_DEMOS), q_col=q_col, q_jerk=q_jerk)


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
            log(f"[robo base s{seed}] step {step} loss {float(loss.detach()):.4f}")
    pol.load_state_dict(ema.shadow)
    return frozen(pol)


# ------------------------------------------------------------------ rollouts
@dataclass
class Rollouts:
    idx: np.ndarray          # (E,) setup index
    obs: np.ndarray          # (E, 6, 13) observation at each chunk start (valid where ran)
    noise: np.ndarray        # (E, 6, 16)
    act: np.ndarray          # (E, 6, 16) decoded chunk
    ran: np.ndarray          # (E, 6) chunk was executed
    pos: np.ndarray          # (E, 49, 2)
    negative: np.ndarray
    collided: np.ndarray
    jerky: np.ndarray
    jerk: np.ndarray
    success: np.ndarray
    col_chunk: np.ndarray
    steps: np.ndarray        # executed steps
    clipped: np.ndarray


@torch.no_grad()
def rollout(pol: ChunkFlow, st: E.Setups, idx, noise) -> Rollouts:
    """Closed loop at chunk level: decode a chunk from the CURRENT observation with the given
    noise, execute it open loop, replan. ``noise`` (E, 6, 16)."""
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
    return Rollouts(idx, obs, noise.numpy(), act, ran, ep.pos, neg, col, jerky, js, ep.success,
                    ep.col_chunk, ep.t.copy(), ep.clipped.copy())


def chunk_jerk_in_episode(pos: np.ndarray, ran: np.ndarray) -> np.ndarray:
    """(E, 6) jerk score of each chunk's steps INCLUDING its boundary with the previous chunk."""
    En = pos.shape[0]
    out = np.zeros((En, E.N_CHUNKS))
    for c in range(E.N_CHUNKS):
        lo, hi = max(c * E.H - 1, 0), min((c + 1) * E.H + 1, pos.shape[1])
        seg = pos[:, lo:hi]
        d2 = seg[:, 2:] - 2 * seg[:, 1:-1] + seg[:, :-2]
        q = (d2 ** 2).sum(-1)
        out[:, c] = np.where(np.isnan(q), -np.inf, q).max(1)
    out[~ran] = -np.inf
    return out


# ------------------------------------------------------------------ logged negatives
@dataclass
class NegRows:
    """Everything a method may read about the logged negatives (one row per chunk)."""
    obs: torch.Tensor        # (N, 13)
    xi: torch.Tensor         # (N, 16) logged noise
    a: torch.Tensor          # (N, 16) logged (decoded) chunk
    uid: np.ndarray          # (N,) setup * 10_000 + logged episode * 10 + chunk
    setup: np.ndarray        # (N,)
    episode: np.ndarray      # (N,) global negative-episode id
    chunk: np.ndarray        # (N,)

    def __len__(self):
        return int(self.obs.shape[0])

    def take(self, m) -> "NegRows":
        m = np.asarray(m)
        tm = torch.from_numpy(m) if m.dtype == bool else torch.as_tensor(m)
        return NegRows(self.obs[tm], self.xi[tm], self.a[tm], self.uid[m], self.setup[m], self.episode[m], self.chunk[m])


@dataclass
class NegDiag:
    """PRIVILEGED per-row facts; diagnostics and metrics only."""
    bad_chunk: np.ndarray    # (N,) this chunk is where the episode crashed / jerked hardest
    ep_collided: np.ndarray
    ep_jerky: np.ndarray


@dataclass
class LogResult:
    rows: NegRows
    diag: NegDiag
    episodes: Dict           # per kept negative episode: setup, noise sequence (for replay)
    per_setup: Dict          # logging stats


def log_negatives(ref: ChunkFlow, seed: int, train: E.Setups) -> LogResult:
    keep = n_keep(seed)
    obs, xi, a, uid, sup, epi, chk = [], [], [], [], [], [], []
    bad, ecol, ejer = [], [], []
    ep_setup, ep_noise, ep_log = [], [], []
    stats = {"n_logged": np.zeros(len(train), int), "n_neg_seen": np.zeros(len(train), int),
             "n_kept": np.zeros(len(train), int), "wanted": keep.copy()}
    # all setups' first round in one batch, further rounds only where still short
    need = np.flatnonzero(keep > 0)
    found: Dict[int, List] = {int(s): [] for s in need}
    noises = {int(s): torch.randn(LOG_BATCH * LOG_ROUNDS, E.N_CHUNKS, E.ACT_DIM,
                                  generator=S.gen(seed, S.P_LOG + int(train.sid[s]))) for s in need}
    for r in range(LOG_ROUNDS):
        todo = [int(s) for s in need if len(found[int(s)]) < keep[s]]
        if not todo:
            break
        idx = np.repeat(todo, LOG_BATCH)
        nz = torch.cat([noises[s][r * LOG_BATCH:(r + 1) * LOG_BATCH] for s in todo])
        ro = rollout(ref, train, idx, nz)
        cj = chunk_jerk_in_episode(ro.pos, ro.ran)
        for n, s in enumerate(todo):
            sl = slice(n * LOG_BATCH, (n + 1) * LOG_BATCH)
            stats["n_logged"][s] += LOG_BATCH
            negs = np.flatnonzero(ro.negative[sl])
            stats["n_neg_seen"][s] += len(negs)
            for e in negs:
                if len(found[s]) >= keep[s]:
                    break
                found[s].append((r * LOG_BATCH + e, n * LOG_BATCH + e, ro, cj))
        # rows from this round's kept episodes
        for s in todo:
            for (lid, j, ro_, cj_) in found[s]:
                if ro_ is not ro:
                    continue
                g_ep = len(ep_setup)
                ep_setup.append(s)
                ep_noise.append(ro.noise[j])
                ep_log.append(lid)
                bc = np.zeros(E.N_CHUNKS, bool)
                if ro.collided[j]:
                    bc[ro.col_chunk[j]] = True
                if ro.jerky[j]:
                    bc[int(np.argmax(cj_[j]))] = True
                for c in np.flatnonzero(ro.ran[j]):
                    obs.append(ro.obs[j, c]); xi.append(ro.noise[j, c]); a.append(ro.act[j, c])
                    uid.append(int(train.sid[s]) * 10_000 + lid * 10 + int(c))
                    sup.append(s); epi.append(g_ep); chk.append(int(c))
                    bad.append(bool(bc[c])); ecol.append(bool(ro.collided[j])); ejer.append(bool(ro.jerky[j]))
    for s in need:
        stats["n_kept"][s] = len(found[int(s)])
    rows = NegRows(torch.from_numpy(np.stack(obs)), torch.from_numpy(np.stack(xi)), torch.from_numpy(np.stack(a)),
                   np.array(uid), np.array(sup), np.array(epi), np.array(chk))
    diag = NegDiag(np.array(bad), np.array(ecol), np.array(ejer))
    assert len(set(rows.uid.tolist())) == len(rows.uid), "row uids must be unique"
    return LogResult(rows, diag, {"setup": np.array(ep_setup), "noise": np.stack(ep_noise), "log_id": np.array(ep_log)}, stats)


def setup_groups(seed: int, log: LogResult, train: E.Setups):
    """(with-negatives, without-negatives) training-setup indices."""
    has = np.zeros(len(train), bool)
    has[np.unique(log.rows.setup)] = True
    return np.flatnonzero(has), np.flatnonzero(~has)
