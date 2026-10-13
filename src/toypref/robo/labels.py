"""Label conditions for the stress round: how the logged negatives a method reads are corrupted.

A condition maps the native ``LogResult`` of a seed to the rows a METHOD reads. Evaluation never
changes: groups, CRN noises and the replay set always come from the native log, so every condition
of a seed is scored on the same episodes against the same base rollout.

  native      the logged negatives as logged (every chunk of a negative episode is a row)
  oracle      only the chunk that actually crashed / jerked (PRIVILEGED; a reference, not a method)
  pad{K}      every negative episode also carries K good chunks from clean base episodes of the same
              setup, as if the trajectory were longer and the bad part a smaller share of it
  flip{f}     a share f of the negative episodes is replaced by a clean episode of the same setup
              (false labels; the count of labelled episodes is unchanged). Nested: flip0.1 is a
              subset of flip0.25 is a subset of flip0.5.
  placebo     flip1.0: every labelled episode is clean -- same pipeline, no information
  recall{r}   only a share r of the negative episodes is labelled (missed failures)
  cover{c}    only the negatives of a share c of the setups are labelled

Clean episodes: base rollouts on the stream (seed, P_CLEAN + setup id), never the eval or log noises;
"clean" = success and not negative. Every draw is keyed by (seed, purpose, episode / setup), never by
condition, so conditions differ only by the knob.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import torch

from . import data as DT
from . import env as E
from . import seeds as S

P_CLEAN = 80_000          # + setup id: clean base rollouts used for padding and flips
P_FLIP = 81_000           # per-episode uniform for flips
P_RECALL = 82_000         # per-episode uniform for recall
P_COVER = 83_000          # per-setup uniform for coverage
N_CLEAN = 64              # clean-candidate rollouts per setup
CLEAN_LID0 = 500          # uid offset of clean rows (logged ids are < 128)

CONDITIONS = ["native", "oracle", "pad6", "pad18", "flip0.1", "flip0.25", "flip0.5", "placebo",
              "recall0.5", "cover0.25"]


def _u(seed: int, purpose: int, n: int) -> np.ndarray:
    return S.nprng(seed, purpose).random(n)


@torch.no_grad()
def clean_episodes(ref, seed: int, train: E.Setups, setups) -> Dict[int, list]:
    """setup -> list of clean episodes; each is a list of (obs, xi, a, uid) per executed chunk."""
    setups = [int(s) for s in setups]
    idx = np.repeat(setups, N_CLEAN)
    nz = torch.cat([torch.randn(N_CLEAN, E.N_CHUNKS, E.ACT_DIM, generator=S.gen(seed, P_CLEAN + int(train.sid[s])))
                    for s in setups])
    ro = DT.rollout(ref, train, idx, nz)
    out: Dict[int, list] = {s: [] for s in setups}
    for n, s in enumerate(setups):
        for e in range(N_CLEAN):
            j = n * N_CLEAN + e
            if ro.negative[j] or not ro.success[j]:
                continue
            ep = [(ro.obs[j, c], ro.noise[j, c], ro.act[j, c], int(train.sid[s]) * 10_000 + (CLEAN_LID0 + e) * 10 + int(c))
                  for c in np.flatnonzero(ro.ran[j])]
            out[s].append(ep)
    return out


def apply(cond: str, ref, seed: int, train: E.Setups, log: DT.LogResult) -> Tuple[DT.NegRows, DT.NegDiag, Dict]:
    R, Dg = log.rows, log.diag
    eps = np.unique(R.episode)                        # global episode ids (not contiguous in a smoke subset)
    n_ep = int(R.episode.max()) + 1                   # per-episode uniforms are indexed by the global id
    ep_setup = np.zeros(n_ep, int)
    ep_setup[R.episode] = R.setup
    info: Dict = {"cond": cond, "native_rows": len(R), "native_episodes": int(len(eps))}
    if cond == "native":
        return R, Dg, info
    if cond == "oracle":
        m = Dg.bad_chunk
        info["rows"] = int(m.sum())
        return R.take(m), DT.NegDiag(Dg.bad_chunk[m], Dg.ep_collided[m], Dg.ep_jerky[m]), info
    if (mm := re.fullmatch(r"recall([\d.]+)", cond)):
        keep_ep = _u(seed, P_RECALL, n_ep) < float(mm.group(1))
        m = keep_ep[R.episode]
        info.update(rows=int(m.sum()), episodes=int(keep_ep[eps].sum()))
        return R.take(m), DT.NegDiag(Dg.bad_chunk[m], Dg.ep_collided[m], Dg.ep_jerky[m]), info
    if (mm := re.fullmatch(r"cover([\d.]+)", cond)):
        keep_s = _u(seed, P_COVER, len(train)) < float(mm.group(1))
        m = keep_s[R.setup]
        info.update(rows=int(m.sum()), setups=int(len(np.unique(R.setup[m]))))
        return R.take(m), DT.NegDiag(Dg.bad_chunk[m], Dg.ep_collided[m], Dg.ep_jerky[m]), info

    pad, flip = 0, 0.0
    if (mm := re.fullmatch(r"pad(\d+)", cond)):
        pad = int(mm.group(1))
    elif (mm := re.fullmatch(r"flip([\d.]+)", cond)):
        flip = float(mm.group(1))
    elif cond == "placebo":
        flip = 1.0
    else:
        raise ValueError(cond)
    clean = clean_episodes(ref, seed, train, np.unique(ep_setup[eps]))
    nxt = {s: 0 for s in clean}                       # next unused clean episode per setup
    u = _u(seed, P_FLIP, n_ep)
    rows = {k: [] for k in ("obs", "xi", "a", "uid", "setup", "episode", "chunk", "bad", "col", "jer")}

    def add(o, z, a, uid, s, e, c, bad, col, jer):
        for k, v in zip(rows, (o, z, a, uid, s, e, c, bad, col, jer)):
            rows[k].append(v)

    flipped = short = pad_rows = 0
    for e in eps:
        s = int(ep_setup[e])
        if flip and u[e] < flip:
            if nxt[s] < len(clean[s]):
                for c, (o, z, a, uid) in enumerate(clean[s][nxt[s]]):
                    add(o, z, a, uid, s, e, c, False, False, False)
                nxt[s] += 1
                flipped += 1
                continue
            short += 1                                 # no clean episode left: keep the negative
        m = np.flatnonzero(R.episode == e)
        for i in m:
            add(R.obs[i].numpy(), R.xi[i].numpy(), R.a[i].numpy(), int(R.uid[i]), s, e, int(R.chunk[i]),
                bool(Dg.bad_chunk[i]), bool(Dg.ep_collided[i]), bool(Dg.ep_jerky[i]))
        k = 0
        while pad and k < pad and nxt[s] < len(clean[s]):
            for c, (o, z, a, uid) in enumerate(clean[s][nxt[s]]):
                if k >= pad:
                    break
                add(o, z, a, uid, s, e, E.N_CHUNKS + k, False, bool(Dg.ep_collided[m[0]]), bool(Dg.ep_jerky[m[0]]))
                k += 1
            nxt[s] += 1
        pad_rows += k
        short += int(pad > 0 and k < pad)
    nr = DT.NegRows(torch.from_numpy(np.stack(rows["obs"])), torch.from_numpy(np.stack(rows["xi"])),
                    torch.from_numpy(np.stack(rows["a"])), np.array(rows["uid"]), np.array(rows["setup"]),
                    np.array(rows["episode"]), np.array(rows["chunk"]))
    assert len(set(nr.uid.tolist())) == len(nr.uid), "row uids must be unique"
    info.update(rows=len(nr), flipped_episodes=flipped, pad_rows=pad_rows, short_episodes=short,
                bad_chunk_share=float(np.mean(rows["bad"])),
                clean_eps_per_setup=float(np.mean([len(v) for v in clean.values()])))
    return nr, DT.NegDiag(np.array(rows["bad"]), np.array(rows["col"]), np.array(rows["jer"])), info
