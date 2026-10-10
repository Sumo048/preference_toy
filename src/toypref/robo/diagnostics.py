"""PRIVILEGED diagnostics (never read by a method): separability of the bad chunk at its state and
the oracle quality of rerouting targets.

oracle_chunk_bad(obs, act): execute a chunk open loop from the state encoded in ``obs`` (the
observation carries the position, all <= 3 obstacles and the previous step velocity) and flag a
collision or a chunk-level jerk above the threshold. The boundary second difference with the
previous step is included (the observation now carries that velocity), except at the episode start.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from . import env as E
from . import targets as TG
from .data import NegDiag, NegRows


def _geometry(obs: np.ndarray):
    p = obs[:, :2].astype(np.float64)
    ob = obs[:, 4:E.OBS_VEL].reshape(-1, E.MAX_OBS, 3).astype(np.float64)
    rad = ob[..., 2]
    cen = np.where(rad[..., None] > 0, ob[..., :2] + p[:, None, :], E.FAR)
    return p, cen, rad


def oracle_chunk_bad(obs, act) -> (np.ndarray, np.ndarray):
    obs = np.asarray(obs); act = np.asarray(act)
    p, cen, rad = _geometry(obs)
    vel = act.reshape(-1, E.H, 2).astype(np.float64) * E.V_SCALE
    n = np.linalg.norm(vel, axis=-1, keepdims=True)
    vel = np.where(n > E.MAX_STEP, vel * E.MAX_STEP / np.maximum(n, 1e-12), vel)
    hit = np.zeros(len(p), bool)
    q = p.copy()
    for h in range(E.H):
        new = q + vel[:, h]
        hit |= E.seg_disc_hit(q, new, cen, rad)
        q = new
    jerk = E.chunk_jerk(act, obs[:, E.OBS_VEL:E.OBS_VEL + 2]) > E.JERK_THR
    return hit, jerk


@torch.no_grad()
def separability(ref, seed, rows: NegRows, diag: NegDiag, cache: TG.HopCache = None) -> Dict:
    out: Dict = {}
    bad = np.flatnonzero(diag.bad_chunk)
    if not len(bad):
        return out
    sub = rows.take(bad)
    bas = TG.all_basins(ref, seed, sub)
    own, purity, recall, tail, ntyp, ncl, neg_or = [], [], [], [], [], [], []
    for n, b in enumerate(bas):
        o = np.repeat(sub.obs[n:n + 1].numpy(), len(b.dec), 0)
        hc, hj = oracle_chunk_bad(o, b.dec.numpy())
        ob = (hc | hj)
        typ = b.typ.numpy()
        lab_all = -np.ones(len(ob), int)
        lab_all[typ] = b.lab
        in_bad = lab_all == b.bad
        own.append(b.n_clusters >= 2 and b.bad_frac <= 0.5)
        ncl.append(b.n_clusters)
        ntyp.append(b.neg_typical)
        tail.append(ob.mean())
        purity.append(ob[in_bad].mean() if in_bad.any() else np.nan)
        recall.append(in_bad[ob & typ].mean() if (ob & typ).any() else np.nan)
        c, j = oracle_chunk_bad(sub.obs[n:n + 1].numpy(), sub.a[n:n + 1].numpy())
        neg_or.append(bool(c[0] | j[0]))
    own, ncl = np.array(own), np.array(ncl)
    col = diag.ep_collided[bad]
    out.update({"sep_n_bad_chunks": int(len(bad)),
                "sep_own_basin": float(own.mean()),
                "sep_own_basin_collision": float(own[col].mean()) if col.any() else float("nan"),
                "sep_own_basin_jerk": float(own[~col].mean()) if (~col).any() else float("nan"),
                "sep_single_cluster": float((ncl == 1).mean()),
                "sep_mean_clusters": float(ncl.mean()),
                "sep_neg_typical": float(np.mean(ntyp)),
                "sep_tail_rate_at_state": float(np.nanmean(tail)),
                "sep_bad_basin_purity": float(np.nanmean(purity)),
                "sep_bad_basin_recall": float(np.nanmean(recall)),
                "sep_logged_chunk_oracle_bad": float(np.mean(neg_or))})
    if cache is not None:
        hc, hj = oracle_chunk_bad(rows.obs.numpy(), cache.target.numpy())
        tb = hc | hj
        out["hop_target_oracle_bad"] = float(tb.mean())
        out["hop_target_oracle_bad_on_bad_chunks"] = float(tb[bad].mean())
        out["all_rows_single_cluster"] = float(np.mean(cache.n_clusters == 1))
        out["bad_rows_single_cluster"] = float(np.mean(cache.n_clusters[bad] == 1))
        out["other_rows_single_cluster"] = float(np.mean(cache.n_clusters[~diag.bad_chunk] == 1))
    return out
