"""CRN evaluation on long courses. Every policy of a seed is rolled out on the same courses with the
same noise sequences; every paired metric is against the base rollout computed in the SAME process
(closed-loop rollouts amplify CPU-dependent last bits, so a base from another node is no pair).

Groups: ``panel`` the first N_PANEL labelable courses (labelled at every N_s), ``noneg`` the training
courses that never carry negatives, ``held`` held-out courses.

Metrics per group (prefix ``{group}_``):
  success_clean   last waypoint reached with no contact and no jerk
  bad / collision / jerk / success
  seg_clean       share of segments traversed with no event (unreached segments count as not clean)
  seg_reach       share of waypoints reached
  good_broken     share of base-clean episodes that the policy makes not clean
  bad_fixed       share of base-bad episodes that the policy makes clean
  mode_tv         mean over (course, segment) of the TV between the policy's and the base's homotopy
                  shares on that segment (episodes that completed it; >= MIN_TV each)
  speed_ratio     mean step length (free-space steps) / the base's
  retained_disp   RMS change of the chunk at the base's own observations and noises (base-clean episodes)
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import torch

from . import data as DT
from . import env as E
from . import seeds as S

MIN_TV = 5


def seg_clean(ro: DT.Rollouts, K: int, events) -> np.ndarray:
    """(E, K) segment k completed with no event onset while on it."""
    En = len(ro.idx)
    out = np.zeros((En, K), bool)
    for e in range(En):
        valid = ~np.isnan(ro.pos[e, :, 0])
        w = ro.widx[e]
        last = int(valid.sum()) - 1
        bad_steps = [t for _, t in events[e]]
        for k in range(K):
            on = np.flatnonzero(valid & (w == k + 1))
            if not len(on):
                continue
            done = (w[valid] > k + 1).any() if k < K - 1 else bool(ro.success[e])
            if not done:
                continue
            lo, hi = on[0], on[-1] + 1
            out[e, k] = not any(lo <= t <= hi for t in bad_steps)
    return out


def free_speed(ro: DT.Rollouts) -> np.ndarray:
    d = np.linalg.norm(np.diff(ro.pos, axis=1), axis=-1)            # (E, T_MAX)
    d = np.where(ro.contact, np.nan, d)
    return np.nanmean(d, 1)


class EvalSet:
    def __init__(self, ref, seed, train: E.Setups, held: E.Setups, log: DT.LogResult, M=DT.M_EVAL):
        self.st = E.concat(train, held)
        n_tr = len(train)
        self.groups = {"panel": np.arange(DT.N_PANEL), "noneg": np.arange(DT.N_LABELABLE, n_tr),
                       "held": np.arange(n_tr, n_tr + len(held))}
        sel = np.concatenate(list(self.groups.values()))
        self.idx = np.repeat(sel, M)
        self.noise = torch.cat([torch.randn(M, E.N_CHUNKS, E.ACT_DIM, generator=S.gen(seed, S.P_EVAL + int(self.st.sid[s])))
                                for s in sel])
        self.ref = ref
        self.train = train
        self.base = DT.rollout_chunked(ref, self.st, self.idx, self.noise)
        self._cache_base()
        h = hashlib.sha1()
        for x in (self.idx, self.base.negative, self.base.success, self.base.act, self.base.ran):
            h.update(np.ascontiguousarray(x).tobytes())
        self.base_hash = h.hexdigest()
        self.rep_idx = log.episodes["setup"]
        self.rep_noise = torch.from_numpy(log.episodes["noise"])

    def _cache_base(self):
        b = self.base
        self.base_ev = E.events(b.pos, b.contact)
        self.base_seg = seg_clean(b, self.st.K, self.base_ev)
        self.base_hom = E.homotopy_segments(self.st, self.idx, b.pos, b.widx, b.success)
        self.base_good = (~b.negative) & b.success
        self.base_speed = free_speed(b)

    @torch.no_grad()
    def compute(self, pol) -> Dict:
        is_ref = pol is self.ref
        ro = self.base if is_ref else DT.rollout_chunked(pol, self.st, self.idx, self.noise)
        ev = self.base_ev if is_ref else E.events(ro.pos, ro.contact)
        sc = self.base_seg if is_ref else seg_clean(ro, self.st.K, ev)
        hom = self.base_hom if is_ref else E.homotopy_segments(self.st, self.idx, ro.pos, ro.widx, ro.success)
        spd = self.base_speed if is_ref else free_speed(ro)
        b = self.base
        out: Dict = {"base_hash": self.base_hash}
        clean = ro.success & ~ro.negative
        for gname, gs in self.groups.items():
            m = np.isin(self.idx, gs)
            p = f"{gname}_"
            out[p + "n_ep"] = int(m.sum())
            out[p + "success_clean"] = float(clean[m].mean())
            out[p + "success"] = float(ro.success[m].mean())
            out[p + "bad"] = float(ro.negative[m].mean())
            out[p + "collision"] = float(ro.collided[m].mean())
            out[p + "jerk"] = float(ro.jerky[m].mean())
            out[p + "n_events"] = float(np.mean([len(ev[i]) for i in np.flatnonzero(m)]))
            out[p + "seg_clean"] = float(sc[m].mean())
            out[p + "seg_reach"] = float(np.mean([(ro.widx[i][~np.isnan(ro.pos[i, :, 0])].max() - 1 + ro.success[i]) / self.st.K
                                                  for i in np.flatnonzero(m)]))
            gg = m & self.base_good
            out[p + "good_broken"] = float((~clean)[gg].mean()) if gg.any() else float("nan")
            bb = m & b.negative
            out[p + "bad_fixed"] = float(clean[bb].mean()) if bb.any() else float("nan")
            tvs = []
            for s in gs:
                ms = self.idx == s
                for k in range(self.st.K):
                    x, y = hom[ms, k], self.base_hom[ms, k]
                    x, y = x[x >= 0], y[y >= 0]
                    if len(x) < MIN_TV or len(y) < MIN_TV:
                        continue
                    codes = np.union1d(x, y)
                    tvs.append(0.5 * np.abs(np.array([np.mean(x == c) - np.mean(y == c) for c in codes])).sum())
            out[p + "mode_tv"] = float(np.mean(tvs)) if tvs else float("nan")
            out[p + "speed_ratio"] = float(np.nanmean(spd[m]) / np.nanmean(self.base_speed[m]))
            sel = gg[:, None] & b.ran
            if sel.any() and not is_ref:
                o = torch.from_numpy(b.obs[sel]); z = torch.from_numpy(b.noise[sel])
                d = (pol.sample(z, o) - torch.from_numpy(b.act[sel])).pow(2).mean(-1).sqrt()
                out[p + "retained_disp"] = float(d.mean())
            else:
                out[p + "retained_disp"] = 0.0
        if len(self.rep_idx):
            rp = DT.rollout_chunked(pol, self.train, self.rep_idx, self.rep_noise)
            out["logged_neg_still_bad"] = float(rp.negative.mean())
        return out
