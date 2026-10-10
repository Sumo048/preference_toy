"""CRN evaluation: every policy of a seed is rolled out on the same setups with the same noise
sequences, so per-noise outcome flips are paired.

Groups: ``neg`` training setups that logged negatives, ``noneg`` the pre-registered training
setups without negatives, ``held`` held-out setups (new to the base), and -- only when non-empty --
``neg_missing`` training setups that wanted negatives but logged none. The env labeller is used
here (metrics only).

``base_hash`` fingerprints the frozen base's CRN rollout; every paired metric (bad_fixed,
good_broken, hom_tv, retained_disp) is relative to it, so all arms of a seed must share it.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import torch

from . import data as DT
from . import env as E
from . import seeds as S

MIN_SUCC_TV = 5


def concat(a: E.Setups, b: E.Setups) -> E.Setups:
    return E.Setups(np.concatenate([a.sid, b.sid]), np.concatenate([a.start, b.start]), np.concatenate([a.goal, b.goal]),
                    np.concatenate([a.centre, b.centre]), np.concatenate([a.radius, b.radius]),
                    np.concatenate([a.n_obs, b.n_obs]))


class EvalSet:
    def __init__(self, ref, seed, train: E.Setups, held: E.Setups, log: DT.LogResult, M=DT.M_EVAL, groups=None):
        self.st = concat(train, held)
        if groups is None:
            gr = DT.setup_groups(seed, log, train)
            groups = {"neg": gr["neg"], "noneg": gr["noneg"], "held": np.arange(len(train), len(train) + len(held))}
            if len(gr["neg_missing"]):
                groups["neg_missing"] = gr["neg_missing"]
        self.groups = groups
        sel = np.unique(np.concatenate(list(self.groups.values())))
        self.idx = np.repeat(sel, M)
        self.noise = torch.cat([torch.randn(M, E.N_CHUNKS, E.ACT_DIM, generator=S.gen(seed, S.P_EVAL + int(self.st.sid[s])))
                                for s in sel])
        self.log = log
        self.train = train
        self.base = DT.rollout(ref, self.st, self.idx, self.noise)
        self.base_hom = E.homotopy(self.st, self.idx, self.base.pos)
        self.base_good = (~self.base.negative) & self.base.success
        h = hashlib.sha1()
        for x in (self.idx, self.base.negative, self.base.success, self.base.act, self.base.ran):
            h.update(np.ascontiguousarray(x).tobytes())
        self.base_hash = h.hexdigest()
        self.ref = ref
        # replay of the logged negative episodes with their logged noise sequences
        self.rep_idx = log.episodes["setup"]
        self.rep_noise = torch.from_numpy(log.episodes["noise"])

    @torch.no_grad()
    def compute(self, pol) -> Dict:
        ro = self.base if pol is self.ref else DT.rollout(pol, self.st, self.idx, self.noise)
        hom = self.base_hom if pol is self.ref else E.homotopy(self.st, self.idx, ro.pos)
        b = self.base
        out: Dict = {"base_hash": self.base_hash}
        plen = E.path_length(ro.pos)
        # jerk inside a single executed chunk (zig-zag) vs only at chunk boundaries (replanning)
        # (scored on EXECUTED positions: steps cut by the goal / a collision never count)
        intra = (E.chunk_jerk_executed(ro.pos) > E.JERK_THR).any(1)
        for gname, gs in self.groups.items():
            m = np.isin(self.idx, gs)
            if not m.any():
                continue
            p = f"{gname}_"
            out[p + "n_ep"] = int(m.sum())
            out[p + "collision"] = float(ro.collided[m].mean())
            out[p + "jerk"] = float(ro.jerky[m].mean())
            out[p + "jerk_intra"] = float((ro.jerky & intra)[m].mean())
            out[p + "jerk_boundary_only"] = float((ro.jerky & ~intra)[m].mean())
            out[p + "bad"] = float(ro.negative[m].mean())
            out[p + "bad_col_or_intra"] = float((ro.collided | (ro.jerky & intra))[m].mean())
            out[p + "success"] = float(ro.success[m].mean())
            out[p + "success_clean"] = float((ro.success & ~ro.negative)[m].mean())
            out[p + "timeout"] = float((~ro.success & ~ro.collided)[m].mean())
            sm = m & ro.success
            out[p + "path_len"] = float(plen[sm].mean()) if sm.any() else float("nan")
            out[p + "steps_to_goal"] = float(ro.steps[sm].mean()) if sm.any() else float("nan")
            out[p + "clip_frac"] = float(ro.clipped[m].sum() / max(ro.steps[m].sum(), 1))
            bb = m & b.negative
            out[p + "bad_fixed"] = float((~ro.negative & ro.success)[bb].mean()) if bb.any() else float("nan")
            gg = m & self.base_good
            out[p + "good_broken"] = float((ro.negative | ~ro.success)[gg].mean()) if gg.any() else float("nan")
            both = m & ro.success & b.success
            out[p + "hom_flip"] = float((hom != self.base_hom)[both].mean()) if both.any() else float("nan")
            tvs = []
            for s in gs:
                ms = self.idx == s
                x, y = hom[ms & ro.success], self.base_hom[ms & b.success]
                if len(x) < MIN_SUCC_TV or len(y) < MIN_SUCC_TV:
                    continue
                codes = np.union1d(x, y)
                px = np.array([np.mean(x == c) for c in codes])
                py = np.array([np.mean(y == c) for c in codes])
                tvs.append(0.5 * np.abs(px - py).sum())
            out[p + "hom_tv"] = float(np.mean(tvs)) if tvs else float("nan")
            out[p + "hom_tv_n"] = len(tvs)
            # retained behaviour change: same observation, same noise, base-good episodes
            sel = gg[:, None] & b.ran
            if sel.any():
                o = torch.from_numpy(b.obs[sel])
                z = torch.from_numpy(b.noise[sel])
                ae = pol.sample(z, o) if pol is not self.ref else torch.from_numpy(b.act[sel])
                d = (ae - torch.from_numpy(b.act[sel])).pow(2).mean(-1).sqrt()
                out[p + "retained_disp"] = float(d.mean())
        if len(self.rep_idx):
            rp = DT.rollout(pol, self.train, self.rep_idx, self.rep_noise)
            out["logged_neg_still_bad"] = float(rp.negative.mean())
            out["logged_neg_success_clean"] = float((rp.success & ~rp.negative).mean())
        return out
