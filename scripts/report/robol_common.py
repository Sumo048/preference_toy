"""Shared helpers for the robol report figures (seed 0)."""
import os, sys
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, torch
from toypref.robol import data as DT, env as E, seeds as S
from toypref.robol.policy import load

ROBO = "/grogu/datasets/rollout/soumojib/toypref/runs/robol"
OUT = "/grogu/datasets/rollout/soumojib/toypref/report_robol"
s = 0
M = 32


def setup():
    bd = os.path.join(ROBO, "base", f"s{s}")
    ref = load(os.path.join(bd, "policy.pt"))
    lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
    train, held = DT.experiment_setups(s)
    st = E.concat(train, held)
    n_tr = len(train)
    sel = np.arange(n_tr, n_tr + len(held))
    idx = np.repeat(sel, M)
    noise = torch.cat([torch.randn(M, E.N_CHUNKS, E.ACT_DIM, generator=S.gen(s, S.P_EVAL + int(st.sid[c]))) for c in sel])
    return ref, lg, st, sel, idx, noise


def outcome(ro):
    """0 clean, 1 negative, 2 timeout."""
    clean = ro.success & ~ro.negative
    return np.where(ro.negative, 1, np.where(clean, 0, 2))


def pack_course(st, c):
    pres = st.radius[c] > 0
    return dict(wps=st.wps[c], centre=st.centre[c][pres], radius=st.radius[c][pres])


def pick_courses(base_out, sel, n=3):
    """held-out courses whose 16 base rollouts (first 16 CRN noises) have 3-6 negatives; the 3 closest to 4."""
    cands = []
    for j, c in enumerate(sel):
        o = base_out[j * M:j * M + 16]
        nneg, nto = int((o == 1).sum()), int((o == 2).sum())
        if 3 <= nneg <= 6:
            cands.append((abs(nneg - 4) + 0.1 * (nto == 0), j, c))
    cands.sort()
    return [(j, c) for _, j, c in cands[:n]]
