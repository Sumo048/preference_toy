"""Phase 4 activation tools (PLAN 6.5; W2b): ``contrast`` and ``unit_card``.

Design sources (docs/DEV_LOG.md, W2b):
* the pilot's sandbox code (smoke and r1/r2 sessions): surgeons sampled pi_ref at many c, labelled
  the target behaviour at every c themselves, fitted a logistic probe on activations (recall /
  false-fire rate), then compared a candidate graft's target share per c-bin before and after;
* the exploration sessions (E1-E3, seeds 0-4): the destination is linearly readable from step 0
  per failure cluster (held-out AUC 0.95-0.99), not with one global direction; a detector should
  be pooled over the steps up to k_commit; the cluster's behaviour must be labelled at all c
  (label-free behaviour tracks, ``toyrepair.tracks``), because the gate measures every c.

Both tools label a decode as "target" when its action lies in a behaviour track that the
cluster's members mark (``toyrepair.tracks``; no oracle, no labels). Samples: ``N_SAMPLES`` pairs
(c, xi) from the mining pool, check index ``SAMPLE_INDEX`` (fixed, so calls on different versions
see the same noises). Each call counts N_SAMPLES decodes. Output cap: 800 tokens (PLAN 6.5).
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch

N_SAMPLES = 4096
SAMPLE_INDEX = 11          # mining pool (1), check index 11: activation tools (documented, PLAN 5.3)
AUC_STEPS = (0, 2, 4, 6, 8, 10, 12, 14)
N_TRAIN = 3072             # first 3072 samples train a probe, the last 1024 test it
MIN_TARGET = 20
TOKEN_CAP = 800


class ToolInputError(ValueError):
    pass


def _check(layer: str, k: int):
    from .policy import CAPTURE_LAYERS, N_STEPS
    if layer not in CAPTURE_LAYERS:
        raise ToolInputError(f"unknown layer {layer!r}; use one of {', '.join(CAPTURE_LAYERS)}")
    if not 0 <= int(k) < N_STEPS:
        raise ToolInputError("k must be in 0..19")


def _samples(cell, version, layer: str):
    from . import policy, pools
    c, xi = pools.draw(cell, pools.MINING, N_SAMPLES, index=SAMPLE_INDEX)
    with torch.no_grad():
        dec = policy.decode(version, xi, c, save_states=True, capture=(layer,))
    return c, xi, dec


def _probe(Xtr, ytr, Xte, yte):
    """Standardised L2 logistic probe -> (held-out AUC, w_raw, b_raw, test scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=400)
    m.fit((Xtr - mu) / sd, ytr)
    w = m.coef_[0] / sd
    b = float(m.intercept_[0] - (m.coef_[0] * mu / sd).sum())
    s = Xte @ w + b
    auc = float(roc_auc_score(yte, s)) if 0 < yte.sum() < len(yte) else float("nan")
    return auc, w.astype(np.float32), b, s


def _deciles(y: np.ndarray, c: np.ndarray) -> str:
    bins = np.minimum((c * 10).astype(int), 9)
    return " ".join(f"{y[bins == i].mean():.2f}" if (bins == i).any() else "-" for i in range(10))


def target_of(cell, uids: Sequence[int], a, c) -> np.ndarray:
    from . import tracks
    return tracks.target_labels(tracks.get(cell), cell, uids, a, c)


def contrast(cell, version, uids: Sequence[int], cid: int, layer: str, k: int, scratch: str) -> Tuple[str, float]:
    """Activations of decodes that end in the cluster's behaviour (all c) against all other decodes.

    Returns (text, decodes). Writes the window probe (steps 0..k pooled, raw activation units)
    to ``<scratch>/contrast_c<cid>_<layer>_k<k>.npy``: it fires where acts . w + b > 0, so it can
    be a graft detector directly (``--det_source <layer> --det_w <file> --det_b <b> --det_k 0:<k+1>``)."""
    _check(layer, k)
    k = int(k)
    c, xi, dec = _samples(cell, version, layer)
    cn = c.numpy()
    y = target_of(cell, uids, dec.action, c)
    from . import tracks
    n_marked = len(tracks.get(cell).marked(cell, uids))
    A = dec.acts[layer].numpy()                       # (20, B, d)
    d = A.shape[2]
    L: List[str] = [f"contrast cluster {cid} layer {layer} ({d} units) version {version.id} k={k}: "
                    f"{N_SAMPLES} decodes over all c; target = action in the cluster's behaviour "
                    f"(label-free tracks of pi_ref decodes across c, marked by the members' logged actions; "
                    f"the members mark {n_marked} behaviour track(s))",
                    f"target share by c decile (0.0-0.1 ... 0.9-1.0): {_deciles(y, cn)} | all {y.mean():.3f}"]
    if y.sum() < MIN_TARGET:
        L.append(f"only {int(y.sum())} target decodes: no probe fitted")
        return "\n".join(L), float(dec.n_decodes)
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_SAMPLES)
    aucs = []
    for s in AUC_STEPS:
        auc, *_ = _probe(A[s, tr], y[tr], A[s, te], y[te])
        aucs.append(f"{s}:{auc:.3f}")
    L.append("probe held-out AUC by step: " + " ".join(aucs))
    # window probe pooled over steps 0..k (E1: a detector pooled up to k_commit works best)
    ks = np.arange(0, k + 1)
    Xtr = A[ks][:, tr].reshape(-1, d)
    ytr = np.tile(y[tr], len(ks))
    if len(Xtr) > 30000:
        sel = np.random.default_rng(0).choice(len(Xtr), 30000, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    auc, w, b, _ = _probe(Xtr, ytr, A[k, te], y[te])
    fires = np.stack([A[s, te] @ w + b > 0 for s in ks])     # (len(ks), n_test)
    yt = y[te]
    per_step_recall = fires[:, yt].mean(1)
    per_step_ff = fires[:, ~yt].mean(1)
    any_ff = fires[:, ~yt].any(0).mean()
    fname = f"contrast_c{cid}_{layer}_k{k}.npy"
    np.save(os.path.join(scratch, fname), w)
    L.append(f"window probe (steps 0..{k} pooled) -> {fname} (raw units; fires where acts.w + b > 0, b={b:.4f}): "
             f"AUC at k {auc:.3f}; recall mean {per_step_recall.mean():.3f} (min {per_step_recall.min():.3f}); "
             f"false-fire on other decodes mean {per_step_ff.mean():.3f}, any step {any_ff:.3f}")
    a1, a0 = A[k][y], A[k][~y]
    sdp = np.sqrt(0.5 * (a1.var(0) + a0.var(0))) + 1e-6
    eff = (a1.mean(0) - a0.mean(0)) / sdp
    top = np.argsort(-np.abs(eff))[:10]
    L.append("top units at k (idx d mean_target/mean_other): " +
             "; ".join(f"{i} {eff[i]:+.2f} {a1[:, i].mean():.2f}/{a0[:, i].mean():.2f}" for i in top))
    v = dec.v[k].numpy()
    dv = v[y].mean(0) - v[~y].mean(0)
    L.append(f"velocity at k: mean target ({v[y].mean(0)[0]:+.3f},{v[y].mean(0)[1]:+.3f}) "
             f"other ({v[~y].mean(0)[0]:+.3f},{v[~y].mean(0)[1]:+.3f}) diff ({dv[0]:+.3f},{dv[1]:+.3f})")
    return "\n".join(L), float(dec.n_decodes)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def unit_card(cell, version, uids: Sequence[int], cid: int, layer: str, idx: int) -> Tuple[str, float]:
    """Correlates of one unit with network inputs, step, and whether x_hat is in the cluster's behaviour."""
    from .policy import t_of_step
    _check(layer, 0)
    c, xi, dec = _samples(cell, version, layer)
    A = dec.acts[layer].numpy()
    if not 0 <= int(idx) < A.shape[2]:
        raise ToolInputError(f"idx must be in 0..{A.shape[2] - 1}")
    U = A[:, :, int(idx)]                                   # (20, B)
    S, V = dec.states.numpy(), dec.v.numpy()
    cn = c.numpy()
    y = target_of(cell, uids, dec.action, c)
    L = [f"unit_card {layer}[{idx}] version {version.id}, cluster {cid}: {N_SAMPLES} decodes over all c; "
         f"r = Pearson correlation; target = final action in the cluster's behaviour; "
         f"xh_tgt = predicted action x_hat = x_k - t_k v_k in it",
         "k t mean sd r(x0) r(x1) r(c) r(target) r(xh_tgt) mean_target mean_other"]
    for k in range(0, 20, 2):
        t = t_of_step(k)
        xh = S[k] - t * V[k]
        yh = target_of(cell, uids, torch.from_numpy(xh), c).astype(float)
        u = U[k]
        r_yh = _corr(u, yh)
        L.append(f"{k} {t:.2f} {u.mean():+.3f} {u.std():.3f} {_corr(u, S[k][:, 0]):+.2f} {_corr(u, S[k][:, 1]):+.2f} "
                 f"{_corr(u, cn):+.2f} {_corr(u, y.astype(float)):+.2f} {'-' if np.isnan(r_yh) else f'{r_yh:+.2f}'} "
                 f"{u[y].mean() if y.any() else float('nan'):+.3f} {u[~y].mean():+.3f}")
    tt = np.repeat(1.0 - np.arange(20) / 20.0, U.shape[1])
    L.append(f"pooled over steps: r(t) {_corr(U.reshape(-1), tt):+.2f}; "
             f"r(|x_k|) {_corr(U.reshape(-1), np.linalg.norm(S[:20], axis=2).reshape(-1)):+.2f}; "
             f"active share (|u| > 0.05) {(np.abs(U) > 0.05).mean():.2f}")
    return "\n".join(L), float(dec.n_decodes)
