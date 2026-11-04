"""E1: gated grafts. Hidden units beside a layer, hard-gated by a linear detector on network inputs
and activations, that redirect trajectories heading to bad basins during the early decode steps.

Fitting data are label-free: random (c, xi) decoded by pi_ref, flagged bad by the tracker (pi_ref
basins followed across c, seeded by the logged negatives). Rescue targets are the nearest
noise-space exit (a hop at the sample's own condition) that decodes into a non-bad basin.
The oracle is used only in oracle_eval.
"""
from __future__ import annotations

import argparse, json, os, time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from common import *


@dataclass
class GCfg:
    det_layer: str = "b2.z"      # detector reads this activation (+ x, c features)
    feat_layer: str = "b2.z"     # graft units read this activation (+ x, c features)
    out_layer: str = "out"       # "out": add to velocity; "b2": add to residual stream after block 2
    k_gate: int = 4              # graft active at steps 0..k_gate-1
    det_steps: int = 1           # detector trained on ref activations at steps 0..det_steps-1 (pooled)
    det_q: float = 0.5           # detector threshold on logistic probability
    n_units: int = 32
    n_train: int = 8192
    iters: int = 1500
    lr: float = 3e-3
    w_good: float = 4.0          # weight of the stay-put loss on falsely gated good samples
    hop_dirs: int = 16
    hop_max: float = 2.5
    hop_step: float = 0.1
    per_track: bool = False      # one gated path per bad track (cluster) instead of one for all
    balance: bool = False        # choose exits toward under-filled good tracks


def feats(acts, x, c_feat, layer):
    return torch.cat([acts[layer], x, c_feat], -1)


class Graft(torch.nn.Module):
    """Gated path: delta = 1[t-window] * 1[w_d . phi_d + b_d > 0] * W_out SiLU(W_in phi + b)."""

    def __init__(self, cfg: GCfg, d_det, d_feat, det_w, det_b, mu_d, sd_d, mu_f, sd_f, out_dim):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("det_w", det_w); self.register_buffer("det_b", det_b)
        self.register_buffer("mu_d", mu_d); self.register_buffer("sd_d", sd_d)
        self.register_buffer("mu_f", mu_f); self.register_buffer("sd_f", sd_f)
        n = cfg.n_units
        self.w_in = torch.nn.Parameter(torch.randn(n, d_feat) / np.sqrt(d_feat))
        self.b_in = torch.nn.Parameter(torch.zeros(n))
        self.w_out = torch.nn.Parameter(torch.zeros(out_dim, n))
        self.t_min = 1.0 - (cfg.k_gate - 0.5) * DT       # t-detector: fires for t > t_min

    def gate(self, acts, x, t, c_feat):
        pd = (feats(acts, x, c_feat, self.cfg.det_layer) - self.mu_d) / self.sd_d
        s = pd @ self.det_w + self.det_b                     # (N,) or (N, G) for per-track
        on = (s > 0)
        if on.dim() == 2:
            on = on.any(1)
        return on & (t > self.t_min)

    def forward(self, acts, x, t, c, c_feat):
        on = self.gate(acts, x, t, c_feat)
        pf = (feats(acts, x, c_feat, self.cfg.feat_layer) - self.mu_f) / self.sd_f
        d = F.silu(pf @ self.w_in.T + self.b_in) @ self.w_out.T
        return d * on.float().unsqueeze(-1)


def make_hooks(net, graft):
    cache = {}

    def hook(acts, x, t, c):
        key = id(c)
        if cache.get("k") != key:
            cache["k"], cache["cf"] = key, net.c_feat(c)
        return graft(acts, x, t, c, cache["cf"])
    return {graft.cfg.out_layer: hook}


def decode_grad(net, xi, c, hooks):
    """Euler decode with autograd (for fitting graft params)."""
    x = xi
    for k in range(K_STEPS):
        t = torch.full((x.shape[0],), 1.0 - k * DT)
        x = x - DT * net_forward(net, x, t, c, hooks=hooks)
    return x


@torch.no_grad()
def hop_targets(ref, T: Tracker, c, xi, cfg: GCfg, g, fill=None):
    """Nearest noise-space exit into a non-bad track at the sample's own condition.
    Returns targets (N,2), ok (N,) and the exit's track."""
    D = cfg.hop_dirs
    dirs = torch.randn(D, 2, generator=g); dirs = dirs / dirs.norm(dim=1, keepdim=True)
    taus = torch.arange(cfg.hop_step, cfg.hop_max + 1e-9, cfg.hop_step)
    N, Tn = xi.shape[0], len(taus)
    Z = xi[:, None, None, :] + taus[None, None, :, None] * dirs[None, :, None, :]      # (N, D, T, 2)
    cc = c[:, None, None].expand(N, D, Tn)
    a = decode_batched(ref.net, Z.reshape(-1, 2), cc.reshape(-1)).reshape(N, D, Tn, 2)
    lab = T.label(a.reshape(-1, 2), cc.reshape(-1)).reshape(N, D, Tn)
    good = (lab >= 0) & ~np.isin(lab, T.bad)
    tgt = torch.zeros(N, 2); ok = np.zeros(N, bool); trk = -np.ones(N, int)
    # candidate exits: along each direction, the first decode of each new good track (+2-step margin)
    cand = []                      # per sample: list of (tau_index, d, s, track)
    for i in range(N):
        ci = []
        for d in range(D):
            seen = set()
            for s in np.flatnonzero(good[i, d]):
                tk = lab[i, d, s]
                if tk in seen:
                    continue
                seen.add(tk)
                s2 = min(s + 2, Tn - 1)
                if not (good[i, d, s2] and lab[i, d, s2] == tk):
                    s2 = s
                ci.append((float(taus[s2]), d, s2, int(tk)))
                if len(seen) >= 3:
                    break
        cand.append(ci)
    price = {}
    if fill is not None and cfg.balance and len(fill):
        # Dual ascent on per-track prices: sample i takes the exit minimising tau + price[track],
        # and prices rise on tracks that would end above an equal share of the good mass.
        # fill[k] = pi_ref mass on good track k (fraction of all n_all samples).
        tracks = sorted(fill)
        tix = {k: j for j, k in enumerate(tracks)}
        C = max((len(ci) for ci in cand), default=0)
        TAU = np.full((N, max(C, 1)), np.inf); TRK = np.zeros((N, max(C, 1)), int)
        for i, ci in enumerate(cand):
            for j, e in enumerate(ci):
                if e[3] in tix:
                    TAU[i, j] = e[0]; TRK[i, j] = tix[e[3]]
        has = np.isfinite(TAU).any(1)
        n_all = FILL_N[0]
        base = np.array([fill[k] for k in tracks]) * n_all
        final = (base.sum() + has.sum()) / len(tracks)
        want = np.maximum(final - base, 0.0)
        p = np.zeros(len(tracks))
        for _ in range(400):
            ch = np.argmin(TAU + p[TRK], 1)
            load = np.bincount(TRK[np.arange(N), ch][has], minlength=len(tracks))
            p += 0.5 * (load - want) / max(has.sum(), 1)
        price = {k: float(p[tix[k]]) for k in tracks}
    for i, ci in enumerate(cand):
        if not ci:
            continue
        tau, d, s, tk = min(ci, key=lambda e: e[0] + price.get(e[3], 0.0))
        tgt[i] = a[i, d, s]; ok[i] = True; trk[i] = tk
    return tgt, ok, trk


FILL_N = [1]


def fit_graft(cell: Cell, T: Tracker, cfg: GCfg, log=print):
    ref, net, seed = cell.ref, cell.ref.net, cell.seed
    t0 = time.time()
    g = egen(seed, 10)
    n = cfg.n_train
    c = torch.rand(n, generator=g); xi = torch.randn(n, 2, generator=g)
    layers = tuple(sorted({cfg.det_layer, cfg.feat_layer}))
    a, tr = decode(net, xi, c, layers=layers)
    lab = T.label(a, c, loose=2.0)
    bad = np.isin(lab, T.bad)
    cf = net.c_feat(c)
    # ---- detector: logistic probe on pooled early-step activations (label = bad final action)
    Fd = torch.cat([feats({cfg.det_layer: tr[cfg.det_layer][k]}, tr["x"][k], cf, cfg.det_layer)
                    for k in range(cfg.det_steps)])
    yd = np.tile(bad, cfg.det_steps)
    mu_d, sd_d = Fd.mean(0), Fd.std(0) + 1e-6
    Xd = ((Fd - mu_d) / sd_d).numpy()
    if cfg.per_track:
        W, B = [], []
        for b_ in T.bad:
            y_ = np.tile(lab == b_, cfg.det_steps)
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xd, y_)
            W.append(clf.coef_[0]); B.append(clf.intercept_[0] - np.log(cfg.det_q / (1 - cfg.det_q)))
        det_w = torch.tensor(np.stack(W, 1), dtype=torch.float32); det_b = torch.tensor(B, dtype=torch.float32)
    else:
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xd, yd)
        det_w = torch.tensor(clf.coef_[0], dtype=torch.float32)
        det_b = torch.tensor(clf.intercept_[0] - np.log(cfg.det_q / (1 - cfg.det_q)), dtype=torch.float32)
    Ff = feats({cfg.feat_layer: tr[cfg.feat_layer][0]}, tr["x"][0], cf, cfg.feat_layer)
    mu_f, sd_f = Ff.mean(0), Ff.std(0) + 1e-6
    out_dim = 2 if cfg.out_layer == "out" else 128
    gr = Graft(cfg, Fd.shape[1], Ff.shape[1], det_w, det_b, mu_d, sd_d, mu_f, sd_f, out_dim)
    # ---- rescue targets for bad samples
    tgt = a.clone()
    ib = np.flatnonzero(bad)
    fill = None
    if cfg.balance:
        good_tr = [k for k in range(T.n_tracks) if k not in T.bad and (lab == k).sum() > 0.02 * n]
        fill = {k: float((lab == k).sum()) / n for k in good_tr}
        FILL_N[0] = n
    ht, ok, htrk = hop_targets(ref, T, c[ib], xi[ib], cfg, egen(seed, 11), fill=fill)
    if cfg.balance:
        log("  balanced target fill " + str({k: round(fill[k] + float((htrk == k).sum()) / n, 3) for k in fill}))
    tgt[ib] = ht
    use = np.ones(n, bool); use[ib[~ok]] = False
    t_tg = time.time() - t0
    # ---- which samples does the gate ever touch on the ref trajectory (approx.; recomputed live)
    w = torch.where(torch.from_numpy(bad), 1.0, cfg.w_good)
    opt = torch.optim.Adam(gr.parameters(), lr=cfg.lr)
    hooks = make_hooks(net, gr)
    for p in net.parameters():
        p.requires_grad_(False)
    idx_all = np.flatnonzero(use)
    gi = torch.Generator().manual_seed(seed)
    for it in range(cfg.iters):
        j = torch.from_numpy(idx_all[torch.randint(0, len(idx_all), (1024,), generator=gi).numpy()])
        lr = cfg.lr * 0.5 * (1 + np.cos(np.pi * it / cfg.iters))
        for pg in opt.param_groups:
            pg["lr"] = lr
        out = decode_grad(net, xi[j], c[j], hooks)
        loss = (w[j] * ((out - tgt[j]) ** 2).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 300 == 0 or it == cfg.iters - 1:
            log(f"  it {it} loss {loss.item():.4f}")
    gr.eval()
    for p in gr.parameters():
        p.requires_grad_(False)
    # ---- label-free quick check on held-out samples
    with torch.no_grad():
        g2 = egen(seed, 12); m = 8192
        c2 = torch.rand(m, generator=g2); x2 = torch.randn(m, 2, generator=g2)
        a0 = decode_batched(net, x2, c2)
        a1 = decode_batched(net, x2, c2, hooks=hooks)
        l0, l1 = T.label(a0, c2, 2.0), T.label(a1, c2, 2.0)
        b0 = np.isin(l0, T.bad)
        qc = {"bad_frac_ref": float(b0.mean()), "bad_frac_new": float(np.isin(l1, T.bad).mean()),
              "good_moved": float((torch.linalg.norm(a1 - a0, dim=1).numpy()[~b0] > 0.25).mean()),
              "off_new": float((l1 == -1).mean()), "off_ref": float((l0 == -1).mean())}
        good_tr = [k for k in range(T.n_tracks) if k not in T.bad and (l0 == k).sum() > 0.02 * m]
        fill = {int(k): round(float((l1 == k).mean()), 4) for k in good_tr}
        qc["good_fill"] = fill
    info = {"fit_s": round(time.time() - t0, 1), "target_s": round(t_tg, 1), "n_bad": int(bad.sum()),
            "hop_ok": float(ok.mean()), "quick": qc,
            "units": cfg.n_units, "params_added": int(sum(p.numel() for p in gr.parameters()))}
    return gr, hooks, info


def run(seed, layout, npm, cfg: GCfg, tag, outdir="results"):
    cell = load_cell(seed, layout, npm)
    T = build_tracker(cell.ref, seed, cell.ed)
    gr, hooks, info = fit_graft(cell, T, cfg)
    ev = oracle_eval(cell, Policy(cell.ref.net, hooks))
    rec = {"cell": cell.name, "seed": seed, "layout": layout, "npm": npm, "tag": tag, "cfg": asdict(cfg),
           "tracker": tracker_diag(T, cell), **info, **ev}
    os.makedirs(outdir, exist_ok=True)
    json.dump(rec, open(os.path.join(outdir, f"graft_{tag}_{cell.name}.json"), "w"), indent=1)
    print(f"[{tag}] {cell.name} D {ev['D']:.3f} Rem {ev['Removed']:.3f} Col {ev['Collateral']:.3f} off {ev['off']:.3f} "
          f"quick {info['quick']} ({info['fit_s']}s)", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--layout", default="random")
    ap.add_argument("--npm", type=int, default=4); ap.add_argument("--tag", default="v1")
    for f, v in asdict(GCfg()).items():
        ap.add_argument(f"--{f}", type=type(v) if not isinstance(v, bool) else (lambda s: s in ("1", "true", "True")), default=v)
    a = ap.parse_args()
    cfg = GCfg(**{f: getattr(a, f) for f in asdict(GCfg())})
    run(a.seed, a.layout, a.npm, cfg, a.tag)
