"""E5 confirmation runner: E1's K10c gated graft, unchanged, on 7 settings and fresh seeds 5-9.

Thin wrapper over explore/E1/{common,graft}.py (imported, not edited). It changes only:
  - load_cell: any of the 7 method-map settings (n_forget, layout) and any seed (E1's body minus
    its 'seeds 0-4' assert); cell name '<setting>_s<seed>_n<npm>' (E1's names for k4 / alt);
  - the unit-weight fit loop of graft.fit_graft is split into resumable chunks (checkpoint =
    graft params + Adam state + batch-generator state + iteration), so one unit stays < 20 min.
    Every line of the fit is E1's; `fit_graft_chunked` below is graft.fit_graft cut at the loop,
    and test_chunk.py checks chunked == unchunked bit for bit.
  - outputs go to explore/E5/{results,cache,grafts}.

Usage (one unit; rerun until it prints DONE):
  OMP_NUM_THREADS=1 python explore/E5/e5.py fit  --setting k4 --seed 5     # npm 4 fit (+ npm 4 eval)
  OMP_NUM_THREADS=1 python explore/E5/e5.py eval --setting k4 --seed 5 --npm 16
"""
from __future__ import annotations

import argparse, json, os, sys, time
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "explore", "E1"))
import common  # noqa: E402
import graft   # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from common import (E, Cell, CondFlow, frozen, egen, decode, decode_batched, build_tracker,  # noqa: E402
                    tracker_diag, oracle_eval, Policy, _POOL_CACHE)
from graft import GCfg, Graft, feats, combine_hooks, decode_grad, hop_targets, FILL_N  # noqa: E402

SETTINGS = {"k1": (1, "random"), "k2": (2, "random"), "k4": (4, "random"),
            "contig": (4, "contig"), "alt": (4, "alt"), "pairs": (4, "pairs"), "b3p1": (4, "b3p1")}
# K10c exactly as E1 ran it (explore/E1/results/graft_K10c_*.json "cfg")
K10C = GCfg(det_layer="b2.z", feat_layer="b2.z", out_layer="out", k_gate=10, det_steps=8, det_q=0.5,
            n_units=128, n_train=8192, iters=1500, lr=1e-3, w_good=4.0, hop_dirs=16, hop_max=2.5,
            hop_step=0.1, per_track=True, balance=True, clip=1.0, rounds=2)
TAG = "K10c"
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
GRAFTS = os.path.join(HERE, "grafts")
CHUNK_S = float(os.environ.get("E5_CHUNK_S", 12 * 60))     # stop a unit after the chunk that passes this
STEP = int(os.environ.get("E5_STEP", 250))                  # iterations per chunk


def load_cell(setting, seed, npm) -> Cell:
    """common.load_cell without the seed assert, for any of the 7 settings."""
    k, layout = SETTINGS[setting]
    ref = common.load_ref(seed)
    if seed not in _POOL_CACHE:
        _POOL_CACHE[seed] = E.make_pool(ref, seed, 200_000)
    ed = E.editset(_POOL_CACHE[seed], seed, k, npm, layout)
    bank = E.Bank(ref, seed, ed)
    return Cell(seed=seed, layout=layout, npm=npm, ref=ref, ed=ed, bank=bank, name=f"{setting}_s{seed}_n{npm}")


# ---------------------------------------------------------------- graft.fit_graft, cut at the loop
def fit_setup(cell: Cell, T, cfg: GCfg, log=print, prev=()):
    """graft.fit_graft up to (not including) the unit-fit loop. Verbatim apart from the cache path."""
    ref, net, seed = cell.ref, cell.ref.net, cell.seed
    t0 = time.time()
    g = egen(seed, 10)
    n = cfg.n_train
    c = torch.rand(n, generator=g); xi = torch.randn(n, 2, generator=g)
    layers = tuple(sorted({cfg.det_layer, cfg.feat_layer}))
    prev_hooks = combine_hooks(net, list(prev)) if prev else None
    a, tr = decode(net, xi, c, layers=layers, hooks=prev_hooks)      # current policy
    lab = T.label(a, c, loose=2.0)
    bad = np.isin(lab, T.bad)
    cf = net.c_feat(c)
    a_ref = lab_ref = None
    if prev:
        a_ref = decode_batched(net, xi, c)
        lab_ref = T.label(a_ref, c, loose=2.0)
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
    torch.manual_seed(seed + 1000 * len(prev))           # graft init (reproducible)
    gr = Graft(cfg, Fd.shape[1], Ff.shape[1], det_w, det_b, mu_d, sd_d, mu_f, sd_f, out_dim)
    tgt = a.clone() if not prev else a_ref.clone()
    ib = np.flatnonzero(bad if not prev else np.isin(lab_ref, T.bad))
    key = os.path.join(CACHE, f"tg_{cell.name}_n{n}_d{cfg.hop_dirs}_m{cfg.hop_max}_s{cfg.hop_step}_b{int(cfg.balance)}.pt")
    if os.path.exists(key):
        tgt, ok = torch.load(key, weights_only=False)
    else:
        fill = None
        if cfg.balance:
            assert not prev, "targets are computed (and cached) in round 1"
            good_tr = [k for k in range(T.n_tracks) if k not in T.bad and (lab == k).sum() > 0.02 * n]
            fill = {k: float((lab == k).sum()) / n for k in good_tr}
            FILL_N[0] = n
        ht, ok, htrk = hop_targets(ref, T, c[ib], xi[ib], cfg, egen(seed, 11), fill=fill)
        if cfg.balance:
            log("  balanced target fill " + str({k: round(fill[k] + float((htrk == k).sum()) / n, 3) for k in fill}))
        tgt[ib] = ht
        os.makedirs(CACHE, exist_ok=True)
        torch.save((tgt, ok), key + ".tmp"); os.replace(key + ".tmp", key)
    use = np.ones(n, bool); use[ib[~ok]] = False
    if prev:
        tgt = torch.where(torch.from_numpy(bad).unsqueeze(-1), tgt, a)
        use &= ~(bad & ~np.isin(lab_ref, T.bad))
    t_tg = time.time() - t0
    w = torch.where(torch.from_numpy(bad), 1.0, cfg.w_good)
    opt = torch.optim.Adam(gr.parameters(), lr=cfg.lr)
    hooks = combine_hooks(net, list(prev) + [gr])
    for p in net.parameters():
        p.requires_grad_(False)
    idx_all = np.flatnonzero(use)
    gi = torch.Generator().manual_seed(seed)
    return dict(gr=gr, opt=opt, hooks=hooks, gi=gi, idx_all=idx_all, xi=xi, c=c, tgt=tgt, w=w, bad=bad, ok=ok,
                tr=tr, cf=cf, n=n, t0=t0, t_tg=t_tg)


def fit_loop(S, cfg, net, it0, it1, log=print):
    """graft.fit_graft's loop, iterations it0..it1-1 (verbatim)."""
    gr, opt, hooks, gi, idx_all = S["gr"], S["opt"], S["hooks"], S["gi"], S["idx_all"]
    xi, c, tgt, w = S["xi"], S["c"], S["tgt"], S["w"]
    for it in range(it0, it1):
        j = torch.from_numpy(idx_all[torch.randint(0, len(idx_all), (1024,), generator=gi).numpy()])
        lr = cfg.lr * 0.5 * (1 + np.cos(np.pi * it / cfg.iters))
        for pg in opt.param_groups:
            pg["lr"] = lr
        out = decode_grad(net, xi[j], c[j], hooks)
        loss = (w[j] * ((out - tgt[j]) ** 2).sum(-1)).mean()
        opt.zero_grad(); loss.backward()
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(gr.parameters(), cfg.clip)
        opt.step()
        if it % 300 == 0 or it == cfg.iters - 1:
            log(f"  it {it} loss {loss.item():.4f}")


def fit_finish(S, cell, T, cfg, log=print, fit_s_prior=0.0):
    """graft.fit_graft after the loop (verbatim; fit_s adds the time of earlier chunks)."""
    net, seed = cell.ref.net, cell.seed
    gr, hooks, bad, ok, tr, cf, n = S["gr"], S["hooks"], S["bad"], S["ok"], S["tr"], S["cf"], S["n"]
    gr.eval()
    for p in gr.parameters():
        p.requires_grad_(False)
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
        qc["good_fill"] = {int(k): round(float((l1 == k).mean()), 4) for k in good_tr}
    with torch.no_grad():
        on0 = gr.gate({cfg.det_layer: tr[cfg.det_layer][0]}, tr["x"][0], torch.ones(n), cf).numpy()
    info_det = {"det_tpr0": float(on0[bad].mean()), "det_fpr0": float(on0[~bad].mean())}
    log("  detector " + str(info_det))
    info = {**info_det, "fit_s": round(fit_s_prior + time.time() - S["t0"], 1), "target_s": round(S["t_tg"], 1),
            "n_bad": int(bad.sum()), "hop_ok": float(ok.mean()), "quick": qc,
            "units": cfg.n_units, "params_added": int(sum(p.numel() for p in gr.parameters()))}
    return gr, hooks, info


# ---------------------------------------------------------------- resumable driver (graft.run, chunked)
def _load_grafts(sds, cfg):
    grs = []
    for sd in sds:
        g = Graft(cfg, sd["mu_d"].shape[0], sd["mu_f"].shape[0], sd["det_w"], sd["det_b"], sd["mu_d"], sd["sd_d"],
                  sd["mu_f"], sd["sd_f"], sd["w_out"].shape[0])
        g.load_state_dict(sd); g.eval()
        for p in g.parameters():
            p.requires_grad_(False)
        grs.append(g)
    return grs


def unit_fit(setting, seed, cfg=K10C, tag=TAG, npm=4):
    """One resumable unit of the npm-`npm` fit. Returns True when the cell's result JSON exists."""
    t_unit = time.time()
    name = f"{setting}_s{seed}_n{npm}"
    res = os.path.join(OUT, f"graft_{tag}_{name}.json")
    if os.path.exists(res):
        print("DONE (exists)", res); return True
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True); os.makedirs(GRAFTS, exist_ok=True)
    ck = os.path.join(CACHE, f"ckpt_{tag}_{name}.pt")
    st = torch.load(ck, weights_only=False) if os.path.exists(ck) else \
        {"done": [], "rounds": [], "infos": [], "r": 0, "it": 0, "gr": None, "opt": None, "gi": None, "fit_s": 0.0}
    cell = load_cell(setting, seed, npm)
    T = build_tracker(cell.ref, seed, cell.ed)
    net = cell.ref.net
    grafts = _load_grafts(st["done"], cfg)
    while st["r"] < cfg.rounds:
        r = st["r"]
        t_setup = time.time()
        S = fit_setup(cell, T, cfg, prev=grafts)
        if st["it"] > 0:
            S["gr"].load_state_dict(st["gr"]); S["opt"].load_state_dict(st["opt"]); S["gi"].set_state(st["gi"])
            # resumed: the time already spent is in st["fit_s"], minus this setup (counted again)
            S["t0"] = t_setup
        step = STEP
        while st["it"] < cfg.iters:
            it1 = min(st["it"] + step, cfg.iters)
            fit_loop(S, cfg, net, st["it"], it1)
            st["it"] = it1
            if st["it"] < cfg.iters and time.time() - t_unit > CHUNK_S:
                st["fit_s"] += time.time() - S["t0"]
                st.update(gr=S["gr"].state_dict(), opt=S["opt"].state_dict(), gi=S["gi"].get_state())
                torch.save(st, ck + ".tmp"); os.replace(ck + ".tmp", ck)
                print(f"[{tag}] {name} round {r + 1} checkpoint at it {st['it']} ({time.time() - t_unit:.0f}s)", flush=True)
                return False
        gr, hooks, info = fit_finish(S, cell, T, cfg, fit_s_prior=st["fit_s"])
        grafts.append(gr)
        if cfg.rounds > 1:
            ev_r = oracle_eval(cell, Policy(net, hooks))
            st["rounds"].append({"round": r + 1, **ev_r, "quick": info["quick"], "det_tpr0": info["det_tpr0"],
                                 "det_fpr0": info["det_fpr0"]})
            print(f"  round {r + 1}: D {ev_r['D']:.3f} Rem {ev_r['Removed']:.3f} Col {ev_r['Collateral']:.3f} off {ev_r['off']:.3f}", flush=True)
        st["infos"].append(info)
        st.update(done=[g_.state_dict() for g_ in grafts], r=r + 1, it=0, gr=None, opt=None, gi=None, fit_s=0.0)
        torch.save(st, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        if st["r"] < cfg.rounds and time.time() - t_unit > CHUNK_S:
            print(f"[{tag}] {name} round {r + 1} done, checkpoint ({time.time() - t_unit:.0f}s)", flush=True)
            return False
    info = dict(st["infos"][-1]); info["rounds"] = st["rounds"]
    info["params_added"] = int(sum(p.numel() for g_ in grafts for p in g_.parameters()))
    hooks = combine_hooks(net, grafts)
    ev = st["rounds"][-1] if st["rounds"] else oracle_eval(cell, Policy(net, hooks))
    ev = {k: ev[k] for k in ("D", "Removed", "Collateral", "off", "D_ref")}
    torch.save([g_.state_dict() for g_ in grafts], os.path.join(GRAFTS, f"grafts_{tag}_{name}.pt"))
    rec = {"cell": name, "setting": setting, "seed": seed, "layout": cell.layout, "npm": npm, "tag": tag,
           "cfg": asdict(cfg), "tracker": tracker_diag(T, cell), "bad_tracks": list(T.bad), **info, **ev}
    json.dump(rec, open(res + ".tmp", "w"), indent=1); os.replace(res + ".tmp", res)
    print(f"[{tag}] {name} D {ev['D']:.3f} Rem {ev['Removed']:.3f} Col {ev['Collateral']:.3f} off {ev['off']:.3f} DONE", flush=True)
    return True


def unit_eval(setting, seed, npm, cfg=K10C, tag=TAG, fit_npm=4):
    """eval_n16.py for one cell: reuse the npm-4 graft if both cells name the same bad tracks."""
    name = f"{setting}_s{seed}_n{npm}"
    dst = os.path.join(OUT, f"graft_{tag}_{name}.json")
    if os.path.exists(dst):
        print("DONE (exists)", dst); return True
    src_name = f"{setting}_s{seed}_n{fit_npm}"
    src = os.path.join(OUT, f"graft_{tag}_{src_name}.json")
    assert os.path.exists(src), f"needs the npm-{fit_npm} fit first: {src}"
    rec = json.load(open(src))
    c16 = load_cell(setting, seed, npm)
    T16 = build_tracker(c16.ref, seed, c16.ed)
    if list(T16.bad) != rec["bad_tracks"]:
        print(f"{name}: bad tracks differ ({rec['bad_tracks']} vs {T16.bad}); fitting its own graft (prereg rule)")
        return unit_fit(setting, seed, cfg, tag, npm=npm)
    grs = _load_grafts(torch.load(os.path.join(GRAFTS, f"grafts_{tag}_{src_name}.pt"), weights_only=False), cfg)
    ev = oracle_eval(c16, Policy(c16.ref.net, combine_hooks(c16.ref.net, grs)))
    out = {**rec, "cell": name, "npm": npm, "tracker": tracker_diag(T16, c16), "bad_tracks": list(T16.bad),
           "note": f"graft fitted on {src_name}; same bad tracks {rec['bad_tracks']}, so the npm-{npm} fit is identical", **ev}
    json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
    print(f"[{tag}] {name} D {ev['D']:.3f} Rem {ev['Removed']:.3f} Col {ev['Collateral']:.3f} (n{fit_npm}: D {rec['D']:.3f}) DONE", flush=True)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fit", "eval"])
    ap.add_argument("--setting", required=True, choices=list(SETTINGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--npm", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.what == "fit":
        unit_fit(a.setting, a.seed)
    else:
        unit_eval(a.setting, a.seed, a.npm)
