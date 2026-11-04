"""GPROP: can a GATED edit reach B6's D when trained toward PROPORTIONAL good-behaviour targets?

Arms (one cell = setting x seed x npm; all fitting data label-free: logged negatives, unlabelled
pi_ref samples, label-free tracks; the oracle (experiment.compute / MIX.assign) is used only in
`evaluate`):

  E1   K10c gated graft (explore/E1/graft.py), nearest balanced noise-space exits as targets.
       Re-evaluated from stored grafts (E1 / E5) or fitted here with targets="hop" (E1's code path).
  GP1  K10c gated graft, identical except the rescue targets: each flagged decode at condition c gets
       a target drawn from pi_ref samples at c on GOOD tracks, in proportion to each good track's mass
       at c, matched to the flagged noises by an optimal-transport (Hungarian) assignment in noise
       space within narrow c-bins. Stay-put loss on unflagged decodes (E1's w_good term) as in K10c.
  GP2  rank-r LoRA on inp, every block's fc1/fc2 and out (E3's lora_wrap set), whose delta is
       multiplied per sample by a soft gate g = sigmoid((p - tau)/T) * 1[step < k_commit], where
       p = max over bad tracks of the K10c per-track detector probability computed from FROZEN pi_ref
       activations (blocks.2 SiLU hidden + x + c features). Trained with the B6 loss: flow matching on
       pi_ref samples whose E3 track is assigned and not bad (60k pool, purpose 90,001), fresh noise.
  GP3  the same LoRA with g = 1 everywhere (control).

Units are resumable (checkpoint state in explore/GPROP/cache); each unit stops after the chunk that
passes CHUNK_S seconds. Results: explore/GPROP/results/<arm>/<cell>.json; fitted edit weights:
explore/GPROP/edits/<arm>/<cell>.pt.

Usage:
  OMP_NUM_THREADS=1 python explore/GPROP/gprop.py <arm> --setting k4 --seed 0 [--npm 4] [--cfg NAME]
  arm in {gp1, gp2, gp3, e1fit, e1eval}
"""
from __future__ import annotations

import argparse, copy, json, math, os, sys, time
from dataclasses import dataclass, asdict, field, replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "explore", "E1"))
sys.path.insert(0, os.path.join(ROOT, "explore", "E3"))
sys.path.insert(0, ROOT)
import common  # noqa: E402
import graft   # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from common import (E, TG, Cell, egen, decode, decode_batched, build_tracker, tracker_diag, net_forward,  # noqa: E402
                    Policy, _POOL_CACHE, K_STEPS, DT)
from graft import GCfg, Graft, feats, combine_hooks, decode_grad, hop_targets, FILL_N  # noqa: E402
from toypref.flow import FlowPolicy  # noqa: E402
from toypref.train import cosine  # noqa: E402

SETTINGS = {"k1": (1, "random"), "k2": (2, "random"), "k4": (4, "random"),
            "contig": (4, "contig"), "alt": (4, "alt"), "pairs": (4, "pairs"), "b3p1": (4, "b3p1")}
K10C = GCfg(det_layer="b2.z", feat_layer="b2.z", out_layer="out", k_gate=10, det_steps=8, det_q=0.5,
            n_units=128, n_train=8192, iters=1500, lr=1e-3, w_good=4.0, hop_dirs=16, hop_max=2.5,
            hop_step=0.1, per_track=True, balance=True, clip=1.0, rounds=2)
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
EDITS = os.path.join(HERE, "edits")
CHUNK_S = float(os.environ.get("GP_CHUNK_S", 12 * 60))
P_GP = 920_000          # GPROP's own streams: egen-style gen(seed, P_GP + j)


def ggen(seed, j):
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + P_GP + j)


def load_cell(setting, seed, npm) -> Cell:
    """E1's common.load_cell for any of the 7 settings and any seed (as explore/E5/e5.py)."""
    k, layout = SETTINGS[setting]
    ref = common.load_ref(seed)
    if seed not in _POOL_CACHE:
        _POOL_CACHE[seed] = E.make_pool(ref, seed, 200_000)
    ed = E.editset(_POOL_CACHE[seed], seed, k, npm, layout)
    bank = E.Bank(ref, seed, ed)
    return Cell(seed=seed, layout=layout, npm=npm, ref=ref, ed=ed, bank=bank, name=f"{setting}_s{seed}_n{npm}")


def _save_json(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rec, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)


def _save_pt(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# =============================================================== GP1: proportional OT targets
@dataclass
class PCfg:
    """GP1 target knobs (the graft itself is K10c)."""
    bins: int = 64          # equal-width c bins for the OT assignment
    cand_mult: int = 3      # noises drawn per flagged sample per attempt
    cost: str = "noise"     # "noise": |xi_i - z_j|^2 ; "action": |a_ref(xi_i) - a_ref(z_j)|^2
    redecode: bool = True   # target = pi_ref(z_j, c_i) at the flagged sample's own c


@torch.no_grad()
def prop_targets(ref, T, c, xi, a_cur, pc: PCfg, seed, log=print):
    """For flagged samples (c_i, xi_i): targets from pi_ref samples at c_i on good tracks, in proportion
    to each good track's pi_ref mass at c_i, matched by a Hungarian assignment within each c-bin.
    Returns targets (N,2), ok (N,), target track (N,)."""
    g = ggen(seed, 1)
    N = xi.shape[0]
    tgt = torch.zeros(N, 2); ok = np.zeros(N, bool); trk = -np.ones(N, int)
    edges = np.linspace(0.0, 1.0, pc.bins + 1)
    cn = c.numpy()
    bi = np.clip(np.searchsorted(edges, cn, side="right") - 1, 0, pc.bins - 1)
    for b in range(pc.bins):
        F_ = np.flatnonzero(bi == b)
        if not len(F_):
            continue
        nf = len(F_)
        Zs, As, Cs = [], [], []
        have, tries = 0, 0
        while have < nf and tries < 20:
            m = pc.cand_mult * nf
            z = torch.randn(m, 2, generator=g)
            cc = torch.from_numpy(edges[b] + (edges[b + 1] - edges[b]) * torch.rand(m, generator=g).numpy()).float()
            a = decode_batched(ref.net, z, cc)
            lab = T.label(a, cc, loose=1.0)
            gd = (lab >= 0) & ~np.isin(lab, T.bad)
            Zs.append(z[gd]); As.append(a[gd]); Cs.append(cc[gd])
            have += int(gd.sum()); tries += 1
        if have == 0:
            continue
        Z = torch.cat(Zs)[:nf]; A = torch.cat(As)[:nf]; Cc = torch.cat(Cs)[:nf]
        if pc.cost == "noise":
            C = torch.cdist(xi[F_], Z) ** 2
        else:
            C = torch.cdist(a_cur[F_], A) ** 2
        r, col = linear_sum_assignment(C.numpy())        # rectangular if have < nf: some rows unmatched
        rows = F_[r]
        if pc.redecode:
            t_ = decode_batched(ref.net, Z[col], c[rows])
            lt = T.label(t_, c[rows], loose=1.0)
            good = (lt >= 0) & ~np.isin(lt, T.bad)
            la = T.label(A[col], Cc[col], loose=2.0)
            t_[~good] = A[col][~good]
            lt[~good] = la[~good]
        else:
            t_ = A[col]; lt = T.label(t_, c[rows], loose=2.0)
        tgt[rows] = t_; ok[rows] = True; trk[rows] = lt
    return tgt, ok, trk


def fit_setup(cell: Cell, T, cfg: GCfg, pc: PCfg | None, targets: str, cache_tag: str, prev=(), log=print):
    """explore/E5/e5.py fit_setup (= graft.fit_graft up to the loop), with the rescue-target step
    switchable: targets="hop" is E1's (balanced nearest exits), "prop" is GP1's proportional OT."""
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
    det_w, det_b, mu_d, sd_d, Fd = _detectors(cfg, tr, cf, lab, bad, T)
    Ff = feats({cfg.feat_layer: tr[cfg.feat_layer][0]}, tr["x"][0], cf, cfg.feat_layer)
    mu_f, sd_f = Ff.mean(0), Ff.std(0) + 1e-6
    out_dim = 2 if cfg.out_layer == "out" else 128
    torch.manual_seed(seed + 1000 * len(prev))           # graft init (reproducible)
    gr = Graft(cfg, Fd.shape[1], Ff.shape[1], det_w, det_b, mu_d, sd_d, mu_f, sd_f, out_dim)
    tgt = a.clone() if not prev else a_ref.clone()
    ib = np.flatnonzero(bad if not prev else np.isin(lab_ref, T.bad))
    key = os.path.join(CACHE, f"tg_{cache_tag}_{cell.name}.pt")
    fill_info = None
    if os.path.exists(key):
        tgt, ok, fill_info = torch.load(key, weights_only=False)
    else:
        assert not prev, "targets are computed (and cached) in round 1"
        good_tr = [k for k in range(T.n_tracks) if k not in T.bad and (lab == k).sum() > 0.02 * n]
        fill = {k: float((lab == k).sum()) / n for k in good_tr}
        if targets == "hop":
            FILL_N[0] = n
            ht, ok, htrk = hop_targets(ref, T, c[ib], xi[ib], cfg, egen(seed, 11), fill=fill if cfg.balance else None)
        else:
            ht, ok, htrk = prop_targets(ref, T, c[ib], xi[ib], a[ib], pc, seed, log=log)
        fill_info = {"ref_fill": {int(k): round(v, 4) for k, v in fill.items()},
                     "target_fill": {int(k): round(fill[k] + float((htrk[ok] == k).sum()) / n, 4) for k in fill}}
        log(f"  {targets} target fill " + str(fill_info["target_fill"]))
        tgt[ib] = ht
        _save_pt(key, (tgt, ok, fill_info))
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
                tr=tr, cf=cf, n=n, t0=t0, t_tg=t_tg, fill_info=fill_info)


def _detectors(cfg, tr, cf, lab, bad, T):
    """graft.fit_graft's detector block (verbatim)."""
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
    return det_w, det_b, mu_d, sd_d, Fd


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


def load_grafts(sds, cfg):
    grs = []
    for sd in sds:
        g = Graft(cfg, sd["mu_d"].shape[0], sd["mu_f"].shape[0], sd["det_w"], sd["det_b"], sd["mu_d"], sd["sd_d"],
                  sd["mu_f"], sd["sd_f"], sd["w_out"].shape[0])
        g.load_state_dict(sd); g.eval()
        for p in g.parameters():
            p.requires_grad_(False)
        grs.append(g)
    return grs


GP1_CFGS = {
    # name: (GCfg, PCfg, targets)
    "e1": (K10C, None, "hop"),                                     # E1 K10c itself (for cells E1/E5 lack)
    "A": (K10C, PCfg(), "prop"),                                   # proportional OT, noise-space cost
    "B": (K10C, PCfg(cost="action"), "prop"),                      # proportional OT, action-space cost
    "C": (K10C, PCfg(bins=8), "prop"),                             # proportional OT, wide c-bins (smoother map)
    "A1": (replace(K10C, rounds=1), PCfg(), "prop"),
    "Ak12": (replace(K10C, k_gate=12, det_steps=10), PCfg(), "prop"),
    "Aw1": (replace(K10C, w_good=1.0), PCfg(), "prop"),
}


def unit_graft(arm, cfgname, setting, seed, npm=4):
    """One resumable unit of a K10c-style graft fit (E1 or GP1). True when the result exists."""
    cfg, pc, targets = GP1_CFGS[cfgname]
    tag = f"{arm}-{cfgname}"
    t_unit = time.time()
    name = f"{setting}_s{seed}_n{npm}"
    res = os.path.join(RES, tag, f"{name}.json")
    if os.path.exists(res):
        print("DONE (exists)", res); return True
    ck = os.path.join(CACHE, f"ckpt_{tag}_{name}.pt")
    st = torch.load(ck, weights_only=False) if os.path.exists(ck) else \
        {"done": [], "rounds": [], "infos": [], "r": 0, "it": 0, "gr": None, "opt": None, "gi": None, "fit_s": 0.0}
    cell = load_cell(setting, seed, npm)
    T = build_tracker(cell.ref, seed, cell.ed)
    net = cell.ref.net
    grafts = load_grafts(st["done"], cfg)
    cache_tag = f"{targets}" + ("" if pc is None else f"_b{pc.bins}_m{pc.cand_mult}_{pc.cost}_r{int(pc.redecode)}")
    while st["r"] < cfg.rounds:
        r = st["r"]
        t_setup = time.time()
        S = fit_setup(cell, T, cfg, pc, targets, cache_tag, prev=grafts)
        if st["it"] > 0:
            S["gr"].load_state_dict(st["gr"]); S["opt"].load_state_dict(st["opt"]); S["gi"].set_state(st["gi"])
            S["t0"] = t_setup
        while st["it"] < cfg.iters:
            it1 = min(st["it"] + 250, cfg.iters)
            fit_loop(S, cfg, net, st["it"], it1)
            st["it"] = it1
            if st["it"] < cfg.iters and time.time() - t_unit > CHUNK_S:
                st["fit_s"] += time.time() - S["t0"]
                st.update(gr=S["gr"].state_dict(), opt=S["opt"].state_dict(), gi=S["gi"].get_state())
                _save_pt(ck, st)
                print(f"[{tag}] {name} round {r + 1} checkpoint at it {st['it']} ({time.time() - t_unit:.0f}s)", flush=True)
                return False
        gr = S["gr"]; gr.eval()
        for p in gr.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            on0 = gr.gate({cfg.det_layer: S["tr"][cfg.det_layer][0]}, S["tr"]["x"][0], torch.ones(S["n"]), S["cf"]).numpy()
        info = {"det_tpr0": float(on0[S["bad"]].mean()), "det_fpr0": float(on0[~S["bad"]].mean()),
                "fit_s": round(st["fit_s"] + time.time() - S["t0"], 1), "target_s": round(S["t_tg"], 1),
                "n_bad": int(S["bad"].sum()), "target_ok": float(S["ok"].mean()), "fill": S["fill_info"]}
        grafts.append(gr)
        if cfg.rounds > 1:
            ev_r = evaluate(cell, GraftPolicy(net, grafts, cfg))
            st["rounds"].append({"round": r + 1, **{k: ev_r[k] for k in ("D", "Removed", "Collateral", "off", "flips", "gate_retained")}, **info})
            print(f"  round {r + 1}: D {ev_r['D']:.4f} Rem {ev_r['Removed']:.3f} Col {ev_r['Collateral']:.3f} off {ev_r['off']:.3f}", flush=True)
        st["infos"].append(info)
        st.update(done=[g_.state_dict() for g_ in grafts], r=r + 1, it=0, gr=None, opt=None, gi=None, fit_s=0.0)
        _save_pt(ck, st)
        if st["r"] < cfg.rounds and time.time() - t_unit > CHUNK_S:
            print(f"[{tag}] {name} round {r + 1} done, checkpoint ({time.time() - t_unit:.0f}s)", flush=True)
            return False
    _save_pt(os.path.join(EDITS, tag, f"{name}.pt"), {"kind": "graft", "cfg": asdict(cfg), "grafts": [g_.state_dict() for g_ in grafts]})
    ev = evaluate(cell, GraftPolicy(net, grafts, cfg))
    rec = {"arm": arm, "cfgname": cfgname, "cell": name, "setting": setting, "seed": seed, "npm": npm,
           "cfg": asdict(cfg), "pcfg": asdict(pc) if pc else None, "targets": targets,
           "tracker": tracker_diag(T, cell), "bad_tracks": list(T.bad),
           "params_added": int(sum(p.numel() for g_ in grafts for p in g_.parameters())),
           "buffers_added": int(sum(b.numel() for g_ in grafts for b in g_.buffers())),
           "rounds": st["rounds"], "infos": st["infos"], **ev}
    _save_json(res, rec)
    print(f"[{tag}] {name} D {ev['D']:.4f} Rem {ev['Removed']:.3f} Col {ev['Collateral']:.3f} off {ev['off']:.3f} "
          f"flips {ev['flips']:.3f} gate {ev['gate_retained']:.3f} DONE", flush=True)
    return True


class GraftPolicy:
    """Policy with hard-gated grafts; can report whether any graft gate fired during the decode."""

    def __init__(self, net, grafts, cfg):
        self.net, self.grafts, self.cfg = net, list(grafts), cfg
        self.hooks = combine_hooks(net, self.grafts)

    @torch.no_grad()
    def sample(self, xi, c, batch=32768):
        return decode_batched(self.net, xi, c, hooks=self.hooks, bs=batch)

    @torch.no_grad()
    def sample_gate(self, xi, c, batch=32768):
        outs, fired = [], []
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float(); cc = c[i:i + batch].float()
            cf = self.net.c_feat(cc)
            f = torch.zeros(x.shape[0], dtype=torch.bool)
            for k in range(K_STEPS):
                t = torch.full((x.shape[0],), 1.0 - k * DT)
                v, acts = net_forward(self.net, x, t, cc, capture=True, hooks=self.hooks)
                for gr in self.grafts:
                    f |= gr.gate(acts, x, t, cf)
                x = x - DT * v
            outs.append(x); fired.append(f)
        return torch.cat(outs), torch.cat(fired).numpy()


# =============================================================== GP2 / GP3: (gated) LoRA, B6 loss
@dataclass
class LCfg:
    rank: int = 4
    gate: bool = True
    tau: float = 0.5
    temp: float = 0.05
    k_commit: int = 10           # gate active at decode steps 0..k_commit-1 (t > 1 - (k_commit - 0.5)/K)
    steps: int = 20_000
    lr: float = 1e-3
    batch: int = 256
    pool: int = 60_000
    gate_src: str = "cur"        # decode-time detector input: "cur" = pi_ref activations at the edited state;
                                 # "ref" = pi_ref activations along pi_ref's own (unedited) trajectory from the same noise
    latch: bool = False          # decode-time: gate = running max over steps (until k_commit)


LORA_CFGS = {
    "gA": LCfg(),                               # tau 0.5, T 0.05
    "gB": LCfg(tau=0.3, temp=0.1),              # wider soft gate
    "gC": LCfg(tau=0.2, temp=0.1, k_commit=12),
    "gR": LCfg(gate_src="ref"),
    "gRB": LCfg(gate_src="ref", tau=0.3, temp=0.1),
    "gL": LCfg(latch=True),
    "gK12": LCfg(tau=0.3, temp=0.1, k_commit=12),
    "gK14": LCfg(tau=0.3, temp=0.1, k_commit=14),
    "gK12L": LCfg(k_commit=12, gate_src="ref", latch=True),
    "gK14L": LCfg(k_commit=14, gate_src="ref", latch=True),
    "nA": LCfg(gate=False),                     # GP3 control
}


class Holder:
    g = None


class GLoRALinear(nn.Module):
    """base(x) + g * (x A^T) B^T, with g a per-sample gate shared by all adapters of one forward."""

    def __init__(self, base: nn.Linear, r: int, holder: Holder):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.holder = holder

    def forward(self, x):
        d = (x @ self.A.t()) @ self.B.t()
        g = self.holder.g
        if g is not None:
            d = d * g.unsqueeze(-1)
        return self.base(x) + d


class Detector(nn.Module):
    """K10c per-track detectors (fit exactly as in E1's round 1) read from FROZEN pi_ref activations."""

    def __init__(self, det_w, det_b, mu_d, sd_d, det_q, layer):
        super().__init__()
        for k, v in dict(det_w=det_w, det_b=det_b, mu_d=mu_d, sd_d=sd_d).items():
            self.register_buffer(k, v)
        self.lq = float(np.log(det_q / (1 - det_q)))
        self.layer = layer

    @torch.no_grad()
    def prob(self, ref_net, x, t, c):
        _, acts = net_forward(ref_net, x, t, c, capture=True)
        pd = (feats(acts, x, ref_net.c_feat(c), self.layer) - self.mu_d) / self.sd_d
        s = pd @ self.det_w + self.det_b + self.lq          # logit of per-track probability
        p = torch.sigmoid(s)
        return p.max(1).values if p.dim() == 2 else p


@torch.no_grad()
def fit_k10c_detectors(cell: Cell, T, cfg: GCfg = K10C):
    """Detectors of the K10c round-1 graft (same streams and code as graft.fit_graft)."""
    net, seed = cell.ref.net, cell.seed
    g = egen(seed, 10)
    n = cfg.n_train
    c = torch.rand(n, generator=g); xi = torch.randn(n, 2, generator=g)
    layers = tuple(sorted({cfg.det_layer, cfg.feat_layer}))
    a, tr = decode(net, xi, c, layers=layers)
    lab = T.label(a, c, loose=2.0)
    bad = np.isin(lab, T.bad)
    det_w, det_b, mu_d, sd_d, _ = _detectors(cfg, tr, net.c_feat(c), lab, bad, T)
    return Detector(det_w, det_b, mu_d, sd_d, cfg.det_q, cfg.det_layer)


class LoRAPolicy:
    def __init__(self, ref, lnet, holder, det: Detector | None, lc: LCfg):
        self.ref, self.lnet, self.holder, self.det, self.lc = ref, lnet, holder, det, lc
        self.t_min = 1.0 - (lc.k_commit - 0.5) * DT

    def gate(self, x, t, c):
        if self.det is None:
            return None
        p = self.det.prob(self.ref.net, x, t, c)
        return torch.sigmoid((p - self.lc.tau) / self.lc.temp) * (t > self.t_min).float()

    def velocity(self, x, t, c):
        self.holder.g = self.gate(x, t, c)
        v = self.lnet(x, t, c)
        self.holder.g = None
        return v

    @torch.no_grad()
    def sample_gate(self, xi, c, batch=32768):
        outs, gm = [], []
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float(); cc = c[i:i + batch].float()
            m = torch.zeros(x.shape[0]) if self.det is not None else torch.ones(x.shape[0])   # no gate: always on
            xr = x.clone()
            run = torch.zeros(x.shape[0])
            for k in range(K_STEPS):
                t = torch.full((x.shape[0],), 1.0 - k * DT)
                g = self.gate(xr if self.lc.gate_src == "ref" else x, t, cc)
                if g is not None:
                    if self.lc.latch:
                        run = torch.maximum(run, g)
                        g = run * (t > self.t_min).float()
                    m = torch.maximum(m, g)
                if self.det is not None and self.lc.gate_src == "ref":
                    xr = xr - DT * self.ref.net(xr, t, cc)
                self.holder.g = g
                v = self.lnet(x, t, cc)
                self.holder.g = None
                x = x - DT * v
            outs.append(x); gm.append(m)
        return torch.cat(outs), torch.cat(gm).numpy()

    @torch.no_grad()
    def sample(self, xi, c, batch=32768):
        return self.sample_gate(xi, c, batch)[0]


def build_lora(ref, rank, holder):
    lnet = copy.deepcopy(ref.net)
    for p in lnet.parameters():
        p.requires_grad_(False)
    for b in lnet.blocks:
        b.fc1 = GLoRALinear(b.fc1, rank, holder)
        b.fc2 = GLoRALinear(b.fc2, rank, holder)
    lnet.inp = GLoRALinear(lnet.inp, rank, holder)
    lnet.out = GLoRALinear(lnet.out, rank, holder)
    params = [q for n_, q in lnet.named_parameters() if n_.endswith(".A") or n_.endswith(".B")]
    for q in params:
        q.requires_grad_(True)
    return lnet, params


def b6_data(cell: Cell, n_pool):
    """B6's training set: pi_ref samples whose E3 track is assigned and not bad (explore/E3 tracks.py and
    e3.anchor_fn 'track60k', = toyrepair/arms/b6_track.py; that module's toyrepair.arms.base is not on this branch)."""
    import tracks as TR
    POOL_PURPOSE = 90_001
    with torch.no_grad():
        tr = TR.Tracks(cell.ref, cell.seed)
    bad, miss = TR.bad_tracks(tr, cell.ed.c, cell.ed.a)
    P = E.make_pool(cell.ref, cell.seed, n_pool, purpose=POOL_PURPOSE)
    t = tr.assign(P.a, P.c)
    keep = torch.from_numpy((t >= 0) & ~np.isin(t, bad))
    info = {"track_n": int(tr.n_tracks), "track_bad": len(bad), "track_neg_miss": miss, "track_keep": int(keep.sum())}
    return P.c[keep], P.a[keep], info


def unit_lora(arm, cfgname, setting, seed, npm=4):
    lc = LORA_CFGS[cfgname]
    tag = f"{arm}-{cfgname}"
    t_unit = time.time()
    name = f"{setting}_s{seed}_n{npm}"
    res = os.path.join(RES, tag, f"{name}.json")
    if os.path.exists(res):
        print("DONE (exists)", res); return True
    ck = os.path.join(CACHE, f"ckpt_{tag}_{name}.pt")
    cell = load_cell(setting, seed, npm)
    holder = Holder()
    det, det_info = None, {}
    if lc.gate:
        dk = os.path.join(CACHE, f"det_{name}.pt")
        if os.path.exists(dk):
            det_sd, det_info = torch.load(dk, weights_only=False)
            det = Detector(det_sd["det_w"], det_sd["det_b"], det_sd["mu_d"], det_sd["sd_d"], K10C.det_q, K10C.det_layer)
        else:
            T = build_tracker(cell.ref, seed, cell.ed)
            det = fit_k10c_detectors(cell, T)
            det_info = {"bad_tracks_E1": list(T.bad), "tracker": tracker_diag(T, cell)}
            _save_pt(dk, ({k: v for k, v in det.state_dict().items()}, det_info))
    torch.manual_seed(seed)
    lnet, params = build_lora(cell.ref, lc.rank, holder)
    pol = LoRAPolicy(cell.ref, lnet, holder, det, lc)
    c_tr, a_tr, dinfo = b6_data(cell, lc.pool)
    opt = torch.optim.AdamW(params, lr=lc.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    g = E.gen(seed, E.P_TRAIN)
    step0, train_s = 0, 0.0
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        lnet.load_state_dict(st["lnet"]); opt.load_state_dict(st["opt"]); g.set_state(st["g"])
        step0, train_s = st["step"], st["train_s"]
    t0 = time.time()
    B = lc.batch
    for step in range(step0, lc.steps):
        lr = cosine(step, lc.steps, min(500, lc.steps // 10), lc.lr, 0.1 * lc.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        j = E._pick(c_tr.shape[0], B, g)
        t_ = FlowPolicy.sample_time(B, g)
        a = a_tr[j]; noise = torch.randn(a.shape, generator=g); cc = c_tr[j]
        x_t = t_.reshape(-1, 1) * noise + (1 - t_.reshape(-1, 1)) * a
        v = pol.velocity(x_t, t_, cc)
        loss = ((v - (noise - a)) ** 2).mean(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 500 == 0 and step + 1 < lc.steps and time.time() - t_unit > CHUNK_S:
            _save_pt(ck, {"lnet": lnet.state_dict(), "opt": opt.state_dict(), "g": g.get_state(), "step": step + 1,
                          "train_s": train_s + time.time() - t0})
            print(f"[{tag}] {name} checkpoint at step {step + 1} ({time.time() - t_unit:.0f}s)", flush=True)
            return False
    train_s += time.time() - t0
    lnet.eval()
    for q in lnet.parameters():
        q.requires_grad_(False)
    ad = {k: v for k, v in lnet.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    _save_pt(os.path.join(EDITS, tag, f"{name}.pt"), {"kind": "lora", "lcfg": asdict(lc), "lora": ad,
                                                       "detector": det.state_dict() if det is not None else None})
    ev = evaluate(cell, pol)
    rec = {"arm": arm, "cfgname": cfgname, "cell": name, "setting": setting, "seed": seed, "npm": npm,
           "lcfg": asdict(lc), **dinfo, **det_info, "params_added": int(sum(q.numel() for q in params)),
           "train_s": round(train_s, 1), **ev}
    _save_json(res, rec)
    print(f"[{tag}] {name} D {ev['D']:.4f} Rem {ev['Removed']:.3f} Col {ev['Collateral']:.3f} off {ev['off']:.3f} "
          f"flips {ev['flips']:.3f} gate {ev['gate_retained']:.3f} ({train_s:.0f}s) DONE", flush=True)
    return True


# =============================================================== evaluation (oracle side)
def evaluate(cell: Cell, pol):
    """experiment.compute + outcome flips + gate firing rate on retained decodes (oracle labels for scoring)."""
    m = E.compute(cell.bank, pol)
    out = {"D": m["tv_to_ideal"], "Removed": 1.0 - m["neg_mass_ratio"], "Collateral": m["retain_tv"],
           "off": m["mass_off"], "D_ref": m["tv_to_ideal_ref"], "flips": m["retain_noise_flip"],
           "neg_to_retain": m["neg_noise_to_retain"], "disp_retain": m["disp_retain_mean"],
           "masses": [m[f"mass_m{j}"] for j in range(E.MIX.n_modes)]}
    if hasattr(pol, "sample_gate"):
        b = cell.bank
        _, gm = pol.sample_gate(b.xi, b.c)
        fired = gm > 0.5 if gm.dtype != bool else gm
        ret = np.isin(b.lab_ref, cell.ed.retain)
        neg = np.isin(b.lab_ref, cell.ed.forget)
        out["gate_retained"] = float(fired[ret].mean())
        out["gate_forget"] = float(fired[neg].mean()) if neg.any() else float("nan")
    else:
        out["gate_retained"] = float("nan")
    return out


def unit_e1eval(setting, seed, npm=4):
    """Re-evaluate a stored E1 / E5 K10c graft on its cell (D check + flips + gate rate)."""
    name = f"{setting}_s{seed}_n{npm}"
    res = os.path.join(RES, "E1-stored", f"{name}.json")
    if os.path.exists(res):
        print("DONE (exists)", res); return True
    cands = [os.path.join(ROOT, "explore", "E1", "grafts", f"grafts_K10c_{name}.pt"),
             os.path.join(HERE, "ref", "E5", "grafts", f"grafts_K10c_{name}.pt")]
    src = [p for p in cands if os.path.exists(p)]
    assert src, f"no stored graft for {name}"
    jsrc = [os.path.join(ROOT, "explore", "E1", "results", f"graft_K10c_{name}.json"),
            os.path.join(HERE, "ref", "E5", "results", f"graft_K10c_{name}.json")]
    stored = [json.load(open(p)) for p in jsrc if os.path.exists(p)]
    cell = load_cell(setting, seed, npm)
    grs = load_grafts(torch.load(src[0], weights_only=False), K10C)
    ev = evaluate(cell, GraftPolicy(cell.ref.net, grs, K10C))
    rec = {"arm": "E1", "cfgname": "stored", "cell": name, "setting": setting, "seed": seed, "npm": npm,
           "source": os.path.relpath(src[0], ROOT), "stored_D": stored[0]["D"] if stored else None,
           "params_added": int(sum(p.numel() for g_ in grs for p in g_.parameters())),
           "buffers_added": int(sum(b.numel() for g_ in grs for b in g_.buffers())), **ev}
    _save_json(res, rec)
    print(f"[E1-stored] {name} D {ev['D']:.4f} (stored {rec['stored_D']}) flips {ev['flips']:.3f} gate {ev['gate_retained']:.3f} DONE", flush=True)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["gp1", "gp2", "gp3", "e1fit", "e1eval"])
    ap.add_argument("--setting", required=True, choices=list(SETTINGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--npm", type=int, default=4)
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.arm in ("gp1", "e1fit"):
        unit_graft("GP1" if a.arm == "gp1" else "E1", a.cfg or ("A" if a.arm == "gp1" else "e1"), a.setting, a.seed, a.npm)
    elif a.arm in ("gp2", "gp3"):
        unit_lora(a.arm.upper(), a.cfg or ("gA" if a.arm == "gp2" else "nA"), a.setting, a.seed, a.npm)
    else:
        unit_e1eval(a.setting, a.seed, a.npm)
