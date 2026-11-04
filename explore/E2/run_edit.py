"""Run one E2 direction-edit method on one cell; writes explore/E2/results/<cell>/<method>.json.

Method names:  <kind>[_<layer>][_t<lo>-<hi>][_r<rank>][_s<steps>]
  r1      rank-r edit W += sum_i u_i v_i^T (+ bias via beta), u and v learned       layer: out | fc2.j | fc1.j
  r1p     rank-1 with v fixed to the contrast-probe direction of the layer's input; u, beta learned
  abl     ablate the probe direction (mean-ablation) at the layer's input site, no training
  t<lo>-<hi>: gate the edit to t in [lo, hi] (t = 1 noise, 0 action)
Probe direction: logistic regression (label-free labels: in B_i vs not, at matched conditions) on
the layer's input activations, pooled over steps of the gate window (all steps if ungated).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
import lib, fit

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--layout", default="random")
ap.add_argument("--npm", type=int, required=True); ap.add_argument("--method", required=True)
ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--force", action="store_true")
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()
torch.set_num_threads(args.threads)

cell = lib.load_cell(args.seed, args.layout, args.npm)
od = os.path.join(os.path.dirname(__file__), "results", cell.name)
os.makedirs(od, exist_ok=True)
outp = os.path.join(od, f"{args.method}.json")
if os.path.exists(outp) and not args.force:
    print("exists", outp); sys.exit(0)

parts = args.method.split("_")
kind = parts[0]
layer = parts[1] if len(parts) > 1 and not parts[1][0] in "trs" else "out"
lo, hi, rank, steps = 0.0, 1.0, 1, 2000
for p in parts[1:]:
    if p.startswith("t") and "-" in p:
        lo, hi = map(float, p[1:].split("-"))
    elif p.startswith("r") and p[1:].isdigit():
        rank = int(p[1:])
    elif p.startswith("s") and p[1:].isdigit():
        steps = int(p[1:])
gated = (lo, hi) != (0.0, 1.0)
gate = (lambda t: lib.tgate(t, lo, hi)) if gated else None

LAYERS = layer.split("+")
lk, _, lj = LAYERS[0].partition(".")
src = {"fc2": f"b{lj}.z", "out": "phi", "fc1": f"b{lj}.in"}[lk]
din = {"fc2": 128, "out": 128, "fc1": 256}[lk]
dout = {"fc2": 128, "out": 2, "fc1": 128}[lk]

t0 = time.time()
info = {"method": args.method, "kind": kind, "layer": layer, "t_window": [lo, hi], "rank": rank}


def probe_direction():
    """Label-free contrast direction at `src`, pooled over the steps inside the gate window."""
    B = lib.neg_basins(cell)
    c, xi, a, bad, own, _ = lib.matched_samples(cell, B, m=64, seed_off=1)
    _, cap = lib.decode(cell.ref.net, xi, c, capture=[src])
    ks = [k for k in range(20) if lo <= 1 - k / 20 <= hi]
    X = torch.cat([cap[src][k] for k in ks]).numpy()
    y = np.tile(bad, len(ks))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = LogisticRegression(C=0.1, max_iter=500).fit((X - mu) / sd, y)
    w = p.coef_[0] / sd                      # direction in raw activation space
    v = torch.tensor(w / np.linalg.norm(w), dtype=torch.float32)
    proj = X @ v.numpy()
    info.update({"probe_bad_frac": float(bad.mean()), "probe_steps": ks,
                 "probe_proj_mean_bad": float(proj[y].mean()), "probe_proj_mean_good": float(proj[~y].mean()),
                 "probe_proj_sd": float(proj.std())})
    return v, float(proj[~y].mean()), float(proj.mean())


def probe_directions_per_cluster(min_members=2):
    """One label-free contrast direction per miner cluster (PLAN 6.2 affinity), at `src`."""
    import clusters
    B = lib.neg_basins(cell)
    cl, _ = clusters.cluster(cell, B)
    c, xi, a, bad, own, _ = lib.matched_samples(cell, B, m=64, seed_off=1)
    _, cap = lib.decode(cell.ref.net, xi, c, capture=[src])
    ks = [k for k in range(20) if lo <= 1 - k / 20 <= hi]
    V, MG = [], []
    sizes = []
    for q in np.unique(cl):
        mem = np.flatnonzero(cl == q)
        if len(mem) < min_members:
            continue
        sel = np.isin(own, mem)
        X = torch.cat([cap[src][k][torch.from_numpy(sel)] for k in ks]).numpy()
        y = np.tile(bad[sel], len(ks))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        p = LogisticRegression(C=0.1, max_iter=500).fit((X - mu) / sd, y)
        w = p.coef_[0] / sd
        v = w / np.linalg.norm(w)
        proj = X @ v
        if kind == "ablc":       # difference-of-means direction (carries real variance)
            v = X[y].mean(0) - X[~y].mean(0)
            v = v / np.linalg.norm(v)
            proj = X @ v
        V.append(v); MG.append(float(np.quantile(proj[~y], 0.95)) if kind != "ablc" else float(proj[~y].mean()))
        sizes.append(len(mem))
    info.update({"n_clusters": int(len(np.unique(cl))), "cluster_sizes_used": sizes})
    return torch.tensor(np.stack(V), dtype=torch.float32), torch.tensor(MG)


edits = []
if kind == "ablc":
    # ablate every cluster's mean-difference direction at the site (orthonormalised, sequential
    # projections onto the good-side mean, i.e. mean-ablation of the bad-vs-good contrast)
    Vfix, MG = probe_directions_per_cluster()
    Q, _ = torch.linalg.qr(Vfix.T)
    with torch.no_grad():
        B_ = lib.neg_basins(cell)
    for i in range(Q.shape[1]):
        edits.append(lib.Ablate(src, Q[:, i], gate=gate, mean=None))
    # mean-ablation offsets: project onto the pooled activation mean (computed on the anchor pool)
    xa, ca = cell.anchor.xi[:2048], cell.anchor.c[:2048]
    _, cap0 = lib.decode(cell.ref.net, xa, ca, capture=[src])
    mu = cap0[src].reshape(-1, cap0[src].shape[-1]).mean(0)
    for e in edits:
        e.mean = float(mu @ e.v)
    info["ablc_n_dirs"] = int(Q.shape[1])
elif kind == "r1pc":
    Vfix, MG = probe_directions_per_cluster()
    R = Vfix.shape[0]
    g = torch.Generator().manual_seed(1234 + args.seed)
    params = {"u": 1e-3 * torch.randn(R, dout, generator=g), "beta": -MG.clone()}

    def build(P):
        return [lib.Rank1(layer, P["u"][i], Vfix[i], 1.0, beta=P["beta"][i], gate=gate) for i in range(R)]
    tr = fit.Trainable(params, build)
    logs = []
    edits = fit.train_edit(cell, tr, steps=steps, lr=args.lr, log=lambda s: (logs.append(s), print(s, flush=True)))
    info["train_log"] = logs
elif kind == "abl":
    v, mgood, mall = probe_direction()
    edits = [lib.Ablate(src, v, gate=gate, mean=mgood)]
elif kind == "r1" and len(LAYERS) > 1:
    # rank-r learned edits on several layers at once
    g = torch.Generator().manual_seed(1234 + args.seed)
    params, dims = {}, {}
    for L in LAYERS:
        k_, _, _ = L.partition(".")
        di, do = {"fc2": 128, "out": 128, "fc1": 256}[k_], {"fc2": 128, "out": 2, "fc1": 128}[k_]
        params[f"{L}|u".replace(".", "_")] = 1e-3 * torch.randn(rank, do, generator=g)
        params[f"{L}|v".replace(".", "_")] = torch.randn(rank, di, generator=g) / di ** 0.5
        params[f"{L}|b".replace(".", "_")] = torch.zeros(rank)

    def build(P):
        out = []
        for L in LAYERS:
            q = L.replace(".", "_")
            out += [lib.Rank1(L, P[q + "|u"][i], P[q + "|v"][i], 1.0, beta=P[q + "|b"][i], gate=gate)
                    for i in range(rank)]
        return out
    tr = fit.Trainable(params, build)
    logs = []
    edits = fit.train_edit(cell, tr, steps=steps, lr=args.lr, log=lambda s: (logs.append(s), print(s, flush=True)))
    info["train_log"] = logs
elif kind in ("r1", "r1p"):
    g = torch.Generator().manual_seed(1234 + args.seed)
    params = {"u": 1e-3 * torch.randn(rank, dout, generator=g), "beta": torch.zeros(rank)}
    if kind == "r1":
        params["v"] = torch.randn(rank, din, generator=g) / din ** 0.5
    else:
        v, mgood, mall = probe_direction()
        vfix = v.reshape(1, -1)
        params["beta"] = torch.full((1,), -mgood)      # zero response on the good mean

    def build(P):
        V = P["v"] if "v" in P else vfix
        return [lib.Rank1(layer, P["u"][i], V[i], 1.0, beta=P["beta"][i], gate=gate) for i in range(rank)]
    tr = fit.Trainable(params, build)
    logs = []
    edits = fit.train_edit(cell, tr, steps=steps, lr=args.lr, log=lambda s: (logs.append(s), print(s, flush=True)))
    info["train_log"] = logs
    with torch.no_grad():
        info["norm_uv"] = [float(e.u.norm() * e.v.norm()) for e in edits]
else:
    raise SystemExit(f"unknown kind {kind}")

r = lib.evaluate(cell, edits)
info.update(r)
info["sec"] = round(time.time() - t0, 1)
json.dump(info, open(outp, "w"))
print(cell.name, args.method, lib.fmt(r), f"({info['sec']}s)", flush=True)
