"""Where along the decode are forgotten modes decided?  (label-free; oracle only for diagnostics)

1. Rescue curve p(k) for each logged negative: M renoise branches from step k (PLAN Sec. 2).
   Rescue = branch ends in a non-bad tracker basin.
2. Linear separability by step and layer: logistic probe on activations at step k, trained on
   random (c, xi) labelled bad/good by the tracker on pi_ref's final action; held-out AUC.
"""
import json, sys, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *

def renoise_branch(net, x_k, k, c, g):
    t = 1.0 - k * DT
    if k == 0:
        return decode(net, torch.randn(x_k.shape, generator=g), c)
    tt = torch.full((x_k.shape[0],), t)
    xh = x_k - t * net_forward(net, x_k, tt, c)
    x2 = t * torch.randn(x_k.shape, generator=g) + (1 - t) * xh
    return decode(net, None, c, k_start=k, x_start=x2)

def main(seed, layout, npm, M=32, out=None):
    cell = load_cell(seed, layout, npm); ref = cell.ref; net = ref.net
    T = build_tracker(ref, seed, cell.ed)
    res = {"cell": cell.name, "tracker": tracker_diag(T, cell)}
    ed = cell.ed
    # 1. rescue curves of logged negatives
    _, tr = decode(net, ed.xi, ed.c, layers=("v",))
    X = tr["x"]                                   # (21, N, 2)
    g = egen(seed, 2)
    P = np.zeros((K_STEPS, len(ed.c)))
    for k in range(K_STEPS):
        xk = X[k].repeat_interleave(M, 0); cc = ed.c.repeat_interleave(M)
        a = renoise_branch(net, xk, k, cc, g)
        P[k] = (~T.is_bad(a, cc, loose=2.0)).reshape(-1, M).mean(1)
    pk = P.mean(1)
    kc = [int(next((k for k in range(K_STEPS) if (P[k:, i] < 0.2).all()), K_STEPS - 1)) for i in range(P.shape[1])]
    res["p_k_mean"] = pk.round(3).tolist(); res["k_commit_neg"] = kc
    res["k_commit_median"] = float(np.median(kc))
    # 2. probe AUC by step x layer
    n = 6000
    gg = egen(seed, 3); c = torch.rand(n, generator=gg); xi = torch.randn(n, 2, generator=gg)
    layers = ("inp", "b0", "b1", "b2", "b0.z", "b1.z", "b2.z", "pre_out")
    a, tr = decode(net, xi, c, layers=layers)
    y = T.is_bad(a, c, loose=2.0).astype(int)
    lab_o = np.isin(E.MIX.assign(a, c), ed.forget).astype(int)   # oracle diag
    res["probe_label_agree_oracle"] = float((y == lab_o).mean())
    cf = net.c_feat(c).numpy()
    tr_i, te_i = np.arange(0, 4000), np.arange(4000, n)
    auc = {}
    for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19]:
        row = {}
        feats = {"x+c": np.concatenate([tr["x"][k].numpy(), cf], 1)}
        for l in layers:
            feats[l] = np.concatenate([tr[l][k].numpy(), tr["x"][k].numpy(), cf], 1)
        feats["all"] = np.concatenate([tr[l][k].numpy() for l in layers] + [tr["x"][k].numpy(), cf], 1)
        for nm, Fm in feats.items():
            mu, sd = Fm[tr_i].mean(0), Fm[tr_i].std(0) + 1e-6
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((Fm[tr_i] - mu) / sd, y[tr_i])
            row[nm] = round(float(roc_auc_score(y[te_i], clf.decision_function((Fm[te_i] - mu) / sd))), 4)
        auc[k] = row
        print(cell.name, k, row, flush=True)
    res["probe_auc"] = auc
    json.dump(res, open(out or f"results/commit_{cell.name}.json", "w"), indent=1)
    print(cell.name, "p(k)", res["p_k_mean"], "k_commit median", res["k_commit_median"], flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
