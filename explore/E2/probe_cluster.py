"""Per-cluster probe: does a linear direction separate a cluster's bad-basin trajectories at matched
conditions? Label-free labels; ORACLE purity printed for the record."""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lib, clusters

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--layout", default="random")
ap.add_argument("--npm", type=int, default=16)
args = ap.parse_args()
torch.set_num_threads(1)
cell = lib.load_cell(args.seed, args.layout, args.npm)
B = lib.neg_basins(cell)
cl, A = clusters.cluster(cell, B)
mode = lib.E.MIX.assign(cell.ed.a, cell.ed.c)
print(cell.name, "clusters:", {int(k): int((cl == k).sum()) for k in np.unique(cl)})
for k in np.unique(cl):
    print(f"  cluster {k}: ORACLE modes {np.bincount(mode[cl == k], minlength=8).tolist()}")
c, xi, a, bad, own, _ = lib.matched_samples(cell, B, m=128)
sites = ["h0", "b0.z", "h1", "b1.z", "b2.z", "phi"]
_, cap = lib.decode(cell.ref.net, xi, c, capture=sites)
ks = [0, 4, 8, 12, 16, 19]
out = {}
for k_ in np.unique(cl):
    mem = np.flatnonzero(cl == k_)
    if len(mem) < 3:
        continue
    sel = np.isin(own, mem)
    print(f"cluster {k_} n={len(mem)} bad frac {bad[sel].mean():.3f}")
    for s in sites:
        row = []
        for k in ks:
            X = StandardScaler().fit_transform(cap[s][k].numpy()[sel])
            y, gr = bad[sel], own[sel]
            aucs = []
            for tr, te in GroupKFold(3).split(X, y, gr):
                p = LogisticRegression(C=0.1, max_iter=300).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], p.decision_function(X[te])))
            row.append(round(float(np.mean(aucs)), 3))
        out[f"{k_}/{s}"] = row
        print(f"  {s:5s}", " ".join(f"{v:.3f}" for v in row), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", f"probe_cluster_{cell.name}.json"), "w"))
