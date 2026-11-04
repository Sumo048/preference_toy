"""Linear separability of bad-basin trajectories at matched conditions, per site and step.

Label-free labels: at each negative's condition c_i, fresh pi_ref decodes are 'bad' if they land in
B_i. Probe = logistic regression, held out by negative (GroupKFold). Prints AUC per site and k.
ORACLE line: share of 'bad' decodes in forget modes (purity), for the record only.
"""
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lib

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--layout", default="random")
ap.add_argument("--npm", type=int, default=16); ap.add_argument("--m", type=int, default=128)
args = ap.parse_args()
torch.set_num_threads(2)
t0 = time.time()
cell = lib.load_cell(args.seed, args.layout, args.npm)
B = lib.neg_basins(cell)
c, xi, a, bad, own, _ = lib.matched_samples(cell, B, m=args.m)
lab = lib.E.MIX.assign(a, c)
print(f"{cell.name} N={len(c)} bad={bad.mean():.3f} ORACLE purity={np.isin(lab[bad], cell.ed.forget).mean():.3f} "
      f"bad-recall-of-forget={bad[np.isin(lab, cell.ed.forget)].mean():.3f}")
sites = ["x", "h0", "b0.z", "h1", "b1.z", "h2", "b2.z", "h3", "phi"]
_, cap = lib.decode(cell.ref.net, xi, c, capture=[s for s in sites if s != "x"])
res = {}
for s in sites:
    A = cap[s] if s != "x" else cap["x"][:20]
    row = []
    for k in [0, 2, 4, 6, 8, 10, 13, 16, 19]:
        X = A[k].numpy()
        if s == "x":
            X = np.concatenate([X, c.numpy()[:, None]], 1)
        X = StandardScaler().fit_transform(X)
        aucs = []
        for tr, te in GroupKFold(3).split(X, bad, own):
            p = LogisticRegression(C=0.1, max_iter=300).fit(X[tr], bad[tr])
            aucs.append(roc_auc_score(bad[te], p.decision_function(X[te])))
        row.append(round(float(np.mean(aucs)), 3))
    res[s] = row
    print(f"{s:5s}", " ".join(f"{v:.3f}" for v in row), flush=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "out", f"probe_{cell.name}.json"), "w"))
print(f"done {time.time()-t0:.0f}s")
