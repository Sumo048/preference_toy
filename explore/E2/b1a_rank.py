"""Effective rank of B1a's weight change per layer: train the frozen recipe in-process (same code as
cond_cell.py), SVD of W_B1a - W_ref, and the share of energy in the top 1/2/4/8 singular values.
Also: D of B1a truncated to its top-r singular directions per layer (does a rank-r B1a keep its D?)."""
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import lib, fit
from lib import E

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--layout", default="random")
ap.add_argument("--npm", type=int, default=16)
args = ap.parse_args()
torch.set_num_threads(2)
cell = lib.load_cell(args.seed, args.layout, args.npm)
ctx, rows = fit.ctx_rows(cell, steps=5000, lr=1e-4)
cp = os.path.join(os.environ["TOYPREF_RUNS"], "E2cache", f"b1a_{cell.name}.pt")
if os.path.exists(cp):
    pol = copy.deepcopy(cell.ref); pol.load_state_dict(torch.load(cp))
else:
    # same as experiment.train_arm for the reroute arm (rows identical by construction)
    fn = E.step_fn(fit.ARM, ctx, rows)
    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref); pol.train()
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(pol.parameters(), lr=ctx.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    g = E.gen(ctx.seed, E.P_TRAIN)
    for step in range(ctx.steps):
        lr = E.cosine(step, ctx.steps, min(500, ctx.steps // 10), ctx.lr, 0.1 * ctx.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    torch.save(pol.state_dict(), cp)
full = lib.evaluate(cell, []) if False else None
r_full = E.compute(cell.bank, pol)
print(cell.name, f"B1a in-process D {r_full['tv_to_ideal']:.3f}")
ref_sd, sd = cell.ref.state_dict(), pol.state_dict()
out = {"cell": cell.name, "D_full": r_full["tv_to_ideal"], "layers": {}}
names = [k for k in sd if k.endswith("weight") and sd[k].ndim == 2]
tot = sum(float(((sd[k] - ref_sd[k]) ** 2).sum()) for k in sd)
for k in names:
    d = (sd[k] - ref_sd[k]).double()
    s = torch.linalg.svdvals(d)
    e = (s ** 2) / (s ** 2).sum()
    cum = torch.cumsum(e, 0)
    out["layers"][k] = {"fro": float(d.norm()), "share_total": float((d ** 2).sum()) / tot,
                        "top1": float(cum[0]), "top2": float(cum[min(1, len(cum) - 1)]),
                        "top4": float(cum[min(3, len(cum) - 1)]), "top8": float(cum[min(7, len(cum) - 1)])}
    print(f"  {k:28s} |dW| {float(d.norm()):.3f} share {out['layers'][k]['share_total']:.3f} "
          f"top1 {float(cum[0]):.2f} top4 {out['layers'][k]['top4']:.2f} top8 {out['layers'][k]['top8']:.2f}")
bias_share = sum(float(((sd[k] - ref_sd[k]) ** 2).sum()) for k in sd if k not in names) / tot
out["bias_and_norm_share"] = bias_share
for r in (1, 4, 8, 16):
    tsd = copy.deepcopy(ref_sd)
    for k in sd:
        d = sd[k] - ref_sd[k]
        if k in names:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            d = (U[:, :r] * S[:r]) @ Vh[:r]
        tsd[k] = ref_sd[k] + d
    q = copy.deepcopy(cell.ref); q.load_state_dict(tsd)
    rr = E.compute(cell.bank, q)
    out[f"D_rank{r}"] = rr["tv_to_ideal"]
    print(f"  B1a truncated to rank {r} per layer: D {rr['tv_to_ideal']:.3f} Rem {1-rr['neg_mass_ratio']:.3f} Col {rr['retain_tv']:.3f}")
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", f"b1a_rank_{cell.name}.json"), "w"))
