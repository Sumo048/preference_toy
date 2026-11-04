"""Post-hoc low-rank projection of the frozen recipe: train B1a (same code path as cond_cell.py),
then keep only the top-r singular directions of each weight matrix's change (biases and LayerNorm
changes kept in full, or dropped with --nobias). Writes results/<cell>/b1a_trunc_r<r>[_nb].json.
Also writes b1a_inproc.json (full B1a, a check against cond_cell.py)."""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import lib, fit
from lib import E

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--layout", default="random")
ap.add_argument("--npm", type=int, required=True); ap.add_argument("--ranks", default="1,2,3,4,6,8")
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()
torch.set_num_threads(args.threads)
cell = lib.load_cell(args.seed, args.layout, args.npm)
od = os.path.join(os.path.dirname(__file__), "results", cell.name)
os.makedirs(od, exist_ok=True)
t0 = time.time()


def b1a_state(cell):
    cp = os.path.join(os.environ["TOYPREF_RUNS"], "E2cache", f"b1a_{cell.name}.pt")
    if os.path.exists(cp):
        return torch.load(cp)
    ctx, rows = fit.ctx_rows(cell, steps=5000, lr=1e-4)
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
    sd = {k: v.detach().clone() for k, v in pol.state_dict().items()}
    torch.save(sd, cp)
    return sd


def met(q):
    r = E.compute(cell.bank, q)
    return {"D": r["tv_to_ideal"], "Removed": 1 - r["neg_mass_ratio"], "Collateral": r["retain_tv"], "off": r["mass_off"]}


sd = b1a_state(cell)
ref_sd = cell.ref.state_dict()
q = copy.deepcopy(cell.ref); q.load_state_dict(sd)
full = met(q)
json.dump({**full, "method": "b1a_inproc"}, open(os.path.join(od, "b1a_inproc.json"), "w"))
print(cell.name, "b1a_inproc", lib.fmt(full), flush=True)
names = [k for k in sd if k.endswith("weight") and sd[k].ndim == 2]
for nb in (False, True):
    for r in [int(x) for x in args.ranks.split(",")]:
        tsd = copy.deepcopy(ref_sd)
        n_par = 0
        for k in sd:
            d = sd[k] - ref_sd[k]
            if k in names:
                U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                rr = min(r, S.shape[0])
                d = (U[:, :rr] * S[:rr]) @ Vh[:rr]
                n_par += rr * (d.shape[0] + d.shape[1])
            elif nb:
                d = torch.zeros_like(d)
            else:
                n_par += d.numel()
            tsd[k] = ref_sd[k] + d
        q = copy.deepcopy(cell.ref); q.load_state_dict(tsd)
        res = met(q)
        name = f"b1a_trunc_r{r}" + ("_nb" if nb else "")
        json.dump({**res, "method": name, "n_params_edit": n_par, "sec": round(time.time() - t0, 1)},
                  open(os.path.join(od, f"{name}.json"), "w"))
        print(cell.name, name, lib.fmt(res), flush=True)
