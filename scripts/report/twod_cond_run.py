"""Report figures, conditional toy: train one arm on one cell (mirrors scripts/cond_cell.py) and dump
samples at fixed conditions from one fixed noise batch, probe history and the repo's compute() metrics.
usage: twod_cond_run.py --arm NAME|common [--seed 0 --n_forget 4 --n_neg 16 --layout random]
TOYPREF_RUNS must be the report_2d dir (cond base in $TOYPREF_RUNS/cond_base/s0)."""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, torch
from toypref.config import RUNS
from toypref.cond import experiment as E
from toypref.cond.task import CondFlow, frozen

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_forget", type=int, default=4); ap.add_argument("--n_neg", type=int, default=16)
ap.add_argument("--layout", default="random"); ap.add_argument("--steps", type=int, default=5000)
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
sd = torch.load(os.path.join(RUNS, "cond_base", f"s{s}", "policy.pt"), map_location="cpu", weights_only=False)
ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); ref = frozen(ref)
pool = E.make_pool(ref, s, 200_000)
ed = E.editset(pool, s, args.n_forget, args.n_neg, args.layout)
assert len(set(ed.c.tolist())) == len(ed.c)
CS = [0.1, 0.35, 0.6, 0.85]
NZ = 4000
gz = torch.Generator().manual_seed(1234 + s)
XI = torch.randn(NZ, 2, generator=gz)
out = os.path.join(RUNS, "report_cond"); os.makedirs(out, exist_ok=True)
if args.arm == "common":
    ra = np.stack([ref.sample(XI, torch.full((NZ,), c)).numpy() for c in CS])
    np.savez(os.path.join(out, "common.npz"), XI=XI.numpy(), CS=np.array(CS), ref_a=ra,
             neg_c=ed.c.numpy(), neg_a=ed.a.numpy(), neg_xi=ed.xi.numpy(), neg_mode=pool.mode[ed.uid],
             forget=np.array(ed.forget), retain=np.array(ed.retain))
    print("common", ed.forget, len(ed.uid)); sys.exit(0)
anchor = E.make_pool(ref, s, 20_000, purpose=E.P_ANCHOR)
ctx = E.Ctx(ref=ref, seed=s, ed=ed, pool=pool, anchor=anchor, steps=args.steps, pu=None)
bank = E.Bank(ref, s, ed); probe = E.make_probe(bank)
arm = E.ARMS[args.arm]; t0 = time.time()
pol, info = E.train_arm(arm, ctx, probe=probe)
met = E.compute(bank, pol)
a = np.stack([pol.sample(XI, torch.full((NZ,), c)).numpy() for c in CS])
labs = np.stack([E.MIX.assign(a[i], np.full(NZ, c)) for i, c in enumerate(CS)])
hist = [{k: h[k] for k in h if k == "step" or k.startswith("probe_")} for h in (info.get("train_history") or [])]
np.savez(os.path.join(out, f"{args.arm}.npz"), a=a, lab=labs, final=json.dumps(met), hist=json.dumps(hist))
print(args.arm, {k: round(met[k], 3) for k in ("tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off")}, f"{time.time()-t0:.0f}s", flush=True)
