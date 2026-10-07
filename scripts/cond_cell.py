"""Run conditional-toy arms on one cell (seed, n_forget, n_neg); one JSON per arm, skip-if-exists."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.config import RUNS
from toypref.cond import experiment as E
from toypref.cond.task import CondFlow, frozen

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n_forget", type=int, required=True)
ap.add_argument("--n_neg", type=int, required=True); ap.add_argument("--arms", default="all")
ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--out", default=os.path.join(RUNS, "cond", "cells"))
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
sd = torch.load(os.path.join(RUNS, "cond_base", f"s{s}", "policy.pt"), map_location="cpu", weights_only=False)
ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); ref = frozen(ref)
cell = f"s{s}_k{args.n_forget}_n{args.n_neg}"
od = os.path.join(args.out, cell); os.makedirs(od, exist_ok=True)
names = list(E.ARMS) if args.arms == "all" else args.arms.split(",")
todo = [n for n in names if args.force or not os.path.exists(os.path.join(od, f"{n}.json"))]
if not todo:
    sys.exit(0)
pool = E.make_pool(ref, s, 200_000)
ed = E.editset(pool, s, args.n_forget, args.n_neg)
assert len(set(ed.c.tolist())) == len(ed.c), "conditions must be unique"
anchor = E.make_pool(ref, s, 20_000, purpose=E.P_ANCHOR)
pu = None
if any(E.ARMS[n].rule == "pu" for n in todo):
    upool = E.make_pool(ref, s, 20_000, purpose=E.P_PU + 1)
    pu = E.train_pu(ed, upool, s, prior=args.n_forget / E.MIX.n_modes)
ctx = E.Ctx(ref=ref, seed=s, ed=ed, pool=pool, anchor=anchor, steps=args.steps, pu=pu)
bank = E.Bank(ref, s, ed)
probe = E.make_probe(bank)
print(f"[{cell}] forget={ed.forget} |T_n|={len(ed.uid)} arms={len(todo)}", flush=True)
for n in todo:
    arm = E.ARMS[n]
    t0 = time.time()
    pol, info = E.train_arm(arm, ctx, probe=probe)
    met = E.compute(bank, pol)
    rec = {"cell": cell, "seed": s, "n_forget": args.n_forget, "n_neg": args.n_neg, "forget": ed.forget,
           "arm": n, "kind": arm.kind, "rule": arm.rule, "support": arm.support, "transplant": arm.transplant,
           "anchor": arm.anchor, "practical": arm.practical, "train_s": round(time.time() - t0, 1),
           **{k: v for k, v in info.items() if k != "train_history"}, **met, "train_history": info.get("train_history")}
    tmp = os.path.join(od, f".{n}.tmp"); json.dump(rec, open(tmp, "w")); os.replace(tmp, os.path.join(od, f"{n}.json"))
    print(f"[{cell}] {n:24s} tv {met['tv_to_ideal']:.3f} forget {met['neg_mass_ratio']:.3f} rTV {met['retain_tv']:.3f} "
          f"off {met['mass_off']:.3f} far {met['neg_still_forget_far']:.3f} "
          f"tgtF {info.get('rows_target_frac_in_forget', float('nan')):.2f} tgtOff {info.get('rows_target_frac_off', float('nan')):.2f} "
          f"tx {info.get('rows_transplant_frac_bad', float('nan')):.2f} ({rec['train_s']}s)", flush=True)
