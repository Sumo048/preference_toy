"""Train the conditional base policy for one seed; report mode masses per condition band."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.config import RUNS
from toypref.cond import experiment as E

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--steps", type=int, default=40_000)
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
pol = E.train_base(args.seed, steps=args.steps)
d = os.path.join(RUNS, "cond_base", f"s{args.seed}"); os.makedirs(d, exist_ok=True)
torch.save({"state": pol.state_dict(), "kw": pol.kw}, os.path.join(d, "policy.pt"))
g = E.gen(args.seed, E.P_EVAL); c = torch.rand(20000, generator=g); xi = torch.randn(20000, 2, generator=g)
lab = E.MIX.assign(pol.sample(xi, c), c)
rep = {"seed": args.seed, "mass": E.masses(lab).round(4).tolist(),
       "by_c": {f"{lo:.1f}": E.masses(lab[(c.numpy() >= lo) & (c.numpy() < lo + 0.2)]).round(3).tolist() for lo in np.arange(0, 1, 0.2)}}
json.dump(rep, open(os.path.join(d, "report.json"), "w"), indent=1); print(json.dumps(rep))
