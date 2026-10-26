"""Train the robo2d arms on seed 0 at the native setting (5000 steps, m64) and SAVE the policies.

robo_cell.py only keeps metrics; the report figures need the fine-tuned weights. This reproduces the
robo_cell.py native path (same rows, same ctx, same MT.train_arm) and writes, under --out:
  <arm>.pt   state dict + kw (FILTER_rej: nothing, rebuilt from seed + rows)
  metrics.json   per-arm EvalSet metrics (neg / noneg / held groups) computed in this task
"""
import argparse, json, os, sys, time
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, torch
from toypref.robo import data as DT, evaluate as EV, methods as MT, targets as TG
from toypref.robo.policy import load

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--arms", default="B0_frozen,NPO_b100_neg,KTO_b100_neg,R_hop_bsup_neg_AL,FILTER_rej,SDPO_fresh_b25,SDPO_pin_b25")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed; os.makedirs(args.out, exist_ok=True)
bd = os.path.join(args.base, f"s{s}")
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
log = DT.LogResult(lg["rows"], lg["diag"], lg["episodes"], lg["per_setup"])
train, held = DT.experiment_setups(s)
ev = EV.EvalSet(ref, s, train, held, log, M=DT.M_EVAL)
ctx = MT.Ctx(ref=ref, seed=s, rows=log.rows, steps=args.steps)
t0 = time.time()
names = args.arms.split(",")
if any(MT.ARMS[n].rule == "hop" or MT.ARMS[n].local or MT.ARMS[n].pos == "hop" for n in names):
    ctx.cache = TG.build_hop_cache(ref, s, log.rows); print(f"hop cache {time.time()-t0:.0f}s", flush=True)
res = {}
for n in names:
    arm = MT.ARMS[n]; t0 = time.time()
    pol, info = MT.train_arm(arm, ctx)
    if arm.kind not in ("frozen", "filter"):
        torch.save({"kw": pol.kw, "state": pol.state_dict()}, os.path.join(args.out, f"{n}.pt"))
    met = ev.compute(pol)
    res[n] = {"host": os.uname().nodename, "train_s": round(time.time() - t0, 1), "edit_norm": info.get("edit_norm"),
              **{k: v for k, v in met.items() if isinstance(v, (int, float))}}
    f = lambda k: met.get(k, float("nan"))
    print(f"{n:20s} neg succ_clean {f('neg_success_clean'):.3f} bad {f('neg_bad'):.3f} | held succ_clean {f('held_success_clean'):.3f} bad {f('held_bad'):.3f} ({res[n]['train_s']}s)", flush=True)
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
print("=== ROBO TRAIN DONE ===")
