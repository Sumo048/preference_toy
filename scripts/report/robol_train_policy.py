"""Train ONE seed-0 arm on trajectory labels, N_s = 512, and save the policy for the report figures.

    python scripts/report/robol_train_policy.py --name pin
Names: npo kto fresh pin pin5000 sn8 sn64  (same settings as robol_showcase.py / the k-sweep / SEGMENTS cells).
Also writes the held-out (128 courses x 32 CRN noises) outcome rates of the saved policy, to cross-check cells JSONs.
"""
import argparse, json, os, sys, time
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, torch
from toypref.robol import data as DT, methods as MT
from toypref.robol.policy import load, save

ROBO = "/grogu/datasets/rollout/soumojib/toypref/runs/robol"
OUT = "/grogu/datasets/rollout/soumojib/toypref/report_robol/policies"
# name: (arm, k (steps per labelled row) or fixed steps, n_shift)
SPEC = {"npo": ("NPO_b100_neg", 1 / 32, 1, None), "kto": ("KTO_b100_neg", 1 / 32, 1, None),
        "fresh": ("SDPO_fresh_b25", 1 / 8, 1, None), "pin": ("SDPO_pin_b25", 1 / 32, 1, None),
        "pin5000": ("SDPO_pin_b25", None, 1, 5000),
        "sn8": ("SDPO_fresh_b25", 1.0, 8, None), "sn64": ("SDPO_fresh_b25", 2.0, 64, None)}
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s, NS = 0, 512
arm, k, S_, fixed = SPEC[args.name]
bd = os.path.join(ROBO, "base", f"s{s}")
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
R = lg["rows"]
train, held = DT.experiment_setups(s)
rows = R.take(R.setup < NS)
steps = fixed or max(5, int(round(k * len(rows))))
print(args.name, arm, "rows", len(rows), "steps", steps, "n_shift", S_, flush=True)
t0 = time.time()
if S_ == 1:
    pol, info = MT.train_arm(MT.ARMS[arm], MT.Ctx(ref=ref, seed=s, rows=rows, steps=steps))
else:
    labelled = np.unique(R.setup[R.setup < NS])
    refresh = lambda p_, i_: DT.collect_negatives(p_, s, train, labelled, i_, "traj")
    pol, info = MT.train_arm_shift(MT.ARMS[arm], MT.Ctx(ref=ref, seed=s, rows=rows, steps=steps), S_, refresh_rows=refresh)
tt = time.time() - t0
save(pol, os.path.join(OUT, f"{args.name}.pt"), arm=arm, steps=steps, n_shift=S_, k=k)
from toypref.robol import env as E, seeds as SS
st = E.concat(train, held); n_tr = len(train)
sel = np.arange(n_tr, n_tr + len(held)); idx = np.repeat(sel, 32)
noise = torch.cat([torch.randn(32, E.N_CHUNKS, E.ACT_DIM, generator=SS.gen(s, SS.P_EVAL + int(st.sid[c]))) for c in sel])
ro = DT.rollout_chunked(pol, st, idx, noise)
clean = ro.success & ~ro.negative
rates = dict(clean=float(clean.mean()), neg=float(ro.negative.mean()), timeout=float((~ro.success & ~ro.negative).mean()),
             success=float(ro.success.mean()))
json.dump({"name": args.name, "arm": arm, "steps": steps, "n_shift": S_, "train_s": round(tt, 1), "rows": len(rows), **rates,
           "refresh_rows": info.get("refresh_rows")}, open(os.path.join(OUT, f"{args.name}.json"), "w"), indent=1)
print("done", args.name, rates, round(tt), "s", flush=True)
