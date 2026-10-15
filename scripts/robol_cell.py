"""Run robol arms for one (seed, label condition, N_s, budget rule); one JSON per arm (skip-if-exists).

    python scripts/robol_cell.py --seed 0 --labels traj --ns 128 --budget R1 --arms SDPO_fresh_b25,NPO_b100_neg

labels  traj    every chunk of each negative episode (the ACTUAL method's data)
        sym     symptom-window chunks only (oracle)
        cause   causal-window chunks only (oracle)
        human   human-style window chunks only (oracle)
        approach  THE headline oracle (Soumojit, 2026-10-15): per event, the symptom chunk and the 2
                  chunks before it
        causal_pc per event, the span from the earliest chunk whose resampling removes the event at least
                  PC_MARGIN more often than resampling the placebo chunks 6-7 back, through the symptom
                  chunk (the raw causal test does not beat the placebo; see LONG_TOY_DESIGN.md)
ns      negatives come from the first N_s labelable courses only (nested)
budget  R1  steps = clip(6 x rows, 500, 20000);  R2  5000 steps

Every arm JSON carries the in-task B0 metrics ("b0"): pair against those, never against a B0 from
another task (rollouts differ across nodes).
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robol import data as DT, evaluate as EV, methods as MT
from toypref.robol.policy import load

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--labels", default="traj", choices=["traj", "approach", "sym", "causal_pc", "cause", "human"])
ap.add_argument("--ns", type=int, default=512)
ap.add_argument("--budget", default="R1", choices=["R1", "R2"])
ap.add_argument("--base", default=os.path.join(ROBO, "base"))
ap.add_argument("--out", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
tag = f"{args.budget}_ns{args.ns}_{args.labels}"
od = args.out or os.path.join(ROBO, "cells", tag, f"s{s}")
os.makedirs(od, exist_ok=True)
bd = os.path.join(args.base, f"s{s}")
policy_sha = hashlib.sha1(open(os.path.join(bd, "policy.pt"), "rb").read()).hexdigest()
names = args.arms.split(",")


def done(n):
    f = os.path.join(od, f"{n}.json")
    return (not args.force and os.path.exists(f) and json.load(open(f)).get("base_policy_sha") == policy_sha)


todo = [n for n in names if not done(n)]
if not todo:
    sys.exit(0)
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
log = DT.LogResult(lg["rows"], lg["diag"], lg["episodes"], lg["per_setup"])
train, held = DT.experiment_setups(s)
R, Dg = log.rows, log.diag
m = R.setup < args.ns
PC_MARGIN = 0.2
if args.labels in ("approach", "causal_pc"):
    span = np.zeros(len(R), bool)
    for e, evs in enumerate(log.episodes["events"]):
        for x in evs:
            hi = min(x["onset"] // 8, 17)
            if args.labels == "approach":
                lo = hi - 2
            else:
                ctl = x.get("control", float("nan"))
                beat = [hi - b for b, r in x.get("removal", {}).items()
                        if not np.isnan(ctl) and r >= ctl + PC_MARGIN]
                lo = min(beat + [hi])
            span |= (R.episode == e) & (R.chunk >= lo) & (R.chunk <= hi)
    m &= span
elif args.labels != "traj":
    m &= getattr(Dg, args.labels)
rows = R.take(m)
n_ep = int(len(np.unique(rows.episode)))
steps = int(np.clip(6 * len(rows), 500, 20000)) if args.budget == "R1" else 5000
label_info = {"labels": args.labels, "ns": args.ns, "budget": args.budget, "rows": len(rows), "episodes": n_ep,
              "share_true_cause": float(Dg.cause[m].mean()) if m.any() else float("nan")}
print(f"[s{s}] {tag}: {label_info}, steps {steps}", flush=True)
ev = EV.EvalSet(ref, s, train, held, log)
b0 = {k: v for k, v in ev.compute(ref).items() if isinstance(v, (int, float))}
host = os.uname().nodename
ctx = MT.Ctx(ref=ref, seed=s, rows=rows, steps=steps)


def write_json(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


for n in todo:
    arm = MT.ARMS[n]
    t0 = time.time()
    pol, info = (ref, {}) if arm.kind == "frozen" else MT.train_arm(arm, ctx)
    met = ev.compute(pol)
    rec = {"seed": s, "arm": n, "kind": arm.kind, "beta": arm.beta, "steps": steps, "base_policy_sha": policy_sha,
           "label_info": label_info, "host": host, "train_s": round(time.time() - t0, 1),
           **{k: v for k, v in info.items() if k != "train_history"}, **met, "b0": b0}
    write_json(rec, os.path.join(od, f"{n}.json"))
    f = lambda k: rec.get(k, float("nan")) - b0.get(k, float("nan"))
    print(f"[s{s}] {tag} {n:16s} dclean panel {f('panel_success_clean'):+.3f} noneg {f('noneg_success_clean'):+.3f} "
          f"held {f('held_success_clean'):+.3f} | dseg {f('held_seg_clean'):+.3f} modeTV {met.get('held_mode_tv', 0):.3f} "
          f"speed {met.get('held_speed_ratio', 1):.3f} ({rec['train_s']}s)", flush=True)
