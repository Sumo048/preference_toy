"""Three SDPO variants (Soumojit, 2026-10-16), scaling curves over k (steps per labelled row) and N_s.

    python scripts/robol_sdpo_variants.py --mode iter  --seed 0 --labels traj --ns 128 --arm SDPO_pin_b25 --ks 0.03125,0.0625
    python scripts/robol_sdpo_variants.py --mode shift --seed 0 --labels traj --ns 128 --arm SDPO_pin_b25 --ks 0.125 --shifts 2,4,8
    python scripts/robol_sdpo_variants.py --mode own   --seed 0 --labels traj --ns 128 --ks 0.03125,0.0625

iter   SDPO on the result of SDPO. Round 1 = a normal SDPO run (k x rows steps) from the base. Round n+1 starts
       from round n's policy with pi_ref = round n's policy (winners re-sampled from it), a fresh optimiser and a
       fresh schedule, k x rows steps again. Same logged negatives every round. Evaluated after rounds 1, 2, 3.
shift  One continuous SDPO run of k x rows steps (one optimiser, one cosine schedule) whose pi_ref is replaced by
       the current policy at S-1 equally spaced points (S segments); winners re-sampled at each shift.
own    Pinned SDPO with the winner at its OWN noise: loser at the negative's logged noise; each of K winners is a
       base sample at the same observation, scored at the noise that generated it; r_w = mean over the K winners.
       K = 1 and K = 8 (SDPO_own_b25_K1 / _K8).

Every result is paired with the ORIGINAL base's rollout computed in this task ("b0"). Outputs:
cells_v/<mode>/<labels>_ns<N>/s<seed>/<name>.json
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robol import data as DT, evaluate as EV, methods as MT
from toypref.robol.policy import load, frozen

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["iter", "shift", "own"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--labels", default="traj", choices=["traj", "approach"])
ap.add_argument("--ns", type=int, default=512)
ap.add_argument("--arm", default="SDPO_pin_b25")
ap.add_argument("--ks", required=True)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--shifts", default="2,4,8")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
bd = os.path.join(ROBO, "base", f"s{s}")
od = os.path.join(ROBO, "cells_v", args.mode, f"{args.labels}_ns{args.ns}", f"s{s}")
os.makedirs(od, exist_ok=True)
policy_sha = hashlib.sha1(open(os.path.join(bd, "policy.pt"), "rb").read()).hexdigest()
ks = [float(x) for x in args.ks.split(",")]
shifts = [int(x) for x in args.shifts.split(",")]
arms = ["SDPO_own_b25_K1", "SDPO_own_b25_K8"] if args.mode == "own" else [args.arm]


def kname(k):
    return f"{k:g}"


def names():
    for arm in arms:
        for k in ks:
            if args.mode == "iter":
                for n in range(1, args.rounds + 1):
                    yield f"{arm}_k{kname(k)}_r{n}"
            elif args.mode == "shift":
                for S in shifts:
                    yield f"{arm}_k{kname(k)}_S{S}"
            else:
                yield f"{arm}_k{kname(k)}"


def done(n):
    f = os.path.join(od, f"{n}.json")
    return os.path.exists(f) and json.load(open(f)).get("base_policy_sha") == policy_sha


if all(done(n) for n in names()):
    sys.exit(0)
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
log = DT.LogResult(lg["rows"], lg["diag"], lg["episodes"], lg["per_setup"])
train, held = DT.experiment_setups(s)
R = log.rows
m = R.setup < args.ns
if args.labels == "approach":
    span = np.zeros(len(R), bool)
    for e, evs in enumerate(log.episodes["events"]):
        for x in evs:
            hi = min(x["onset"] // 8, 17)
            span |= (R.episode == e) & (R.chunk >= hi - 2) & (R.chunk <= hi)
    m &= span
rows = R.take(m)
ev = EV.EvalSet(ref, s, train, held, log)
b0 = {k: v for k, v in ev.compute(ref).items() if isinstance(v, (int, float))}
host = os.uname().nodename
print(f"[s{s}] {args.mode} {args.labels} ns{args.ns}: rows {len(rows)}", flush=True)


def write(name, pol, info, extra):
    met = ev.compute(pol)
    rec = {"seed": s, "name": name, "mode": args.mode, "labels": args.labels, "ns": args.ns, "rows": len(rows),
           "base_policy_sha": policy_sha, "host": host, **extra,
           **{k: v for k, v in info.items() if k != "train_history"}, **met, "b0": b0}
    tmp = os.path.join(od, f"{name}.json.{os.getpid()}.tmp")
    json.dump(rec, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, os.path.join(od, f"{name}.json"))
    d = lambda k: met.get(k, float("nan")) - b0.get(k, float("nan"))
    print(f"[s{s}] {name:34s} dclean held {d('held_success_clean'):+.3f} dbad {d('held_bad'):+.3f} disp {met.get('held_retained_disp', 0):.3f}", flush=True)


for arm in arms:
    A = MT.ARMS[arm]
    for k in ks:
        steps = max(5, int(round(k * len(rows))))
        if args.mode == "iter":
            cur = ref
            for n in range(1, args.rounds + 1):
                nm = f"{arm}_k{kname(k)}_r{n}"
                t0 = time.time()
                pol, info = MT.train_arm(A, MT.Ctx(ref=cur, seed=s, rows=rows, steps=steps))
                write(nm, pol, info, {"arm": arm, "k": k, "round": n, "steps_per_round": steps, "total_steps": n * steps,
                                      "train_s": round(time.time() - t0, 1)})
                cur = frozen(pol)
        elif args.mode == "shift":
            for S in shifts:
                nm = f"{arm}_k{kname(k)}_S{S}"
                if done(nm):
                    continue
                t0 = time.time()
                pol, info = MT.train_arm_shift(A, MT.Ctx(ref=ref, seed=s, rows=rows, steps=steps), S)
                write(nm, pol, info, {"arm": arm, "k": k, "n_shift": S, "total_steps": steps, "train_s": round(time.time() - t0, 1)})
        else:
            nm = f"{arm}_k{kname(k)}"
            if done(nm):
                continue
            t0 = time.time()
            pol, info = MT.train_arm(A, MT.Ctx(ref=ref, seed=s, rows=rows, steps=steps))
            write(nm, pol, info, {"arm": arm, "k": k, "n_win": A.n_win, "total_steps": steps, "train_s": round(time.time() - t0, 1)})
