"""Run robo2d arms for one seed; one JSON per arm (skip-if-exists).

    python scripts/robo_cell.py --seed 0 --arms all                 # full cell
    python scripts/robo_cell.py --seed 0 --arms all --smoke 8 --steps 300   # smoke: 8 setups per group
"""
import argparse, json, os, sys, time
# Reproducible float math: MKL's dynamic threading and per-CPU kernels change the last bits of
# GEMMs between runs / nodes, and closed-loop rollouts amplify those bits into flipped outcomes.
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robo import data as DT, diagnostics as DG, evaluate as EV, methods as MT, targets as TG
from toypref.robo.policy import load

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robo")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arms", default="all")
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--smoke", type=int, default=0, help="restrict to the first N setups of each group")
ap.add_argument("--m_eval", type=int, default=DT.M_EVAL)
ap.add_argument("--base", default=os.path.join(ROBO, "base"))
ap.add_argument("--out", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
tag = f"smoke{args.smoke}_st{args.steps}" if args.smoke else "full"
od = args.out or os.path.join(ROBO, "cells", tag, f"s{s}")
os.makedirs(od, exist_ok=True)
names = list(MT.ARMS) if args.arms == "all" else ([] if args.arms == "none" else args.arms.split(","))
todo = [n for n in names if args.force or not os.path.exists(os.path.join(od, f"{n}.json"))]
if not todo and os.path.exists(os.path.join(od, "separability.json")):
    sys.exit(0)
bd = os.path.join(args.base, f"s{s}")
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
rows, diag = lg["rows"], lg["diag"]
log = DT.LogResult(rows, diag, lg["episodes"], lg["per_setup"])
train, held = DT.experiment_setups(s)
g_neg, g_noneg = DT.setup_groups(s, log, train)
groups = None
if args.smoke:
    keep_setups = g_neg[:args.smoke]
    m = np.isin(rows.setup, keep_setups)
    rows = rows.take(m)
    diag = DT.NegDiag(diag.bad_chunk[m], diag.ep_collided[m], diag.ep_jerky[m])
    em = np.isin(log.episodes["setup"], keep_setups)
    eps = {k: v[em] for k, v in log.episodes.items()}
    log = DT.LogResult(rows, diag, eps, log.per_setup)
    groups = {"neg": keep_setups, "noneg": g_noneg[:args.smoke],
              "held": np.arange(len(train), len(train) + min(args.smoke, len(held)))}
ev = EV.EvalSet(ref, s, train, held, log, M=args.m_eval, groups=groups)
ctx = MT.Ctx(ref=ref, seed=s, rows=rows, steps=args.steps)
need_cache = any(MT.ARMS[n].rule == "hop" or MT.ARMS[n].local or MT.ARMS[n].pos == "hop" for n in todo)
cp = os.path.join(od, "hop_cache.pt")
t0 = time.time()
if need_cache or not os.path.exists(os.path.join(od, "separability.json")):
    # one builder per cell: an O_EXCL lock file; other jobs wait for the cache (stale locks > 3 h are taken over)
    lock = cp + ".lock"
    while not os.path.exists(cp):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            if time.time() - os.path.getmtime(lock) > 3 * 3600:
                os.remove(lock)
            time.sleep(30)
            continue
        try:
            cache = TG.build_hop_cache(ref, s, rows)
            tmp = cp + f".{os.getpid()}.tmp"
            torch.save(cache, tmp)
            os.replace(tmp, cp)
            print(f"[s{s}] hop cache for {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
        finally:
            if os.path.exists(lock):
                os.remove(lock)
    ctx.cache = torch.load(cp, map_location="cpu", weights_only=False)
if not os.path.exists(os.path.join(od, "separability.json")):
    sep = DG.separability(ref, s, rows, diag, ctx.cache)
    sep.update({"n_rows": len(rows), "n_bad_chunk_rows": int(diag.bad_chunk.sum()),
                "n_episodes": int(len(np.unique(rows.episode))), "n_setups": int(len(np.unique(rows.setup)))})
    json.dump(sep, open(os.path.join(od, "separability.json"), "w"), indent=1)
    print(f"[s{s}] separability {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in sep.items()})}", flush=True)
print(f"[s{s}] rows {len(rows)} eval episodes {len(ev.idx)} arms {todo}", flush=True)
for n in todo:
    arm = MT.ARMS[n]
    t0 = time.time()
    pol, info = MT.train_arm(arm, ctx)
    met = ev.compute(pol)
    rec = {"seed": s, "arm": n, "kind": arm.kind, "beta": arm.beta, "steps": args.steps, "smoke": args.smoke,
           "train_s": round(time.time() - t0, 1), **{k: v for k, v in info.items() if k != "train_history"},
           **met, "train_history": info.get("train_history")}
    tmp = os.path.join(od, f".{n}.tmp")
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, os.path.join(od, f"{n}.json"))
    f = lambda k: met.get(k, float("nan"))
    print(f"[s{s}] {n:20s} neg bad {f('neg_bad'):.3f} col {f('neg_collision'):.3f} jerk {f('neg_jerk'):.3f} "
          f"succ {f('neg_success'):.3f} homTV {f('neg_hom_tv'):.3f} disp {f('neg_retained_disp'):.3f} | "
          f"noneg bad {f('noneg_bad'):.3f} | held bad {f('held_bad'):.3f} succ {f('held_success'):.3f} "
          f"homTV {f('held_hom_tv'):.3f} | replay {f('logged_neg_still_bad'):.3f} ({rec['train_s']}s)", flush=True)
