"""Run robo2d arms for one seed; one JSON per arm (skip-if-exists).

    python scripts/robo_cell.py --seed 0 --arms all                 # full cell -> cells/full_st5000_m64/s0
    python scripts/robo_cell.py --seed 0 --arms all --smoke 8 --steps 300   # smoke: 8 setups per group

The output tag carries steps and m_eval, and an arm JSON is only skipped when it was produced
from the same base policy file (sha1 of policy.pt), so reruns never mix settings or bases.
"""
import argparse, hashlib, json, os, sys, time
# Reproducible float math: MKL's dynamic threading and per-CPU kernels change the last bits of
# GEMMs between runs / nodes, and closed-loop rollouts amplify those bits into flipped outcomes.
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robo import data as DT, diagnostics as DG, evaluate as EV, labels as LB, methods as MT, targets as TG
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
ap.add_argument("--labels", default="native", help="label condition (toypref.robo.labels); eval is unchanged")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
tag = (f"smoke{args.smoke}_st{args.steps}" if args.smoke else f"full_st{args.steps}") + f"_m{args.m_eval}"
if args.labels != "native":
    tag += f"_L{args.labels}"
od = args.out or os.path.join(ROBO, "cells", tag, f"s{s}")
os.makedirs(od, exist_ok=True)
bd = os.path.join(args.base, f"s{s}")
policy_sha = hashlib.sha1(open(os.path.join(bd, "policy.pt"), "rb").read()).hexdigest()
sep_path = os.path.join(od, "separability.json")


def done(n):
    f = os.path.join(od, f"{n}.json")
    if not os.path.exists(f):
        return False
    r = json.load(open(f))
    if "b0" not in r:
        return False                                 # pre-stress JSON: no in-task base metrics to pair against
    if r.get("base_policy_sha") != policy_sha or r.get("steps") != args.steps or r.get("m_eval") != args.m_eval:
        print(f"[s{s}] {n}: existing JSON is from another base / setting; recomputing", flush=True)
        return False
    return True


def write_json(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


names = list(MT.ARMS) if args.arms == "all" else ([] if args.arms == "none" else args.arms.split(","))
todo = [n for n in names if args.force or not done(n)]
if not todo and os.path.exists(sep_path) and json.load(open(sep_path)).get("base_policy_sha") == policy_sha:
    sys.exit(0)
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
rows, diag = lg["rows"], lg["diag"]
log = DT.LogResult(rows, diag, lg["episodes"], lg["per_setup"])
train, held = DT.experiment_setups(s)
gr = DT.setup_groups(s, log, train)
g_neg, g_noneg = gr["neg"], gr["noneg"]
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
ev = EV.EvalSet(ref, s, train, held, log, M=args.m_eval, groups=groups)   # always the native log
rows, diag, label_info = LB.apply(args.labels, ref, s, train, log)
print(f"[s{s}] labels {label_info}", flush=True)
ctx = MT.Ctx(ref=ref, seed=s, rows=rows, steps=args.steps)
if len(rows) == 0:                                   # nothing labelled: every arm is the base (flagged)
    print(f"[s{s}] no labelled rows under {args.labels}; arms = base", flush=True)
need_cache = len(rows) > 0 and any(MT.ARMS[n].rule == "hop" or MT.ARMS[n].local or MT.ARMS[n].pos == "hop" for n in todo)
cp = os.path.join(od, "hop_cache.pt")
t0 = time.time()


def separability(cache):
    sep = DG.separability(ref, s, rows, diag, cache)
    sep.update({"n_rows": len(rows), "n_bad_chunk_rows": int(diag.bad_chunk.sum()),
                "n_episodes": int(len(np.unique(rows.episode))), "n_setups": int(len(np.unique(rows.setup))),
                "base_policy_sha": policy_sha})
    write_json(sep, sep_path)
    print(f"[s{s}] separability {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in sep.items()})}", flush=True)


def stale(path):
    """True when the cache / separability file exists but was built from another base."""
    if path.endswith(".json"):
        return os.path.exists(path) and json.load(open(path)).get("base_policy_sha") != policy_sha
    return os.path.exists(path) and getattr(torch.load(path, map_location="cpu", weights_only=False), "base_sha", None) != policy_sha


if len(rows) and (need_cache or not os.path.exists(sep_path)):
    # one builder per cell: an O_EXCL lock file; the builder also writes separability.json (atomically),
    # other jobs wait for the cache (stale locks > 3 h are taken over)
    lock = cp + ".lock"
    while not os.path.exists(cp) or stale(cp):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 3 * 3600:
                    os.remove(lock)
            except FileNotFoundError:
                pass                                     # the builder just released it
            time.sleep(30)
            continue
        try:
            if os.path.exists(cp) and not stale(cp):
                break                                    # another job finished between our checks
            cache = TG.build_hop_cache(ref, s, rows)
            cache.base_sha = policy_sha
            tmp = cp + f".{os.getpid()}.tmp"
            torch.save(cache, tmp)
            os.replace(tmp, cp)
            print(f"[s{s}] hop cache for {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
            separability(cache)
        finally:
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
    ctx.cache = torch.load(cp, map_location="cpu", weights_only=False)
if len(rows) and (not os.path.exists(sep_path) or stale(sep_path)):
    separability(ctx.cache)                              # deterministic; atomic write, so a race is harmless
print(f"[s{s}] rows {len(rows)} eval episodes {len(ev.idx)} arms {todo}", flush=True)
# The base's CRN rollout is computed in THIS task, on this node: closed-loop rollouts amplify
# CPU-dependent last bits, so a B0 from another node is not a valid pair. Every arm JSON carries
# the in-task B0 metrics and the analysis pairs against them.
b0_met = {k: v for k, v in ev.compute(ref).items() if isinstance(v, (int, float))}
host = os.uname().nodename
for n in todo:
    arm = MT.ARMS[n]
    t0 = time.time()
    pol, info = (ref, {"no_rows": True}) if len(rows) == 0 else MT.train_arm(arm, ctx)
    met = ev.compute(pol)
    rec = {"seed": s, "arm": n, "kind": arm.kind, "beta": arm.beta, "steps": args.steps, "smoke": args.smoke,
           "m_eval": args.m_eval, "base_policy_sha": policy_sha,
           "labels": label_info, "b0": b0_met, "host": host,
           "train_s": round(time.time() - t0, 1), **{k: v for k, v in info.items() if k != "train_history"},
           **met, "train_history": info.get("train_history")}
    write_json(rec, os.path.join(od, f"{n}.json"))
    f = lambda k: met.get(k, float("nan"))
    print(f"[s{s}] {n:20s} neg bad {f('neg_bad'):.3f} col {f('neg_collision'):.3f} jerk {f('neg_jerk'):.3f} "
          f"succ {f('neg_success'):.3f} homTV {f('neg_hom_tv'):.3f} disp {f('neg_retained_disp'):.3f} | "
          f"noneg bad {f('noneg_bad'):.3f} | held bad {f('held_bad'):.3f} succ {f('held_success'):.3f} "
          f"homTV {f('held_hom_tv'):.3f} | replay {f('logged_neg_still_bad'):.3f} ({rec['train_s']}s)", flush=True)
