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
ap.add_argument("--budget", default="R1", help="R1: clip(6 x rows, 500, 20000) | R2: 5000 | K<k>: round(k x rows), no clip (>= 20)")
ap.add_argument("--variant", default="", help="environment variant: bases in base_<v>, cells in cells_<v>")
ap.add_argument("--noise", default="none", help="label noise: none | flip<f> (a share f of labelled episodes replaced by a "
                "clean pool episode of the same course) | recall<r> (only a share r of negative episodes labelled)")
ap.add_argument("--base", default="")
ap.add_argument("--out", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
tag = f"{args.budget}_ns{args.ns}_{args.labels}" + ("" if args.noise == "none" else f"_{args.noise}")
vs = f"_{args.variant}" if args.variant else ""
args.base = args.base or os.path.join(ROBO, "base" + vs)
od = args.out or os.path.join(ROBO, "cells" + vs, tag, f"s{s}")
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
P_NOISE = 95_000
noise_info = {}
rows = R.take(m)
if args.noise != "none":
    import re
    from toypref.robol import seeds as SS
    n_glob = int(R.episode.max()) + 1
    u = SS.nprng(s, P_NOISE).random(n_glob)
    if (mm := re.fullmatch(r"recall([\d.]+)", args.noise)):
        m &= (u < float(mm.group(1)))[R.episode]
        rows = R.take(m)
        noise_info = {"kept_episodes": int(len(np.unique(rows.episode)))}
    elif (mm := re.fullmatch(r"flip([\d.]+)", args.noise)):
        f = float(mm.group(1))
        flip_ep = np.flatnonzero(u < f)
        m &= ~np.isin(R.episode, flip_ep)
        keep = R.take(m)
        pool = torch.load(os.path.join(bd, "pool.pt"), map_location="cpu", weights_only=False)
        g = SS.nprng(s, P_NOISE + 1)
        used = {}
        add = {k: [] for k in ("obs", "xi", "a", "uid", "setup", "episode", "chunk")}
        n_flip = 0
        ep_setup = {int(e): int(R.setup[R.episode == e][0]) for e in np.unique(R.episode[R.setup < args.ns])}
        for e, st_ in ep_setup.items():
            if u[e] >= f:
                continue
            cand = [pe for pe in np.unique(pool.episode[pool.setup == st_]) if not pool.negative[pool.episode == pe].any()]
            cand = [pe for pe in cand if pe not in used.get(st_, set())]
            if not cand:
                continue
            pe = int(cand[0]); used.setdefault(st_, set()).add(pe)
            idx = np.flatnonzero(pool.episode == pe)
            ch = pool.chunk[idx]
            if args.labels != "traj":                 # a human marks a window on the (wrongly) labelled episode
                w = 3 if args.labels == "approach" else 1
                lo = int(g.integers(0, max(1, len(idx) - w + 1)))
                idx = idx[(ch >= ch.min() + lo) & (ch < ch.min() + lo + w)]
            for i in idx:
                add["obs"].append(pool.obs[i]); add["xi"].append(pool.xi[i]); add["a"].append(pool.a[i])
                add["uid"].append(int(train.sid[st_]) * 100_000 + (200 + int(pe) % 100) * 100 + int(pool.chunk[i]))
                add["setup"].append(st_); add["episode"].append(1_000_000 + int(pe)); add["chunk"].append(int(pool.chunk[i]))
            n_flip += 1
        if add["uid"]:
            rows = DT.NegRows(torch.cat([keep.obs, torch.stack(add["obs"])]), torch.cat([keep.xi, torch.stack(add["xi"])]),
                              torch.cat([keep.a, torch.stack(add["a"])]), np.concatenate([keep.uid, add["uid"]]),
                              np.concatenate([keep.setup, add["setup"]]), np.concatenate([keep.episode, add["episode"]]),
                              np.concatenate([keep.chunk, add["chunk"]]))
        else:
            rows = keep
        assert len(set(rows.uid.tolist())) == len(rows.uid), "row uids must be unique"
        noise_info = {"flipped_episodes": n_flip, "flipped_rows": len(add["uid"])}
    else:
        raise ValueError(args.noise)
n_ep = int(len(np.unique(rows.episode)))
if args.budget == "R1":
    steps = int(np.clip(6 * len(rows), 500, 20000))
elif args.budget == "R2":
    steps = 5000
elif args.budget.startswith("K"):
    steps = max(20, int(round(float(args.budget[1:]) * len(rows))))
else:
    raise ValueError(args.budget)
label_info = {"labels": args.labels, "ns": args.ns, "budget": args.budget, "noise": args.noise, "variant": args.variant,
              **noise_info, "rows": len(rows), "episodes": n_ep,
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
