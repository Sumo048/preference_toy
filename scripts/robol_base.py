"""Train one robol base policy, log the negatives (+ window oracles), collect the unlabelled pool and
report base rates and the design's validation checks (docs/LONG_TOY_DESIGN.md).

    python scripts/robol_base.py --seed 0 [--q_col .. --q_jerk .. --q_mom .. --q_route .. --rho ..] [--out DIR]

Outputs under --out/s{seed}/: policy.pt, log.pt (NegRows + privileged NegDiag + episodes), pool.pt,
base_rates.json.
"""
import argparse, json, os, sys, time
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robol import data as DT, demos as D, env as E, evaluate as EV
from toypref.robol.policy import save, load

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=30_000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--q_col", type=float, default=D.Q_COL)
ap.add_argument("--q_jerk", type=float, default=D.Q_JERK)
ap.add_argument("--q_mom", type=float, default=D.Q_MOM)
ap.add_argument("--q_route", type=float, default=D.Q_ROUTE)
ap.add_argument("--rho", type=float, default=D.RHO)
ap.add_argument("--clr", type=float, nargs=2, default=list(D.CLR))
ap.add_argument("--n_setups", type=int, default=DT.N_BASE_SETUPS)
ap.add_argument("--out", default=os.path.join(ROBO, "base"))
ap.add_argument("--skip_train", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
D.CLR = tuple(args.clr)
od = os.path.join(args.out, f"s{args.seed}")
os.makedirs(od, exist_ok=True)
t0 = time.time()
inj = dict(q_col=args.q_col, q_jerk=args.q_jerk, q_mom=args.q_mom, q_route=args.q_route, rho=args.rho)
print(f"[s{args.seed}] injection {inj} clearance {D.CLR}", flush=True)
if not args.skip_train:
    ds = DT.base_demos(args.seed, n_setups=args.n_setups, **inj)
    demo_stats = {"n_samples": int(len(ds.obs)), "n_demos": int(ds.demo.max() + 1),
                  **{f"frac_kind{k}": float((ds.kind == k).mean()) for k in range(4)},
                  "frac_route_demos": float(np.mean([ds.route[ds.demo == d][0] for d in np.unique(ds.demo)]))}
    print(f"[s{args.seed}] demos {demo_stats} ({time.time() - t0:.0f}s)", flush=True)
    pol = DT.train_base(args.seed, ds, steps=args.steps, lr=args.lr)
    save(pol, os.path.join(od, "policy.pt"), steps=args.steps, demo_stats=demo_stats, clr=list(D.CLR), **inj)
    del ds
ck = torch.load(os.path.join(od, "policy.pt"), map_location="cpu", weights_only=False)
ref = load(os.path.join(od, "policy.pt"))
train, held = DT.experiment_setups(args.seed)
print(f"[s{args.seed}] courses ready ({time.time() - t0:.0f}s)", flush=True)
log = DT.log_negatives(ref, args.seed, train)
torch.save({"rows": log.rows, "diag": log.diag, "episodes": log.episodes, "per_setup": log.per_setup}, os.path.join(od, "log.pt"))
print(f"[s{args.seed}] negatives {len(log.episodes['setup'])} episodes, {len(log.rows)} rows ({time.time() - t0:.0f}s)", flush=True)
pool = DT.unlabelled_pool(ref, args.seed, train)
torch.save(pool, os.path.join(od, "pool.pt"))
ev = EV.EvalSet(ref, args.seed, train, held, log)
met = ev.compute(ref)
b = ev.base
# validation checks
rp = DT.rollout_chunked(ref, train, log.episodes["setup"], torch.from_numpy(log.episodes["noise"]))
multimodal = []
for s in np.unique(ev.idx):
    ms = ev.idx == s
    for k in range(ev.st.K):
        x = ev.base_hom[ms, k]
        x = x[x >= 0]
        if len(x) >= EV.MIN_TV:
            multimodal.append(len(np.unique(x)) >= 2)
ps = log.per_setup
dg = log.diag
ev_types = [x["type"] for evs in log.episodes["events"] for x in evs]
rep = {"seed": args.seed, **{k: ck.get(k) for k in ("q_col", "q_jerk", "q_mom", "q_route", "rho", "steps", "demo_stats")},
       "all_success": float(b.success.mean()), "all_success_clean": float((b.success & ~b.negative).mean()),
       "all_bad": float(b.negative.mean()), "all_collision": float(b.collided.mean()), "all_jerk": float(b.jerky.mean()),
       "all_events_per_ep": float(np.mean([len(x) for x in ev.base_ev])),
       "all_seg_event_rate": float(1 - ev.base_seg.mean()),
       "n_neg_episodes": int(len(log.episodes["setup"])), "n_neg_rows": len(log.rows),
       "events_per_neg_episode": float(len(ev_types) / max(len(log.episodes["setup"]), 1)),
       "event_types": {t: ev_types.count(t) for t in set(ev_types)},
       "share_rows_sym": float(dg.sym.mean()), "share_rows_cause": float(dg.cause.mean()), "share_rows_human": float(dg.human.mean()),
       "causal_fallback": ps["causal_fallback"], "causal_removal_rate": ps["causal_removal_rate"], "lag_hist": ps["lag_hist"],
       "removal_by_back": ps["removal_by_back"], "control_removal_rate": ps["control_removal_rate"],
       "segments_multimodal": float(np.mean(multimodal)),
       "replay_reproduces": float(rp.negative.mean()),
       "log_bad_rate": float(ps["n_neg_seen"].sum() / max(ps["n_logged"].sum(), 1)),
       "labelable_short": int((ps["n_kept"][:DT.N_LABELABLE] < ps["wanted"][:DT.N_LABELABLE]).sum()),
       "labelable_none": int((ps["n_kept"][:DT.N_LABELABLE] == 0).sum()),
       "pool_rows": int(len(pool.setup)), "pool_neg_share": float(pool.negative.mean()),
       **met, "time_s": round(time.time() - t0, 1)}
checks = {"V1_clean_success_0.75_0.85": 0.75 <= rep["all_success_clean"] <= 0.85,
          "V2_event_rows_le_0.15": rep["share_rows_sym"] <= 0.15,
          "V3_causal_removal_ge_0.5": (rep["causal_removal_rate"] or 0) >= 0.5,
          "V4_multimodal_ge_0.5": rep["segments_multimodal"] >= 0.5,
          "V5_replay": rep["replay_reproduces"] == 1.0}
rep["checks"] = checks
json.dump(rep, open(os.path.join(od, "base_rates.json"), "w"), indent=1, default=float)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}, default=float), flush=True)
