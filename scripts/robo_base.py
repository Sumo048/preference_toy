"""Train one robo2d base policy, log the negatives and report its base rates.

Outputs (under --out/s{seed}/): policy.pt, log.pt (NegRows + privileged NegDiag + replay info),
base_rates.json.
"""
import argparse, json, os, sys, time
# Reproducible float math: MKL's dynamic threading and per-CPU kernels change the last bits of
# GEMMs between runs / nodes, and closed-loop rollouts amplify those bits into flipped outcomes.
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robo import data as DT, demos as D, env as E, evaluate as EV
from toypref.robo.policy import save, load

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robo")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=30_000)
ap.add_argument("--q_col", type=float, default=D.Q_COL)
ap.add_argument("--q_jerk", type=float, default=D.Q_JERK)
ap.add_argument("--n_setups", type=int, default=DT.N_BASE_SETUPS)
ap.add_argument("--out", default=os.path.join(ROBO, "base"))
ap.add_argument("--skip_train", action="store_true", help="reuse policy.pt, redo logging + report")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
od = os.path.join(args.out, f"s{args.seed}")
os.makedirs(od, exist_ok=True)
t0 = time.time()
if not args.skip_train:
    ds = DT.base_demos(args.seed, q_col=args.q_col, q_jerk=args.q_jerk, n_setups=args.n_setups)
    demo_stats = {"n_samples": int(len(ds.obs)), "n_demos": int(ds.demo.max() + 1),
                  "frac_col_injected": float((ds.kind == 1).mean()), "frac_jerk_injected": float((ds.kind == 2).mean())}
    print(f"[s{args.seed}] demos {demo_stats} ({time.time() - t0:.0f}s)", flush=True)
    pol = DT.train_base(args.seed, ds, steps=args.steps)
    save(pol, os.path.join(od, "policy.pt"), q_col=args.q_col, q_jerk=args.q_jerk, steps=args.steps, demo_stats=demo_stats)
    del ds
ck = torch.load(os.path.join(od, "policy.pt"), map_location="cpu", weights_only=False)
ref = load(os.path.join(od, "policy.pt"))
train, held = DT.experiment_setups(args.seed)
log = DT.log_negatives(ref, args.seed, train)
torch.save({"rows": log.rows, "diag": log.diag, "episodes": log.episodes, "per_setup": log.per_setup},
           os.path.join(od, "log.pt"))
ev = EV.EvalSet(ref, args.seed, train, held, log)
met = ev.compute(ref)
b = ev.base
hom_split = {}
for s in np.unique(ev.idx):
    m = (ev.idx == s) & b.success
    codes, cnt = np.unique(ev.base_hom[m], return_counts=True)
    hom_split[int(s)] = {int(c): int(n) for c, n in zip(codes, cnt)}
n_classes = np.array([len(v) for v in hom_split.values()])
maxshare = np.array([max(v.values()) / sum(v.values()) for v in hom_split.values() if v])
ps = log.per_setup
g_neg, g_noneg = DT.setup_groups(args.seed, log, train)
rep = {"seed": args.seed, "q_col": ck.get("q_col"), "q_jerk": ck.get("q_jerk"), "steps": ck.get("steps"),
       "demo_stats": ck.get("demo_stats"),
       "all_success": float(b.success.mean()), "all_collision": float(b.collided.mean()),
       "all_jerk": float(b.jerky.mean()), "all_bad": float(b.negative.mean()),
       "all_timeout": float((~b.success & ~b.collided).mean()),
       "jerk_score_pcts": [float(x) for x in np.percentile(b.jerk, [50, 90, 99])],
       "hom_classes_per_setup_mean": float(n_classes.mean()), "hom_max_share_mean": float(maxshare.mean()),
       "hom_setups_multimodal": float((n_classes >= 2).mean()),
       "n_neg_rows": len(log.rows), "n_neg_episodes": int(len(log.episodes["setup"])),
       "n_neg_bad_chunk_rows": int(log.diag.bad_chunk.sum()),
       "neg_ep_collided": float(np.mean([log.diag.ep_collided[log.rows.episode == e][0] for e in range(len(log.episodes["setup"]))])),
       "neg_ep_jerky": float(np.mean([log.diag.ep_jerky[log.rows.episode == e][0] for e in range(len(log.episodes["setup"]))])),
       "setups_with_neg": int(len(g_neg)), "setups_without_neg": int(len(g_noneg)),
       "setups_wanted_but_none_found": int(((ps["wanted"] > 0) & (ps["n_kept"] == 0)).sum()),
       "setups_short": int((ps["n_kept"] < ps["wanted"]).sum()),
       "logged_rollouts": int(ps["n_logged"].sum()), "log_bad_rate": float(ps["n_neg_seen"].sum() / max(ps["n_logged"].sum(), 1)),
       **met, "time_s": round(time.time() - t0, 1)}
json.dump(rep, open(os.path.join(od, "base_rates.json"), "w"), indent=1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}), flush=True)
