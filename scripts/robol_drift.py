"""Drift experiments A and D (Soumojit, 2026-10-18): does pinned SDPO move the policy farther from the base than
fresh SDPO, in which sense, where, and which half of the loss does it?

    python scripts/robol_drift.py --seed 0 --arm SDPO_pin_b25 --labels traj --ns 512

For each fixed step count in STEPS (separate runs, cosine schedule over each run) it trains the arm and measures,
without saving the policy:
  D1  weight distance ||theta - theta_0||
  D2  pathwise action change: RMS of a_theta(xi, o) - a_0(xi, o) at the SAME noise
  D3  distribution distance at a state: energy distance between M = 64 actions of the edited and the base policy
      (shared noises), the mean-action shift, the spread ratio, and a KL surrogate
      KL_hat = E_{a ~ pi_0, xi, t}[l_theta(a) - l_0(a)] (flow-matching loss gap on base samples; >= 0 in expectation
      when theta is worse at the base's own samples)
  at four state sets (256 states each, fixed per seed):
      neg     observations of the labelled negative rows used for training
      lab     unlabelled pool states on the LABELLED courses (setup < N_s)
      unl     pool states on training courses that never carry negatives (setup >= 512)
      held    base-rollout states on held-out courses
  D4  closed-loop: route-mode TV, speed ratio (from the standard eval) and the energy distance between the visited
      positions of edited and base rollouts on 32 held-out courses x 32 CRN episodes
  gate dynamics: on a fixed probe batch of 256 training pairs, every steps/20: mean gate sigma(beta(r_l - r_w)),
      r_w, r_l, margin r_w - r_l (pairs built as the arm builds them)
  plus G and the usual metrics (held clean success etc., paired with the in-task B0).

Outputs: <ROBO>/drift/<labels>_ns<N>/s<seed>/<arm>_st<steps>.json
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.flow import FlowPolicy
from toypref.robol import data as DT, env as E, evaluate as EV, methods as MT, seeds as S, targets as TG

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
STEPS = [50, 100, 200, 500, 1000, 2000, 5000]
P_DRIFT = 300_000
M_NOISE = 64
N_STATES = 256
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--labels", default="traj", choices=["traj", "approach"])
ap.add_argument("--ns", type=int, default=512)
ap.add_argument("--steps", default=",".join(map(str, STEPS)))
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = args.seed
bd = os.path.join(ROBO, "base", f"s{s}")
od = os.path.join(ROBO, "drift", f"{args.labels}_ns{args.ns}", f"s{s}")
os.makedirs(od, exist_ok=True)
sha = hashlib.sha1(open(os.path.join(bd, "policy.pt"), "rb").read()).hexdigest()
steps_list = [int(x) for x in args.steps.split(",")]
todo = [st for st in steps_list if not (os.path.exists(os.path.join(od, f"{args.arm}_st{st}.json")) and
                                        json.load(open(os.path.join(od, f"{args.arm}_st{st}.json"))).get("base_policy_sha") == sha)]
if not todo:
    sys.exit(0)

from toypref.robol.policy import load
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
log = DT.LogResult(lg["rows"], lg["diag"], lg["episodes"], lg["per_setup"])
pool = torch.load(os.path.join(bd, "pool.pt"), map_location="cpu", weights_only=False)
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

# ---- fixed state sets and noises (per seed, shared by every arm and step count)
rng = S.nprng(s, P_DRIFT)
pick = lambda n, k: rng.choice(n, size=min(k, n), replace=False)
n_tr = len(train)
held_mask = (ev.idx >= n_tr)[:, None] & ev.base.ran
held_obs = torch.from_numpy(ev.base.obs[held_mask])
sets = {
    "neg": rows.obs[torch.as_tensor(pick(len(rows), N_STATES))],
    "lab": pool.obs[torch.as_tensor(np.flatnonzero(pool.setup < args.ns)[pick(int((pool.setup < args.ns).sum()), N_STATES)])],
    "unl": pool.obs[torch.as_tensor(np.flatnonzero(pool.setup >= DT.N_LABELABLE)[pick(int((pool.setup >= DT.N_LABELABLE).sum()), N_STATES)])],
    "held": held_obs[torch.as_tensor(pick(len(held_obs), N_STATES))],
}
gN = S.gen(s, P_DRIFT + 1)
noise = {k: torch.randn(len(v), M_NOISE, E.ACT_DIM, generator=gN) for k, v in sets.items()}
tKL = {k: FlowPolicy.sample_time(len(v) * M_NOISE, gN) for k, v in sets.items()}
xiKL = {k: torch.randn(len(v) * M_NOISE, E.ACT_DIM, generator=gN) for k, v in sets.items()}


@torch.no_grad()
def decode(pol, k):
    o = sets[k].repeat_interleave(M_NOISE, 0)
    return pol.sample(noise[k].reshape(-1, E.ACT_DIM), o).reshape(len(sets[k]), M_NOISE, -1)


base_dec = {k: decode(ref, k) for k in sets}


def energy(X, Y):
    """per-state energy distance between sample sets X, Y (n, M, d)."""
    dxy = torch.cdist(X, Y).mean((1, 2))
    dxx = torch.cdist(X, X).sum((1, 2)) / (X.shape[1] * (X.shape[1] - 1))
    dyy = torch.cdist(Y, Y).sum((1, 2)) / (Y.shape[1] * (Y.shape[1] - 1))
    return 2 * dxy - dxx - dyy


@torch.no_grad()
def drift_metrics(pol):
    out = {}
    for k in sets:
        X, Y = decode(pol, k), base_dec[k]
        out[f"{k}_path"] = float((X - Y).pow(2).mean(-1).sqrt().mean())
        out[f"{k}_energy"] = float(energy(X, Y).mean())
        out[f"{k}_meanshift"] = float((X.mean(1) - Y.mean(1)).norm(dim=-1).mean())
        out[f"{k}_spread_ratio"] = float((X.std(1).norm(dim=-1) / Y.std(1).norm(dim=-1).clamp_min(1e-8)).mean())
        a0 = Y.reshape(-1, Y.shape[-1])
        o = sets[k].repeat_interleave(M_NOISE, 0)
        out[f"{k}_kl_hat"] = float((pol.flow_loss(a0, xiKL[k], tKL[k], o) - ref.flow_loss(a0, xiKL[k], tKL[k], o)).mean())
    return out


# closed-loop visited states on 32 held-out courses x 32 CRN episodes
hc = np.arange(n_tr, n_tr + 32)
hsel = np.flatnonzero(np.isin(ev.idx, hc))
hsel = np.concatenate([hsel[ev.idx[hsel] == c][:32] for c in hc])
base_pos = ev.base.pos[hsel]


@torch.no_grad()
def visit_metrics(pol):
    ro = DT.rollout(pol, ev.st, ev.idx[hsel], ev.noise[hsel])
    es = []
    for c in hc:
        mc = ev.idx[hsel] == c
        P = torch.from_numpy(np.nan_to_num(ro.pos[mc][~np.isnan(ro.pos[mc][..., 0])])).float()
        Q = torch.from_numpy(base_pos[mc][~np.isnan(base_pos[mc][..., 0])]).float()
        P, Q = P[torch.randperm(len(P), generator=torch.Generator().manual_seed(int(c)))[:512]], Q[torch.randperm(len(Q), generator=torch.Generator().manual_seed(int(c) + 1))[:512]]
        es.append(float(energy(P[None], Q[None])[0]))
    return {"visit_energy": float(np.mean(es))}


# ---- gate probe: fixed 256 training pairs, built as the arm builds them
A = MT.ARMS[args.arm]
gp = S.gen(s, P_DRIFT + 2)
pi = torch.randint(0, len(rows), (256,), generator=gp)
P = TG.fresh_decodes(ref, s, rows, TG.N_SDPO_POS, S.ROW_SDPO)
p_aw, p_al, p_o = P[pi, 0], rows.a[pi], rows.obs[pi]
p_t = FlowPolicy.sample_time(256, gp)
if A.noise == "pin":
    p_nw = p_nl = rows.xi[pi]
else:
    p_nw, p_nl = torch.randn(256, E.ACT_DIM, generator=gp), torch.randn(256, E.ACT_DIM, generator=gp)
with torch.no_grad():
    p_lrw, p_lrl = ref.flow_loss(p_aw, p_nw, p_t, p_o), ref.flow_loss(p_al, p_nl, p_t, p_o)


@torch.no_grad()
def probe(pol):
    r_w = p_lrw - pol.flow_loss(p_aw, p_nw, p_t, p_o)
    r_l = p_lrl - pol.flow_loss(p_al, p_nl, p_t, p_o)
    return {"gate": float(torch.sigmoid(A.beta * (r_l - r_w)).mean()), "r_w": float(r_w.mean()), "r_l": float(r_l.mean()),
            "margin": float((r_w - r_l).mean())}


print(f"[s{s}] drift {args.arm} {args.labels} ns{args.ns}: rows {len(rows)}, todo {todo}", flush=True)
for st in todo:
    t0 = time.time()
    pol, info = MT.train_arm(A, MT.Ctx(ref=ref, seed=s, rows=rows, steps=st), probe=probe, log_every=max(1, st // 20))
    met = ev.compute(pol)
    rec = {"seed": s, "arm": args.arm, "labels": args.labels, "ns": args.ns, "steps": st, "rows": len(rows),
           "base_policy_sha": sha, "host": os.uname().nodename, "edit_norm": info.get("edit_norm"),
           "gate_history": info.get("train_history"), **drift_metrics(pol), **visit_metrics(pol),
           **{k: v for k, v in met.items() if isinstance(v, (int, float))}, "b0": b0, "train_s": round(time.time() - t0, 1)}
    tmp = os.path.join(od, f"{args.arm}_st{st}.json.{os.getpid()}.tmp")
    json.dump(rec, open(tmp, "w"), default=float)
    os.replace(tmp, os.path.join(od, f"{args.arm}_st{st}.json"))
    print(f"[s{s}] {args.arm} st{st}: G {100 * (rec['held_success_clean'] - b0['held_success_clean']):+.1f} edit {rec['edit_norm']:.2f} "
          f"path(held) {rec['held_path']:.3f} energy(held) {rec['held_energy']:.4f} kl(held) {rec['held_kl_hat']:.4f} "
          f"visit {rec['visit_energy']:.4f} gate_end {rec['gate_history'][-1].get('gate', float('nan')):.3f}", flush=True)
