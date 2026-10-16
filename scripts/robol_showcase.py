"""Data for the robol explainer page: example courses and trajectories (base normal / negative, and the same
episodes under each method, common random numbers), plus a 5-seed metrics table. Writes
results/robol/showcase.json.

Arms are retrained on seed 0 at the doses chosen from the k-sweep (results/robol/KSWEEP.md), on trajectory
labels from N_s = 512 courses unless noted.
"""
import glob, json, os, sys
os.environ.setdefault("MKL_CBWR", "AVX2"); os.environ.setdefault("MKL_DYNAMIC", "FALSE"); os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from toypref.robol import data as DT, env as E, methods as MT, seeds as S
from toypref.robol.policy import load

ROBO = "/grogu/datasets/rollout/soumojib/toypref/runs/robol"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
s = 0
bd = os.path.join(ROBO, "base", f"s{s}")
ref = load(os.path.join(bd, "policy.pt"))
lg = torch.load(os.path.join(bd, "log.pt"), map_location="cpu", weights_only=False)
R, Dg, EP = lg["rows"], lg["diag"], lg["episodes"]
train, held = DT.experiment_setups(s)


def approach_mask(ns):
    span = np.zeros(len(R), bool)
    for e, evs in enumerate(EP["events"]):
        for x in evs:
            hi = min(x["onset"] // 8, 17)
            span |= (R.episode == e) & (R.chunk >= hi - 2) & (R.chunk <= hi)
    return span & (R.setup < ns)


traj512 = R.take(R.setup < 512)
appr512 = R.take(approach_mask(512))
# (display name, arm, rows, k or fixed steps, note)
SPEC = [
    ("Pinned SDPO (ours)", "SDPO_pin_b25", traj512, 1 / 32, "trajectory labels, 512 courses, k = 1/32 (~490 steps)"),
    ("Fresh SDPO (ours)", "SDPO_fresh_b25", traj512, 1 / 8, "trajectory labels, 512 courses, k = 1/8 (~1,960 steps)"),
    ("NPO (baseline)", "NPO_b100_neg", traj512, 1 / 32, "trajectory labels, 512 courses, its best k = 1/32"),
    ("KTO (baseline)", "KTO_b100_neg", traj512, 1 / 32, "trajectory labels, 512 courses, its best k = 1/32"),
    ("Pinned SDPO, oracle windows", "SDPO_pin_b25", appr512, 1 / 16, "approach-window labels (oracle), 512 courses, k = 1/16"),
    ("Pinned SDPO, overtrained", "SDPO_pin_b25", traj512, None, "trajectory labels, 512 courses, fixed 5,000 steps (round-1 budget)"),
]
pols = {}
for name, arm, rows, k, note in SPEC:
    steps = 5000 if k is None else max(5, int(round(k * len(rows))))
    pol, _ = MT.train_arm(MT.ARMS[arm], MT.Ctx(ref=ref, seed=s, rows=rows, steps=steps))
    pols[name] = (pol, note, steps)
    print("trained", name, steps, flush=True)

# ---- example episodes: held-out courses, eval noises (CRN)
st = E.concat(train, held)
n_tr = len(train)
cand = np.arange(n_tr, n_tr + 40)
M = 32
idx = np.repeat(cand, M)
noise = torch.cat([torch.randn(M, E.N_CHUNKS, E.ACT_DIM, generator=S.gen(s, S.P_EVAL + int(st.sid[c]))) for c in cand])
b = DT.rollout_chunked(ref, st, idx, noise)
ev = E.events(b.pos, b.contact)
clean = b.success & ~b.negative
rng = np.random.default_rng(7)
pick = {"normal": [], "contact": [], "jerk": []}
used_course = set()
order = rng.permutation(len(idx))
for i in order:
    c = int(idx[i])
    kind = "normal" if clean[i] else ("jerk" if (b.jerky[i] and not b.collided[i]) else ("contact" if b.collided[i] else None))
    if kind is None or len(pick[kind]) >= 4 or c in used_course:
        continue
    pick[kind].append(int(i)); used_course.add(c)
    if all(len(v) >= 4 for v in pick.values()):
        break
sel = pick["normal"] + pick["contact"] + pick["jerk"]


def pack(ro, j, evs):
    P = ro.pos[j]
    v = ~np.isnan(P[:, 0])
    P = P[v]
    return {"pos": np.round(P, 3).tolist(), "contact": [int(t) for t in np.flatnonzero(ro.contact[j])],
            "events": [[t, int(x)] for t, x in evs[j]], "success": bool(ro.success[j]),
            "negative": bool(ro.negative[j]), "collided": bool(ro.collided[j]), "jerky": bool(ro.jerky[j]),
            "steps": int(v.sum() - 1), "chunk_starts": [int(c * E.H) for c in range(E.N_CHUNKS) if ro.ran[j, c]]}


def course(c):
    pres = st.radius[c] > 0
    return {"wps": np.round(st.wps[c], 3).tolist(), "obstacles": [[round(float(x), 3), round(float(y), 3), round(float(r), 3)]
            for (x, y), r in zip(st.centre[c][pres], st.radius[c][pres])]}


examples = []
sub_idx, sub_noise = idx[sel], noise[sel]
method_ro = {name: DT.rollout(p, st, sub_idx, sub_noise) for name, (p, _, _) in pols.items()}
method_ev = {name: E.events(r.pos, r.contact) for name, r in method_ro.items()}
for n, i in enumerate(sel):
    kind = [k for k, v in pick.items() if i in v][0]
    examples.append({"kind": kind, "course": course(int(idx[i])),
                     "base": pack(b, i, ev),
                     "methods": {name: pack(method_ro[name], n, method_ev[name]) for name in pols}})

# ---- 5-seed metrics (held-out courses) from the k-sweep and round-1 cells
def load_cells(tag, arm):
    out = []
    for f in sorted(glob.glob(os.path.join(ROBO, "cells", tag, "s*", f"{arm}.json"))):
        r = json.load(open(f))
        out.append(r)
    return out


def agg(rs, key, base=False):
    x = np.array([(r["b0"] if base else r)[f"held_{key}"] for r in rs])
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0, int(len(x))]


ROWS = [("Base policy (no edit)", "K0.125_ns512_traj", "SDPO_fresh_b25", True, "frozen base, same courses"),
        ("Pinned SDPO (ours)", "K0.03125_ns512_traj", "SDPO_pin_b25", False, "trajectory labels, 512 courses, k = 1/32"),
        ("Pinned SDPO (ours), 64 courses", "K0.0625_ns64_traj", "SDPO_pin_b25", False, "trajectory labels, 64 courses, k = 1/16"),
        ("Fresh SDPO (ours)", "K0.125_ns512_traj", "SDPO_fresh_b25", False, "trajectory labels, 512 courses, k = 1/8"),
        ("NPO (baseline)", "K0.03125_ns512_traj", "NPO_b100_neg", False, "trajectory labels, 512 courses, best k = 1/32"),
        ("KTO (baseline)", "K0.03125_ns512_traj", "KTO_b100_neg", False, "trajectory labels, 512 courses, best k = 1/32"),
        ("Pinned SDPO, oracle windows", "K0.0625_ns512_approach", "SDPO_pin_b25", False, "approach windows, 512 courses, k = 1/16"),
        ("Fresh SDPO, oracle windows", "K0.25_ns512_approach", "SDPO_fresh_b25", False, "approach windows, 512 courses, k = 1/4"),
        ("Pinned SDPO, overtrained", "R2_ns512_traj", "SDPO_pin_b25", False, "trajectory labels, 512 courses, 5,000 steps"),
        ("NPO, round-1 budget", "R2_ns512_traj", "NPO_b100_neg", False, "trajectory labels, 512 courses, 5,000 steps"),
        ("KTO, round-1 budget", "R2_ns512_traj", "KTO_b100_neg", False, "trajectory labels, 512 courses, 5,000 steps")]
table = []
for name, tag, arm, base, note in ROWS:
    rs = load_cells(tag, arm)
    table.append({"name": name, "note": note, **{k: agg(rs, k, base) for k in
                  ("success", "success_clean", "bad", "collision", "jerk", "seg_clean")}})
base_rates = json.load(open(os.path.join(bd, "base_rates.json")))
out = {"examples": examples, "table": table, "method_notes": {n: {"note": v[1], "steps": v[2]} for n, v in pols.items()},
       "env": {"H": E.H, "N_CHUNKS": E.N_CHUNKS, "T_MAX": E.T_MAX, "K": E.K_SEG, "GOAL_TOL": E.GOAL_TOL, "JERK_THR": E.JERK_THR},
       "base_rates_s0": {k: base_rates[k] for k in ("all_success", "all_success_clean", "all_bad", "all_collision", "all_jerk",
                                                     "n_neg_episodes", "n_neg_rows", "share_rows_sym", "events_per_neg_episode")}}
json.dump(out, open("results/robol/showcase.json", "w"))
print("examples", {k: len(v) for k, v in pick.items()}, "table rows", len(table))
for t in table:
    print(t["name"], {k: round(t[k][0], 3) for k in ("success", "success_clean", "bad")})
