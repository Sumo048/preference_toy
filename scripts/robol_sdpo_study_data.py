"""Collect every number the SDPO-study page shows into results/robol/sdpo_study.json.

Sources (all under ROBO): cells/K*_ns*_{traj,approach} (k-sweep), cells_v/{iter,shift,shiftneg,own} (variants and
segment scaling), drift/ (experiments A and D). G = held-out clean success minus the in-task B0, in points.
"""
import glob, json, os, re
import numpy as np, pandas as pd

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")


def G(r, key="held_success_clean"):
    return 100 * (r[key] - r["b0"][key])


def ms(x):
    x = np.asarray(list(x), float)
    return [round(float(x.mean()), 2), round(float(x.std(ddof=1) / np.sqrt(len(x))), 2) if len(x) > 1 else 0.0, int(len(x))]


out = {}
# ---- k-sweep
rows = []
for f in glob.glob(f"{ROBO}/cells/K*_ns*_*/s*/*.json"):
    m = re.fullmatch(r"K([\d.]+)_ns(\d+)_(traj|approach)", f.split("/")[-3])
    if not m:
        continue
    r = json.load(open(f))
    rows.append(dict(arm=r["arm"], k=float(m.group(1)), ns=int(m.group(2)), lab=m.group(3), G=G(r), steps=r["steps"],
                     clean=r["held_success_clean"], bad=r["held_bad"], succ=r["held_success"],
                     b_clean=r["b0"]["held_success_clean"], b_bad=r["b0"]["held_bad"], b_succ=r["b0"]["held_success"]))
ks = pd.DataFrame(rows)
out["base"] = {"clean": round(ks.b_clean.mean(), 4), "bad": round(ks.b_bad.mean(), 4), "success": round(ks.b_succ.mean(), 4)}
out["ksweep"] = {}
for (arm, lab, ns), d in ks.groupby(["arm", "lab", "ns"]):
    out["ksweep"].setdefault(arm, {}).setdefault(lab, {})[str(ns)] = {
        f"{k:g}": ms(dk.G) + [int(dk.steps.median())] for k, dk in d.groupby("k")}

# ---- variants: best over k per (variant, arm, labels, ns)
vr = []
for f in glob.glob(f"{ROBO}/cells_v/*/*_ns*/s*/*.json"):
    r = json.load(open(f))
    mode = r["mode"]
    if mode == "iter":
        cfg = f"round {r['round']}"
    elif mode in ("shift", "shiftneg"):
        cfg = f"S={r['n_shift']}"
    else:
        cfg = f"K={r['n_win']}"
    arm = "SDPO_own_b25" if mode == "own" else r["arm"]
    vr.append(dict(mode=mode, arm=arm, cfg=cfg, lab=r["labels"], ns=r["ns"], k=r["k"], G=G(r), steps=r["total_steps"],
                   S=r.get("n_shift", 1), clean=r["held_success_clean"], bad=r["held_bad"], succ=r["held_success"],
                   speed=r.get("held_speed_ratio", np.nan), jerk=r["held_jerk"], col=r["held_collision"]))
vd = pd.DataFrame(vr)
best = {}
for (mode, arm, cfg, lab, ns), d in vd.groupby(["mode", "arm", "cfg", "lab", "ns"]):
    mk = d.groupby("k").G.mean()
    k = mk.idxmax()
    best.setdefault(mode, {}).setdefault(arm, {}).setdefault(lab, {}).setdefault(str(ns), {})[cfg] = ms(d[d.k == k].G) + [k]
# plain best from the k-sweep, for reference
for (arm, lab, ns), d in ks.groupby(["arm", "lab", "ns"]):
    mk = d.groupby("k").G.mean()
    k = mk.idxmax()
    best.setdefault("plain", {}).setdefault(arm, {}).setdefault(lab, {}).setdefault(str(ns), {})["plain"] = ms(d[d.k == k].G) + [k]
out["variants_best"] = best

# ---- segment scaling (shiftneg + plain S=1) at fixed k_seg
KSEG = [1 / 32, 1 / 16, 1 / 8, 1 / 4]
seg = vd[vd["mode"] == "shiftneg"].copy()
seg["kseg"] = seg.k / seg.S
pl = ks.rename(columns={"lab": "lab"}).copy()
pl["S"], pl["kseg"] = 1, pl.k
segall = pd.concat([seg[["arm", "lab", "ns", "S", "kseg", "G", "clean", "bad", "succ"]], pl[["arm", "lab", "ns", "S", "kseg", "G", "clean", "bad", "succ"]]])
segall = segall[segall.kseg.apply(lambda x: any(abs(x - q) < 1e-9 for q in KSEG)) & segall.ns.isin([64, 128, 512]) & segall.arm.isin(["SDPO_pin_b25", "SDPO_fresh_b25"])]
segall["kseg"] = segall.kseg.apply(lambda x: min(KSEG, key=lambda q: abs(q - x)))
out["segments"] = {}
for (arm, lab, ns, kq), d in segall.groupby(["arm", "lab", "ns", "kseg"]):
    out["segments"].setdefault(arm, {}).setdefault(lab, {}).setdefault(str(ns), {})[f"1/{round(1 / kq)}"] = {
        str(S): ms(dS.G) for S, dS in d.groupby("S")}
# absolute rates at the headline segment configurations
def absolute(arm, lab, ns, k, S):
    d = vd[(vd["mode"] == "shiftneg") & (vd.arm == arm) & (vd.lab == lab) & (vd.ns == ns) & (np.isclose(vd.k, k)) & (vd.S == S)]
    return {"n": int(len(d)), "clean": round(d.clean.mean(), 4), "bad": round(d.bad.mean(), 4), "success": round(d.succ.mean(), 4),
            "speed": round(d.speed.mean(), 3), "jerk": round(d.jerk.mean(), 4), "collision": round(d.col.mean(), 4)}
out["segment_cases"] = {
    "fresh_traj512_S8": absolute("SDPO_fresh_b25", "traj", 512, 1.0, 8),
    "fresh_traj512_S16": absolute("SDPO_fresh_b25", "traj", 512, 4.0, 16),
    "fresh_traj512_S32": absolute("SDPO_fresh_b25", "traj", 512, 4.0, 32),
    "fresh_traj512_S64": absolute("SDPO_fresh_b25", "traj", 512, 2.0, 64),
    "pin_appr512_S2": absolute("SDPO_pin_b25", "approach", 512, 0.0625, 2),
    "pin_appr512_S32": absolute("SDPO_pin_b25", "approach", 512, 1.0, 32),
    "pin_traj64_S4": absolute("SDPO_pin_b25", "traj", 64, 0.125, 4),
    "pin_traj64_S32": absolute("SDPO_pin_b25", "traj", 64, 1.0, 32),
}

# ---- drift A + D
dr = []
for f in glob.glob(f"{ROBO}/drift/*_ns*/s*/*.json"):
    r = json.load(open(f))
    dr.append(dict(arm=r["arm"], lab=r["labels"], ns=r["ns"], steps=r["steps"], G=G(r), edit=r["edit_norm"],
                   path=r["held_path"], energy=r["held_energy"], kl=r["held_kl_hat"], visit=r["visit_energy"],
                   neg_energy=r["neg_energy"], lab_energy=r["lab_energy"], unl_energy=r["unl_energy"],
                   gate=[(h["step"], h.get("gate")) for h in r["gate_history"] or []]))
dd = pd.DataFrame(dr)
out["drift"] = {}
for (arm, lab, ns, st), d in dd.groupby(["arm", "lab", "ns", "steps"]):
    out["drift"].setdefault(arm, {}).setdefault(lab, {}).setdefault(str(ns), {})[str(st)] = {
        k: ms(d[k]) for k in ("G", "edit", "path", "energy", "kl", "visit", "neg_energy", "lab_energy", "unl_energy")}
out["gate"] = {}
for (arm, lab, ns), d in dd[dd.steps == 5000].groupby(["arm", "lab", "ns"]):
    H = [np.array(h, float) for h in d.gate if len(h)]
    L = min(len(h) for h in H)
    out["gate"].setdefault(arm, {}).setdefault(lab, {})[str(ns)] = [[int(H[0][i, 0]), round(float(np.mean([h[i, 1] for h in H])), 4)] for i in range(L)]
out["counts"] = {"ksweep_runs": int(len(ks)), "variant_runs": int(len(vd)), "drift_runs": int(len(dd))}
json.dump(out, open("results/robol/sdpo_study.json", "w"))
print(json.dumps(out["counts"]), json.dumps(out["base"]), json.dumps(out["segment_cases"]))
