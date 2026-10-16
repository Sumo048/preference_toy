"""robol round 2: label noise (flip0.1, flip0.25, recall0.5 vs clean) on the default env, and the two
environment variants (route mode, speed marker). G = arm - in-task B0, held group, mean +- sem over seeds.
Writes results/robol/RESULTS_round2.md.
"""
import glob, json, os, re
import numpy as np, pandas as pd

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ARMS = ["SDPO_fresh_b25", "SDPO_pin_b25", "NPO_b100_neg", "KTO_b100_neg"]
recs = []
for f in glob.glob(os.path.join(ROBO, "cells*", "*", "s*", "*.json")):
    var = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f)))).replace("cells", "").lstrip("_") or "default"
    m = re.fullmatch(r"(R[12])_ns(\d+)_([a-z_]+?)(?:_(flip[\d.]+|recall[\d.]+))?", os.path.basename(os.path.dirname(os.path.dirname(f))))
    if not m:
        continue
    r = json.load(open(f))
    if r["arm"] not in ARMS:
        continue
    b = r["b0"]
    recs.append({"variant": var, "budget": m.group(1), "ns": int(m.group(2)), "labels": m.group(3), "noise": m.group(4) or "clean",
                 "arm": r["arm"], "seed": r["seed"],
                 "G": r["held_success_clean"] - b["held_success_clean"],
                 "dcol": r["held_collision"] - b["held_collision"], "djerk": r["held_jerk"] - b["held_jerk"],
                 "speed": r["held_speed_ratio"], "mode_tv": r["held_mode_tv"], "broken": r["held_good_broken"],
                 "b0_clean": b["held_success_clean"], "b0_col": b["held_collision"], "b0_jerk": b["held_jerk"]})
df = pd.DataFrame(recs)


def fmt(x):
    if len(x) == 0:
        return "—"
    m, s = x.mean(), x.sem() if len(x) > 1 else 0.0
    t = " ▲" if m > 2 * s else (" ▼" if m < -2 * s else "")
    return f"{100 * m:+.1f}±{100 * s:.1f}{t}"


out = ["# robol round 2: label noise and environment variants", "", "G = change in clean success on held-out courses vs the in-task B0 (points), seeds 0-4.", ""]

d = df[df.variant == "default"]
out += ["## Label noise (default env)", ""]
for budget in ("R1", "R2"):
    out += [f"### {budget}", "", "| arm | labels | N_s | clean | flip0.1 | flip0.25 | recall0.5 |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for L in ("traj", "approach"):
            for ns in (128, 512):
                x = d[(d.budget == budget) & (d.arm == arm) & (d.labels == L) & (d.ns == ns)]
                out.append(f"| {arm} | {L} | {ns} | " + " | ".join(fmt(x[x.noise == z].G) for z in ("clean", "flip0.1", "flip0.25", "recall0.5")) + " |")
    out.append("")

for var in ("route", "speed"):
    v = df[df.variant == var]
    if not len(v):
        continue
    b0 = v.groupby("seed")[["b0_clean", "b0_col", "b0_jerk"]].first()
    out += [f"## Variant: {var}", "", f"B0 held: clean {b0.b0_clean.mean():.3f}, collision {b0.b0_col.mean():.3f}, jerk {b0.b0_jerk.mean():.3f}", ""]
    for budget in ("R1", "R2"):
        out += [f"### {budget}", "", "| arm | labels | G@32 | G@128 | G@512 | Δcollision@512 | Δjerk@512 | speed@512 | mode_tv@512 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for L in ("traj", "approach"):
                x = v[(v.budget == budget) & (v.arm == arm) & (v.labels == L)]
                x5 = x[x.ns == 512]
                out.append(f"| {arm} | {L} | " + " | ".join(fmt(x[x.ns == n].G) for n in (32, 128, 512)) +
                           f" | {fmt(x5.dcol)} | {fmt(x5.djerk)} | {x5.speed.mean():.3f} | {x5.mode_tv.mean():.3f} |")
        out.append("")
    # default-env reference at the same cells
    out += [f"Default-env reference (round 1, same arms): see RESULTS.md.", ""]

os.makedirs("results/robol", exist_ok=True)
open("results/robol/RESULTS_round2.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
