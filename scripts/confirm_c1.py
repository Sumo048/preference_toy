"""Evaluate pre-registered confirmation hypothesis C1 (docs/REROUTE_WINS_PREREG.md)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import analyze_rw as A
import pandas as pd

ROOT = "/grogu/datasets/rollout/soumojib/toypref/runs/rw/confirm_C1/cells"
grid = {(k, n) for k in (1, 2, 3, 4) for n in (1, 2, 4, 8, 16, 32)}
df = A.load(ROOT, "C1", set(range(5, 10)), grid)
R = "R_near_nf_nhood_neg"
OTHERS = ["B3_npo_b10_negonly", "B3_npo_b100_negonly", "B5_kto_b10_negonly", "B5_kto_b100_negonly"]
out = ["# C1 confirmation (fresh seeds 5-9, full n_neg grid)", "",
       f"runs: {len(df)} ({df.groupby('arm').size().to_dict()})", ""]
out.append("| n_forget | " + " | ".join(["R_near_nf_nhood_neg"] + OTHERS + ["R_ours_neg", "B0"]) + " |")
out.append("|---" * 8 + "|")
piv = df.pivot_table(index="n_forget", columns="arm", values="tv_to_ideal")
for k in (1, 2, 3, 4):
    out.append(f"| {k} | " + " | ".join(f"{piv.loc[k, a]:.3f}" for a in [R] + OTHERS + ["R_ours_neg", "B0_frozen"]) + " |")
out.append("\n## C1 test: R minus each non-rerouting arm, paired over (seed, n_neg), n = 30\n")
out.append("| n_forget | vs arm | diff | sem | R lower in | passes (< -2 sem) |\n|---|---|---|---|---|---|")
verdict = {}
for k in (1, 2, 3, 4):
    d = df[df.n_forget == k]
    ok = True
    for o in OTHERS:
        x = d[d.arm == R].set_index(["seed", "n_neg"]).tv_to_ideal
        y = d[d.arm == o].set_index(["seed", "n_neg"]).tv_to_ideal
        z = (x - y).dropna()
        p = z.mean() < -2 * z.sem()
        ok &= p
        out.append(f"| {k} | {o} | {z.mean():+.3f} | {z.sem():.3f} | {int((z < 0).sum())}/{len(z)} | {'Y' if p else 'N'} |")
    m = d.groupby("arm")[["neg_mass_ratio", "retain_tv", "mass_off"]].mean()
    dom = [o for o in m.index if o != R and (m.loc[o] <= m.loc[R]).all() and (m.loc[o] < m.loc[R]).any()]
    verdict[k] = (ok, dom)
out.append("")
for k, (ok, dom) in verdict.items():
    out.append(f"- n_forget={k}: beats every non-rerouting neg-only arm: **{'YES' if ok else 'NO'}**; dominated by: {dom or '-'}")
out.append("")
out.append(f"**C1 (n_forget 3 and 4): {'CONFIRMED' if verdict[3][0] and verdict[4][0] and not verdict[3][1] and not verdict[4][1] else 'NOT CONFIRMED'}**; "
           f"side prediction (no win at n_forget=1): {'held' if not verdict[1][0] else 'FAILED'}")
for col in ("neg_mass_ratio", "retain_tv", "mass_off"):
    out.append(f"\n{col} by n_forget\n")
    t = df.pivot_table(index="arm", columns="n_forget", values=col).round(3)
    out.append(t.to_string())
out.append("\ntv_to_ideal by n_neg at n_forget >= 3\n")
t = df[df.n_forget >= 3].pivot_table(index="arm", columns="n_neg", values="tv_to_ideal").round(3)
out.append(t.to_string())
p = os.path.join(os.path.dirname(__file__), "..", "results", "rw", "confirm_C1.md")
open(p, "w").write("\n".join(out) + "\n")
print("\n".join(out))
