"""Analysis of drift experiments A and D. Writes results/robol/drift_vs_steps.png, drift_gain.png, drift_gate.png and
results/robol/DRIFT.md.
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robol")
ARMS = ["SDPO_pin_b25", "SDPO_fresh_b25", "SDPO_own_b25_K1", "SDPO_pin_b25_W", "SDPO_pin_b25_L", "SDPO_fresh_b25_W", "SDPO_fresh_b25_L"]
NICE = {"SDPO_pin_b25": "pinned", "SDPO_fresh_b25": "fresh", "SDPO_own_b25_K1": "own-noise winner",
        "SDPO_pin_b25_W": "pinned, winner only", "SDPO_pin_b25_L": "pinned, loser only",
        "SDPO_fresh_b25_W": "fresh, winner only", "SDPO_fresh_b25_L": "fresh, loser only"}
COL = {"SDPO_pin_b25": "#c2412d", "SDPO_fresh_b25": "#1f6f78", "SDPO_own_b25_K1": "#7a5195",
       "SDPO_pin_b25_W": "#ef8a62", "SDPO_pin_b25_L": "#f4c09b", "SDPO_fresh_b25_W": "#67a9cf", "SDPO_fresh_b25_L": "#a6d4e6"}
SETS = ["neg", "lab", "unl", "held"]
recs = []
for f in glob.glob(os.path.join(ROBO, "drift", "*_ns*", "s*", "*.json")):
    r = json.load(open(f))
    b = r["b0"]
    row = {"arm": r["arm"], "labels": r["labels"], "ns": r["ns"], "steps": r["steps"], "seed": r["seed"],
           "G": 100 * (r["held_success_clean"] - b["held_success_clean"]), "edit": r["edit_norm"],
           "visit": r["visit_energy"], "mode_tv": r["held_mode_tv"], "speed": r["held_speed_ratio"],
           "gate_end": (r["gate_history"] or [{}])[-1].get("gate", np.nan),
           "margin_end": (r["gate_history"] or [{}])[-1].get("margin", np.nan)}
    for k in SETS:
        for mtr in ("path", "energy", "kl_hat", "meanshift", "spread_ratio"):
            row[f"{k}_{mtr}"] = r[f"{k}_{mtr}"]
    row["gate_hist"] = [(h["step"], h.get("gate", np.nan)) for h in r["gate_history"] or []]
    recs.append(row)
df = pd.DataFrame(recs)
os.makedirs("results/robol", exist_ok=True)
out = [f"# Drift experiments A and D ({len(df)} runs, seeds {sorted(int(x) for x in df.seed.unique())})", ""]

# ---- figure 1: drift vs steps
MET = [("edit", "weight distance"), ("held_path", "pathwise change, held"), ("held_energy", "energy distance, held"),
       ("held_kl_hat", "KL surrogate, held"), ("neg_energy", "energy distance, negative states"), ("visit", "visited-state shift")]
cells = [(L, n) for L in ("traj", "approach") for n in (64, 512)]
fig, axes = plt.subplots(len(cells), len(MET), figsize=(3.3 * len(MET), 2.7 * len(cells)), sharex=True, squeeze=False)
for i, (L, n) in enumerate(cells):
    for j, (mk, ml) in enumerate(MET):
        ax = axes[i, j]
        for a in ARMS:
            d = df[(df.arm == a) & (df.labels == L) & (df.ns == n)]
            g = d.groupby("steps")[mk]
            if len(g.mean()):
                ax.errorbar(g.mean().index, g.mean(), yerr=2 * g.sem().fillna(0), color=COL[a], lw=1.6 if "_" not in a[5:] or a.count("_") == 2 else 1.0,
                            ls="-" if a in ("SDPO_pin_b25", "SDPO_fresh_b25") else "--", marker="o", ms=2, label=NICE[a])
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
        if i == 0: ax.set_title(ml, fontsize=9)
        if j == 0: ax.set_ylabel(f"{L}, N_s={n}", fontsize=9)
        if i == len(cells) - 1: ax.set_xlabel("training steps", fontsize=8)
        ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=6)
fig.tight_layout(); fig.savefig("results/robol/drift_vs_steps.png", dpi=120)

# ---- figure 2: gain vs distribution drift
fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 3.4), squeeze=False)
for j, (L, n) in enumerate(cells):
    ax = axes[0, j]
    for a in ARMS:
        d = df[(df.arm == a) & (df.labels == L) & (df.ns == n)].groupby("steps")[["held_energy", "G"]].mean()
        if len(d):
            ax.plot(d.held_energy.clip(lower=1e-4), d.G, color=COL[a], marker="o", ms=3, lw=1.4,
                    ls="-" if a in ("SDPO_pin_b25", "SDPO_fresh_b25") else "--", label=NICE[a])
    ax.axhline(0, color="0.7", lw=0.7); ax.set_xscale("log"); ax.set_ylim(-80, 18)
    ax.set_title(f"{L}, N_s={n}", fontsize=9); ax.set_xlabel("energy distance from base, held states")
    if j == 0: ax.set_ylabel("Δ clean success (pts)")
axes[0, 0].legend(fontsize=6)
fig.tight_layout(); fig.savefig("results/robol/drift_gain.png", dpi=120)

# ---- figure 3: gate dynamics (5000-step runs)
fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 3.2), squeeze=False)
for j, (L, n) in enumerate(cells):
    ax = axes[0, j]
    for a in ARMS:
        d = df[(df.arm == a) & (df.labels == L) & (df.ns == n) & (df.steps == 5000)]
        if not len(d):
            continue
        H = [np.array(h) for h in d.gate_hist if len(h)]
        L_ = min(len(h) for h in H)
        st = H[0][:L_, 0]; gv = np.mean([h[:L_, 1] for h in H], 0)
        ax.plot(st + 1, gv, color=COL[a], lw=1.4, ls="-" if a in ("SDPO_pin_b25", "SDPO_fresh_b25") else "--", label=NICE[a])
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_title(f"{L}, N_s={n} (5000-step runs)", fontsize=9)
    ax.set_xlabel("step"); ax.set_ylabel("mean gate σ(β(r_l − r_w))") if j == 0 else None
axes[0, 0].legend(fontsize=6)
fig.tight_layout(); fig.savefig("results/robol/drift_gate.png", dpi=120)

# ---- tables
for L, n in cells:
    out += [f"## {L}, N_s = {n}: drift at matched steps (mean over seeds)", ""]
    cols = ["G", "edit", "held_path", "held_energy", "held_kl_hat", "neg_energy", "lab_energy", "unl_energy", "visit", "gate_end"]
    t = df[(df.labels == L) & (df.ns == n)].groupby(["steps", "arm"])[cols].mean().reset_index()
    t["arm"] = t.arm.map(NICE)
    out.append(t.round(4).to_string(index=False))
    out.append("")
    out += [f"### pinned / fresh ratio at matched steps ({L}, N_s = {n})", ""]
    p = df[(df.labels == L) & (df.ns == n) & df.arm.isin(["SDPO_pin_b25", "SDPO_fresh_b25"])].groupby(["steps", "arm"])[cols[1:-1]].mean().unstack()
    ratio = pd.DataFrame({c: p[(c, "SDPO_pin_b25")] / p[(c, "SDPO_fresh_b25")].where(p[(c, "SDPO_fresh_b25")].abs() > 1e-6) for c in cols[1:-1]})
    out.append(ratio.round(2).to_string()); out.append("")
    out += [f"### at each arm's best G ({L}, N_s = {n})", ""]
    for a in ARMS:
        d = df[(df.arm == a) & (df.labels == L) & (df.ns == n)].groupby("steps")[cols].mean()
        if len(d):
            b = d.loc[d.G.idxmax()]
            out.append(f"- {NICE[a]}: best G {b.G:+.1f} at {d.G.idxmax()} steps; edit {b.edit:.2f}, held path {b.held_path:.3f}, "
                       f"held energy {b.held_energy:.4f}, held KL {b.held_kl_hat:.4f}, visit {b.visit:.4f}, gate {b.gate_end:.2f}")
    out.append("")
open("results/robol/DRIFT.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
