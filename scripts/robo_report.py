"""Aggregate robo2d cells: mean +- s.e. over seeds per arm, plus base rates and separability.

    python scripts/robo_report.py --tag full_st5000_m64     # cells/full_st5000_m64/s*/
    python scripts/robo_report.py --tag smoke8_st500_m64

Refuses to aggregate a seed whose arms were scored against different base rollouts (base_hash)
or different base policies: paired deltas vs B0 are only meaningful against one shared base.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

ROBO = os.environ.get("ROBO_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs/robo")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full_st5000_m64")
ap.add_argument("--base", default=os.path.join(ROBO, "base"))
ap.add_argument("--out", default="results/robo")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

KEYS = ["neg_bad", "neg_collision", "neg_jerk", "neg_jerk_intra", "neg_jerk_boundary_only", "neg_bad_col_or_intra", "neg_success", "neg_bad_fixed", "neg_good_broken", "neg_hom_tv",
        "neg_hom_flip", "neg_retained_disp", "neg_path_len", "noneg_bad", "noneg_success", "noneg_hom_tv",
        "noneg_retained_disp", "held_bad", "held_collision", "held_jerk", "held_jerk_intra", "held_jerk_boundary_only", "held_bad_col_or_intra", "held_success", "held_good_broken",
        "held_hom_tv", "held_retained_disp", "held_steps_to_goal", "logged_neg_still_bad"]
recs = []
for f in glob.glob(os.path.join(ROBO, "cells", a.tag, "s*", "*.json")):
    if os.path.basename(f) == "separability.json":
        continue
    r = json.load(open(f))
    r.pop("train_history", None)
    recs.append(r)
if not recs:
    raise SystemExit("no cells")
df = pd.DataFrame(recs)
for col in ("base_hash", "base_policy_sha"):
    if col not in df:
        raise SystemExit(f"cells lack {col}: produced by an older robo_cell.py; rerun them")
    bad = df.groupby("seed")[col].nunique()
    bad = bad[bad > 1]
    if len(bad):
        for sd in bad.index:
            print(df[df.seed == sd][["arm", col]].to_string(index=False))
        raise SystemExit(f"seeds {list(bad.index)}: arms disagree on {col} (node maths or a stale base); "
                         "rerun the odd tasks on matching nodes before reporting")
print(f"base_hash / base_policy_sha consistent across arms for seeds {sorted(df.seed.unique())}")
df.to_csv(os.path.join(a.out, f"cells_{a.tag}.csv"), index=False)
keys = [k for k in KEYS if k in df]
g = df.groupby("arm")[keys]
mean, se, n = g.mean(), g.sem(), g.size()
order = [x for x in ["B0_frozen", "NPO_b10_neg", "NPO_b100_neg", "KTO_b10_neg", "KTO_b100_neg", "R_ours_neg",
                     "R_hop_bsup_neg", "R_hop_bsup_neg_AL", "SDPO_fresh_b1", "SDPO_fresh_b25", "SDPO_pin_b1",
                     "SDPO_pin_b25", "SDPO_hop_pin_b1"] if x in mean.index]
tab = mean.loc[order].round(4)
tab.insert(0, "n_seeds", n.loc[order])
tab.to_csv(os.path.join(a.out, f"summary_{a.tag}.csv"))
pd.set_option("display.width", 250, "display.max_columns", 40)
print(tab.T.to_string())
print("\ns.e. over seeds:\n", se.loc[order].round(4).T.to_string())
# per-seed delta vs B0 on the headline axes (paired by seed: CRN)
if "B0_frozen" in mean.index:
    b0 = df[df.arm == "B0_frozen"].set_index("seed")
    rows = []
    for arm in order:
        d = df[df.arm == arm].set_index("seed")
        for k in ("neg_bad", "neg_bad_col_or_intra", "held_bad", "held_bad_col_or_intra", "neg_hom_tv", "held_hom_tv", "held_success"):
            if k in d:
                dd = (d[k] - b0[k]).dropna()
                rows.append({"arm": arm, "metric": k, "delta_mean": dd.mean(), "delta_se": dd.sem(), "n": len(dd)})
    print("\npaired deltas vs B0:\n", pd.DataFrame(rows).pivot(index="arm", columns="metric", values="delta_mean").loc[order].round(4).to_string())
seps = [json.load(open(f)) | {"seed": f.split("/")[-2]} for f in glob.glob(os.path.join(ROBO, "cells", a.tag, "s*", "separability.json"))]
if seps:
    print("\nseparability:\n", pd.DataFrame(seps).set_index("seed").round(3).T.to_string())
bases = [json.load(open(f)) for f in glob.glob(os.path.join(a.base, "s*", "base_rates.json"))]
if bases:
    bk = ["seed", "all_success", "all_collision", "all_jerk", "all_bad", "hom_setups_multimodal", "hom_max_share_mean",
          "all_jerk_intra", "all_jerk_boundary_only", "neg_episodes_boundary_only_jerk",
          "n_neg_episodes", "n_neg_rows", "setups_with_neg", "setups_without_neg", "setups_wanted_but_none_found"]
    print("\nbase rates:\n", pd.DataFrame(bases)[[k for k in bk if k in bases[0]]].sort_values("seed").round(4).to_string(index=False))
