"""Collect every per-arm JSON into tidy CSVs.

results/all_runs.csv         one row per (seed, n_forget, n_neg, arm), every scalar metric
results/summary.csv          mean / sem / n over seeds, per (arm, n_forget, n_neg)
results/paired_vs_B0.csv     per-seed arm minus B0 differences (same eval noises), summarised
results/extras.jsonl         mass vectors, transition matrices, per-mode log-density deltas
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from toypref.config import ROOT, RUNS

KEYS = ["arm", "n_forget", "n_neg"]
DROP = {"extra", "train_history", "train_history_r2", "config", "online_r2_found_per_mode", "forget"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(RUNS, "cells"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, extras = [], []
    for p in sorted(glob.glob(os.path.join(args.cells, "*", "*.json"))):
        with open(p) as f:
            r = json.load(f)
        extras.append({k: r[k] for k in ("cell", "arm", "seed", "n_forget", "n_neg", "forget")} | {"extra": r["extra"]})
        hist = r.get("train_history") or []
        row = {k: v for k, v in r.items() if k not in DROP}
        row["forget"] = "-".join(map(str, r["forget"]))
        if hist:
            row["final_train_loss"] = hist[-1].get("loss")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)
    with open(os.path.join(args.out, "extras.jsonl"), "w") as f:
        for e in extras:
            f.write(json.dumps(e) + "\n")

    num = [c for c in df.columns if c not in KEYS and c != "seed" and pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c])]
    g = df.groupby(KEYS)[num]
    mean, sem, cnt = g.mean(), g.sem(), g.size().rename("n_seeds")
    summ = mean.add_suffix("_mean").join(sem.add_suffix("_sem")).join(cnt).reset_index()
    meta = df.drop_duplicates("arm").set_index("arm")[["kind", "family", "mix", "negonly", "negative_control",
                                                       "extra_label_budget", "privileged"]]
    summ = summ.join(meta, on="arm")
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    b0 = df[df.arm == "B0_frozen"].set_index(["seed", "n_forget", "n_neg"])[num]
    diffs = []
    for arm, d in df.groupby("arm"):
        d = d.set_index(["seed", "n_forget", "n_neg"])[num]
        common = d.index.intersection(b0.index)
        dd = (d.loc[common] - b0.loc[common]).reset_index()
        dd["arm"] = arm
        diffs.append(dd)
    if diffs:
        dd = pd.concat(diffs)
        gd = dd.groupby(KEYS)[num]
        pv = gd.mean().add_suffix("_dmean").join(gd.sem().add_suffix("_dsem")).reset_index()
        pv.to_csv(os.path.join(args.out, "paired_vs_B0.csv"), index=False)
    print(f"{len(df)} runs, {df.arm.nunique()} arms, {df.cell.nunique()} cells -> {args.out}")


if __name__ == "__main__":
    main()
