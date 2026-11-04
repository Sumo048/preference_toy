"""Table of D, Removed, Collateral (retain TV), off mass per method vs B0 and B1a on the same cells.
Usage: python explore/E2/summarize.py [--seeds 0,1,2] [--methods m1,m2] [--md out.md]"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--methods", default="")
ap.add_argument("--npm", default="4,16")
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]
npms = [int(s) for s in args.npm.split(",")]


def base(setting, s, n, arm):
    d = "cells" if setting == "k4" else "cells_alt"
    f = os.path.join(ROOT, "runs", "cond", d, f"s{s}_k4_n{n}", f"{arm}.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    return {"D": r["tv_to_ideal"], "Removed": 1 - r["neg_mass_ratio"], "Collateral": r["retain_tv"], "off": r["mass_off"]}


def mine(setting, s, n, m):
    f = os.path.join(HERE, "results", f"{setting}_s{s}_n{n}", f"{m}.json")
    return json.load(open(f)) if os.path.exists(f) else None


methods = args.methods.split(",") if args.methods else sorted(
    {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(HERE, "results", "*", "*.json"))})
rows = []
for setting in ("k4", "alt"):
    for m in ["B0", "B1a"] + methods:
        vals, n_ok, wins = [], 0, 0
        for s in seeds:
            for n in npms:
                r = base(setting, s, n, "B0_frozen") if m == "B0" else base(setting, s, n, "R_hopB_bsup_neg_AMdist") if m == "B1a" else mine(setting, s, n, m)
                if r is None:
                    continue
                vals.append([r["D"], r["Removed"], r["Collateral"], r["off"]])
                if m not in ("B0", "B1a"):
                    b = base(setting, s, n, "R_hopB_bsup_neg_AMdist")
                    if b is not None:
                        n_ok += 1
                        wins += r["D"] < b["D"]
        if not vals:
            continue
        v = np.array(vals)
        mu, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(4)
        rows.append((setting, m, len(v), mu, se, f"{wins}/{n_ok}" if n_ok else ""))
print(f"seeds {seeds} npm {npms}")
print("| setting | method | cells | D | Removed | Collateral | off | D < B1a |")
print("|---|---|---|---|---|---|---|---|")
for setting, m, n, mu, se, w in rows:
    print(f"| {setting} | {m} | {n} | {mu[0]:.3f} ± {se[0]:.3f} | {mu[1]:.3f} | {mu[2]:.3f} | {mu[3]:.3f} | {w} |")
