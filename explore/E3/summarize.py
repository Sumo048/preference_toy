"""Aggregate explore/E3/results into a markdown table (per setting and pooled), paired against B1a."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
recs = defaultdict(dict)   # method -> (setting, seed, npm) -> rec
for f in glob.glob(os.path.join(ROOT, "results", "*", "*", "*.json")):
    r = json.load(open(f))
    if r["seed"] in seeds:
        recs[r["method"]][(r["setting"], r["seed"], r["npm"])] = r


def row(m, setting=None):
    cells = {k: v for k, v in recs[m].items() if setting is None or k[0] == setting}
    if not cells:
        return None
    D = np.array([v["D"] for v in cells.values()])
    Rm = np.array([v["Removed"] for v in cells.values()])
    C = np.array([v["Collateral"] for v in cells.values()])
    off = np.array([v["mass_off"] for v in cells.values()])
    common = [k for k in cells if k in recs["B1a"]]
    diff = np.array([cells[k]["D"] - recs["B1a"][k]["D"] for k in common])
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    lower = int((diff < 0).sum())
    return (f"| {m} | {len(cells)} | {D.mean():.3f} | {Rm.mean():.3f} | {C.mean():.3f} | {off.mean():.3f} | "
            f"{diff.mean():+.3f} ± {se:.3f} | {lower}/{len(diff)} |")


hdr = ("| Method | cells | D | Removed | Collateral | mass_off | ΔD vs B1a (± SE) | lower than B1a |\n"
       "|---|---|---|---|---|---|---|---|")
order = sorted(recs, key=lambda m: (m not in ("B0", "B1a"), m != "B0", np.mean([v["D"] for v in recs[m].values()])))
for setting in (None, "k4", "alt"):
    print(f"\n### {'pooled k4+alt' if setting is None else setting} (seeds {seeds}, npm 4 and 16)\n")
    print(hdr)
    for m in order:
        r = row(m, setting)
        if r:
            print(r)
