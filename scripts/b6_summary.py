#!/usr/bin/env python3
"""B6 vs B0 / B1a / B1b on the 140 conditional cells of seeds 5-9 -> results/b6/summary.md (+ .json).

B0, B1a (R_hopB_bsup_neg_AMdist) and B1b (the setting's method-map config) come from W1-C's
results/baselines/<setting>/s<seed>_<setting>_n<npm>/<config>.json (toypref compute); B6 from
results/b6/<setting>/s<seed>/n<npm>/B6.json (eval_final, same toypref bank). Mean over seeds 5-9 and npm,
paired differences per (setting, seed) with D averaged over npm (PLAN 9.5 format).
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
SETTINGS = ("k1", "k2", "k4", "contig", "alt", "pairs", "b3p1")
B1B = {"k1": "R_hopB_bsup_neg_AMdist", "k2": "R_hopB_bsup_mine_neg_AMmap", "k4": "R_hopB_bsup_mine_neg",
       "contig": "R_hopB_bsup_neg_AMdist", "alt": "R_hop_bsup_neg", "pairs": "R_hopB_bsup_mine_neg",
       "b3p1": "R_hopB_bsup_mine_neg"}


def base(setting, seed, npm, cfg):
    p = R / "results" / "baselines" / setting / f"s{seed}_{setting}_n{npm}" / f"{cfg}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"D": d["tv_to_ideal"], "removed": 1 - d["neg_mass_ratio"], "collateral": d["retain_tv"]}


def b6(setting, seed, npm):
    p = R / "results" / "b6" / setting / f"s{seed}" / f"n{npm}" / "B6.json"
    if not p.exists():
        return None
    f = json.loads(p.read_text())["final"]
    return {"D": f["D"], "removed": f["removed"], "collateral": f["collateral"]}


def main():
    rows = []
    for s in SETTINGS:
        for seed in range(5, 10):
            for n in (1, 4, 16, 64):
                rows.append({"setting": s, "seed": seed, "npm": n, "B0": base(s, seed, n, "B0_frozen"),
                             "B1a": base(s, seed, n, "R_hopB_bsup_neg_AMdist"), "B1b": base(s, seed, n, B1B[s]),
                             "B6": b6(s, seed, n)})
    arms = ("B0", "B1a", "B1b", "B6")
    L = ["# B6 (track-masked BC) on the 140 conditional cells, seeds 5-9", "",
         "D = toypref `tv_to_ideal`; mean over seeds 5-9 and npm 1/4/16/64 (cells with all four arms). "
         "Paired: B6 - B1a per (setting, seed), D averaged over npm; mean ± SE, B6 lower in n of N.", "",
         "| setting | cells | B0 | B1a | B1b | B6 | B6 Removed | B6 Collateral | B6 - B1a | B6 lower |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for s in SETTINGS + ("all",):
        rs = [r for r in rows if (s == "all" or r["setting"] == s) and all(r[a] for a in arms)]
        if not rs:
            continue
        m = {a: np.mean([r[a]["D"] for r in rs]) for a in arms}
        keys = sorted({(r["setting"], r["seed"]) for r in rs})
        diffs = np.array([np.mean([r["B6"]["D"] - r["B1a"]["D"] for r in rs if (r["setting"], r["seed"]) == k])
                          for k in keys])
        se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        rem = np.mean([r["B6"]["removed"] for r in rs])
        col = np.mean([r["B6"]["collateral"] for r in rs])
        L.append(f"| {s} | {len(rs)} | {m['B0']:.3f} | {m['B1a']:.3f} | {m['B1b']:.3f} | {m['B6']:.3f} | "
                 f"{rem:.3f} | {col:.3f} | {diffs.mean():+.3f} ± {se:.3f} | {(diffs < 0).sum()}/{len(diffs)} |")
        out[s] = {"cells": len(rs), **{a: float(m[a]) for a in arms}, "b6_minus_b1a": float(diffs.mean()),
                  "se": float(se)}
    missing = [f"{r['setting']} s{r['seed']} n{r['npm']}" for r in rows if not all(r[a] for a in arms)]
    L += ["", f"Missing cells: {len(missing)}" + (f" ({'; '.join(missing[:20])})" if missing else "")]
    (R / "results" / "b6" / "summary.md").write_text("\n".join(L) + "\n")
    (R / "results" / "b6" / "summary.json").write_text(json.dumps({"by_setting": out, "rows": rows}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
