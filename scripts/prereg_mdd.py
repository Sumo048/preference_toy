#!/usr/bin/env python3
"""Minimum detectable D differences at 80% power from Phase 4 variance (W2b, PREREG Section 8).

Unit: one (setting, seed) pair, D averaged over npm (PLAN 9.5). Phase 4 has k4 seeds 0-4 (npm 4, 16).
MDD = (t_{1-a, n-1} + t_{0.8, n-1}) * sd_diff / sqrt(n), with a = 0.05/5 (Holm first step over 6 tests; one-sided for
T1/T2, a/2 per side for T3/T4). sd_diff is the sd of the per-seed paired difference in Phase 4.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

rows = json.loads((Path(__file__).resolve().parent.parent / "results/phase4/table.json").read_text())

def per_seed(arm):
    out = {}
    for r in rows:
        if r.get(arm) and r[arm].get("D") is not None:
            out.setdefault(r["seed"], []).append(r[arm]["D"])
    return {s: np.mean(v) for s, v in out.items() if len(v) == 2}

def sd_diff(a, b):
    A, B = per_seed(a), per_seed(b)
    s = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in s])
    return d, (d.std(ddof=1) if len(d) > 1 else float("nan"))

def mdd(sd, n, two_sided):
    a = 0.05 / 6
    ta = stats.t.ppf(1 - (a / 2 if two_sided else a), n - 1)
    return (ta + stats.t.ppf(0.8, n - 1)) * sd / np.sqrt(n)

N_A3, N_A12 = 35, 15     # (setting, seed) pairs: A3 items 1-7 = 7 settings x 5 seeds; A1/A2 items 1-3
for comp, (a, b), n, two in [("T1a A3 vs B1a", ("A3", "B1a"), N_A3, False),
                             ("T2 A3 vs D1", ("A3", "D1"), N_A3, False),
                             ("A3 vs B4 (proxy for B5, not run in Phase 4)", ("A3", "B4"), N_A3, False),
                             ("T3/T4 proxy: A3 vs D1 sd", ("A3", "D1"), N_A12, True)]:
    d, sd = sd_diff(a, b)
    print(f"{comp}: Phase 4 paired diff mean {d.mean():+.3f} sd {sd:.3f} (n={len(d)} seeds); "
          f"grid n={n}: MDD {mdd(sd, n, two):.3f}")
