"""H1 (scarce positives): the pre-registered decision rule at n_pos in {4, 16, 64, 512}.

T_p-dependent arms come from $RUNS/cells_npos<N> (n_pos sweep, jobs 3990444/5); arms that never
read T_p (NEG / NEG+U, ORACLE) are borrowed from the base regime. Exploration grid only, since
the NEG+U arms exist only there.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import analyze_rw as A
from toypref.arms import ARMS, depends_on_tp
from toypref.config import RUNS

seeds = {0, 1, 2, 3, 4}
grid = {(k, n) for k in (1, 2, 4) for n in (1, 4, 16)}
main = A.load(os.path.join(RUNS, "cells"), "base", seeds, grid)
rwb = A.load(os.path.join(A.RW_ROOT, "base", "cells"), "base", seeds, grid)
base = pd.concat([main[~main.arm.isin(set(rwb.arm))], rwb])
base = base[base.arm.isin(ARMS)]
regimes = {"npos512": base}
borrow = base[~base.arm.map(lambda a: depends_on_tp(ARMS[a]))]
for n in (64, 16, 4):
    d = pd.concat([A.load(os.path.join(RUNS, f"cells_npos{n}"), f"npos{n}", seeds, grid),
                   A.load(os.path.join(A.RW_ROOT, f"npos{n}", "cells"), f"npos{n}", seeds, grid)])
    d = d.drop_duplicates(["seed", "n_forget", "n_neg", "arm"])
    d = d[d.arm.isin(ARMS)]
    regimes[f"npos{n}"] = pd.concat([d, borrow[~borrow.arm.isin(set(d.arm))].assign(regime=f"npos{n}")])
body, verdict = A.evaluate(regimes)
p = os.path.join(os.path.dirname(__file__), "..", "results", "rw", "h1_npos.md")
open(p, "w").write("\n".join(["# H1: scarce positives (n_pos sweep), pre-registered decision rule", ""] + body) + "\n")
print("\n".join(body[-(len(verdict) + 3):]))
