"""Collect B0 and B1a (scripts/cond_cell.py outputs) for the E1 cells into results/baselines.json."""
import json, os, glob
R = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "cond")
out = {}
for L, d in (("k4", "cells"), ("alt", "cells_alt")):
    for s in range(5):
        for n in (4, 16):
            for arm, nm in (("B0_frozen", "B0"), ("R_hopB_bsup_neg_AMdist", "B1a")):
                f = os.path.join(R, d, f"s{s}_k4_n{n}", f"{arm}.json")
                if os.path.exists(f):
                    r = json.load(open(f))
                    out[f"{L}_s{s}_n{n}/{nm}"] = {"D": r["tv_to_ideal"], "Removed": 1 - r["neg_mass_ratio"],
                                                  "Collateral": r["retain_tv"], "off": r["mass_off"], "train_s": r["train_s"]}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "baselines.json"), "w"), indent=1)
for k, v in sorted(out.items()):
    print(f"{k:22s} D {v['D']:.3f} Rem {v['Removed']:.3f} Col {v['Collateral']:.3f} off {v['off']:.3f}")
