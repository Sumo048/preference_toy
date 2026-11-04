"""Tables of D, Removed, Collateral per method vs B0 and B1a on the same cells (markdown to stdout)."""
import glob, json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
base = json.load(open(os.path.join(H, "results", "baselines.json")))
recs = {}
for f in glob.glob(os.path.join(H, "results", "graft_*.json")):
    r = json.load(open(f)); recs.setdefault(r["tag"], {})[r["cell"]] = r
tags = sys.argv[1:] or sorted(recs)

def row(name, cells, get):
    v = [get(c) for c in cells]
    if any(x is None for x in v):
        return None
    v = np.array(v)
    return f"| {name} | " + " | ".join(f"{v[:, i].mean():.3f} ± {v[:, i].std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.3f}" for i in range(3)) + " |"

bget = lambda arm: (lambda c: (lambda b: None if b is None else (b["D"], b["Removed"], b["Collateral"]))(base.get(f"{c}/{arm}")))
for tag in tags:
    for setting in ("k4", "alt"):
        cells = sorted(c for c in recs[tag] if c.startswith(setting + "_"))
        if not cells:
            continue
        print(f"\n**{tag}, {setting}** ({len(cells)} cells: {', '.join(cells)}), mean ± s.e.\n")
        print("| Method | D | Removed | Collateral |\n|---|---|---|---|")
        for nm, gt in (("B0", bget("B0")), ("B1a", bget("B1a")),
                       (f"graft {tag}", lambda c: (recs[tag][c]["D"], recs[tag][c]["Removed"], recs[tag][c]["Collateral"]))):
            r = row(nm, cells, gt)
            if r:
                print(r)
        d = [recs[tag][c]["D"] - base[f"{c}/B1a"]["D"] for c in cells if f"{c}/B1a" in base]
        if d:
            print(f"\ngraft − B1a in D: {np.mean(d):+.3f} ± {np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:.3f}; lower in {sum(x < 0 for x in d)} of {len(d)}")
