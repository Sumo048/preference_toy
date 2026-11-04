"""Reproduction check on a dev cell (k4 seed 0 npm 4): e5's chunked K10c fit vs E1's stored graft and D.
Usage: OMP_NUM_THREADS=1 python explore/E5/repro_dev.py <scratch dir>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5, torch
torch.set_num_threads(1)
w = sys.argv[1]
e5.OUT, e5.CACHE, e5.GRAFTS = (os.path.join(w, d) for d in ("results", "cache", "grafts"))
n = 1
while not e5.unit_fit("k4", 0):
    n += 1
mine = torch.load(os.path.join(e5.GRAFTS, "grafts_K10c_k4_s0_n4.pt"), weights_only=False)
e1 = torch.load(os.path.join(e5.ROOT, "explore", "E1", "grafts", "grafts_K10c_k4_s0_n4.pt"), weights_only=False)
same = len(mine) == len(e1) and all(torch.equal(a[k], b[k]) for a, b in zip(mine, e1) for k in a)
r = json.load(open(os.path.join(e5.OUT, "graft_K10c_k4_s0_n4.json")))
r1 = json.load(open(os.path.join(e5.ROOT, "explore", "E1", "results", "graft_K10c_k4_s0_n4.json")))
print("units", n, "grafts bit-identical to E1:", same, "D", r["D"], "E1 D", r1["D"],
      "all metrics equal:", all(r[k] == r1[k] for k in ("D", "Removed", "Collateral", "off")))
