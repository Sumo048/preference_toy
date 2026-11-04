"""Check that explore/E5/grafts/*.pt load with E1's standalone loader (explore/E1/weights/load.py,
GraftedPolicy; no E1/E5 method code) and reproduce the recorded D. One cell per setting by default.
Usage: python explore/E5/render_check.py [cell ...]     e.g. k4_s5_n4 alt_s7_n16"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "explore", "E1", "weights")); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TOYPREF_RUNS", os.path.join(ROOT, "runs"))
import torch
from load import GraftedPolicy, load_ref
from toypref.cond import experiment as E
SET = {"k1": 1, "k2": 2}
cells = sys.argv[1:] or [f"{S}_s5_n4" for S in ("k4", "alt", "k1", "contig", "pairs", "b3p1", "k2")]
for cell in cells:
    meta = json.load(open(os.path.join(HERE, "results", f"graft_K10c_{cell}.json")))
    fit_cell = cell if "note" not in meta else meta["note"].split()[3].rstrip(";")
    rounds = torch.load(os.path.join(HERE, "grafts", f"grafts_K10c_{fit_cell}.pt"), map_location="cpu", weights_only=True)
    s = meta["seed"]; ref = load_ref(s)
    pol = GraftedPolicy(ref, rounds, meta["cfg"]["k_gate"])
    ed = E.editset(E.make_pool(ref, s, 200_000), s, SET.get(meta["setting"], 4), meta["npm"], meta["layout"])
    D = E.compute(E.Bank(ref, s, ed), pol)["tv_to_ideal"]
    print(f"{cell} (graft {fit_cell}): D {D:.5f} recorded {meta['D']:.5f} {'OK' if abs(D - meta['D']) < 1e-6 else 'MISMATCH'}", flush=True)
