"""t-gated B1a: B1a's network for t >= tau (early decode), pi_ref's network for t < tau.
Equivalent to gating B1a's whole weight change by t. Uses the cached B1a weights (b1a_trunc.py)."""
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
import lib
from lib import E

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--layout", default="random")
ap.add_argument("--npm", type=int, required=True)
args = ap.parse_args()
torch.set_num_threads(1)
cell = lib.load_cell(args.seed, args.layout, args.npm)
cp = os.path.join(os.environ["TOYPREF_RUNS"], "E2cache", f"b1a_{cell.name}.pt")
b1a = copy.deepcopy(cell.ref); b1a.load_state_dict(torch.load(cp))


class Switch(nn.Module):
    def __init__(self, early, late, tau):
        super().__init__()
        self.early, self.late, self.tau = early, late, tau

    def forward(self, x, t, c):
        m = (t >= self.tau).float()[:, None]
        return m * self.early(x, t, c) + (1 - m) * self.late(x, t, c)


od = os.path.join(os.path.dirname(__file__), "results", cell.name)
for tau in (0.2, 0.3, 0.4, 0.5, 0.6):
    pol = copy.deepcopy(cell.ref)
    pol.net = Switch(b1a.net, cell.ref.net, tau)
    r = E.compute(cell.bank, pol)
    res = {"D": r["tv_to_ideal"], "Removed": 1 - r["neg_mass_ratio"], "Collateral": r["retain_tv"], "off": r["mass_off"]}
    name = f"b1a_t{tau}-1"
    json.dump({**res, "method": name}, open(os.path.join(od, f"{name}.json"), "w"))
    print(cell.name, name, lib.fmt(res), flush=True)
