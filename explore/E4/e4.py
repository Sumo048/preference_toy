"""E4 confirmation runner: E3's methods, unchanged, on all 7 conditional settings and seeds 5-9.

This is a thin wrapper over explore/E3/e3.py. It imports e3 and changes only:
  - SETTINGS: the 7 method-map settings (k1, k2, k4 random; contig, alt, pairs, b3p1 at n_forget 4),
  - OUT: explore/E4/results,
  - Cell.__init__: identical to e3.Cell.__init__ except the 'seeds 0-4 only' assert.
Method code (run_method, anchor_fn, tracks.py, b0_record) is e3's, called as is.

Usage: python explore/E4/e4.py --setting k4 --seed 5 --npm 4 --methods B0,B1a,a=track60k-w=1-st=20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "explore", "E3"))
import e3  # noqa: E402  (also puts src/ on sys.path)

import torch  # noqa: E402
from toypref.config import RUNS  # noqa: E402
from toypref.cond import experiment as E  # noqa: E402
from toypref.cond.task import CondFlow, frozen  # noqa: E402

e3.SETTINGS = {"k1": (1, "random"), "k2": (2, "random"), "k4": (4, "random"),
               "contig": (4, "contig"), "alt": (4, "alt"), "pairs": (4, "pairs"), "b3p1": (4, "b3p1")}
e3.OUT = os.path.join(ROOT, "explore", "E4", "results")
METHOD = "a=track60k-w=1-st=20000"


def _init(self, setting, seed, npm):
    k, layout = e3.SETTINGS[setting]
    self.setting, self.seed, self.npm = setting, seed, npm
    sd = torch.load(os.path.join(RUNS, "cond_base", f"s{seed}", "policy.pt"), map_location="cpu",
                    weights_only=False)
    ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); self.ref = frozen(ref)
    pool = E.make_pool(self.ref, seed, 200_000)
    self.ed = E.editset(pool, seed, k, npm, layout)
    anchor = E.make_pool(self.ref, seed, 20_000, purpose=E.P_ANCHOR)
    self.ctx = E.Ctx(ref=self.ref, seed=seed, ed=self.ed, pool=pool, anchor=anchor)
    self.bank = E.Bank(self.ref, seed, self.ed)
    self._rows = {}
    self._oracle = None
    self._tracks = None
    self.diag = {}


e3.Cell.__init__ = _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=list(e3.SETTINGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--npm", type=int, required=True)
    ap.add_argument("--methods", default=f"B0,B1a,{METHOD}")
    ap.add_argument("--out", default=e3.OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    od = os.path.join(args.out, args.setting, f"s{args.seed}_n{args.npm}")
    os.makedirs(od, exist_ok=True)
    todo = [m for m in args.methods.split(",") if not os.path.exists(os.path.join(od, f"{m}.json"))]
    if not todo:
        return
    cell = e3.Cell(args.setting, args.seed, args.npm)
    for m in todo:
        rec = e3.b0_record(cell) if m == "B0" else e3.run_method(cell, m)
        tmp = os.path.join(od, f".{m}.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(od, f"{m}.json"))
        print(f"[{args.setting} s{args.seed} n{args.npm}] {m:28s} D {rec['D']:.3f} Rem {rec['Removed']:.3f} "
              f"Col {rec['Collateral']:.3f} off {rec['mass_off']:.3f} ({rec.get('train_s', 0)}s)", flush=True)


if __name__ == "__main__":
    main()
