"""Final evaluation of a version on a cell (docs/PLAN.md 4.8, 9.5).

One of only two modules allowed to import ``toyrepair.oracle``. Final D, Removed and Collateral
come from the toypref code (``toyrepair.oracle.metrics.toypref_compute``: toypref's fixed eval
bank, 20,000 (c, xi) on stream (seed, P_EVAL)), so the numbers match the method map. Results
also include toyrepair FINAL-pool numbers (``pools.FINAL``) and the target cluster's removed
share. Final numbers never go back into any loop.

Collateral: ``FinalResult.collateral`` is the method-map value (toypref ``retain_tv``, retained
modes only). The PLAN.md Section 2 definition (retained modes + between-modes bucket) is in
``final_pool["collateral"]`` and ``collateral_plan``.

Usage: ``evaluate(cell, version, arm, target_clusters)`` then ``write(result, path)``; or
``python -m toyrepair.eval_final --setting k4 --seed 0 --npm 4`` for pi_ref (B0).

Owner: W1-A2.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from . import oracle, pools
from .oracle import _adapt
from .oracle.metrics import scores_from_masses, toypref_compute

N_FINAL = 65_536


@dataclass
class FinalResult:
    """cell key; arm; version id; D, removed, collateral, mass_off (toypref eval bank);
    toypref: full toypref compute dict; target_removed: removed share of the target clusters'
    oracle modes (None for global arms); final_pool: same scores on the FINAL pool."""
    cell: str
    arm: str
    version: str
    D: float
    removed: float
    collateral: float
    mass_off: float
    toypref: Dict[str, float] = field(default_factory=dict)
    target_removed: Optional[float] = None
    final_pool: Dict[str, float] = field(default_factory=dict)
    collateral_plan: float = 0.0
    target_modes: List[int] = field(default_factory=list)
    target_removed_toypref: Optional[float] = None
    eval_s: float = 0.0


_FINAL_REF: Dict = {}


def _final_ref(cell):
    key = (cell.key, id(cell.ref))
    if key not in _FINAL_REF:
        c, xi = pools.draw(cell, pools.FINAL, N_FINAL, index=0)
        a = _adapt.actions(cell.ref, xi, c)
        _FINAL_REF[key] = (c, xi, oracle.label(cell, a, c))
    return _FINAL_REF[key]


def _uids_of(cl):
    return [int(u) for u in getattr(cl, "uids", cl)]


def evaluate(cell, version, arm: str, target_clusters: Optional[List] = None) -> FinalResult:
    """Final evaluation of ``version`` (``policy.Version`` or anything with ``.net``).

    target_clusters: list of clusters (uid lists or objects with ``.uids``) the run repaired;
    the target modes are the union of their majority oracle modes. None for global arms."""
    t0 = time.time()
    F, R = oracle.forget_modes(cell), oracle.retain_modes(cell)
    tp = toypref_compute(cell, version)
    c, xi, lab_ref = _final_ref(cell)
    lab = oracle.label(cell, _adapt.actions(version, xi, c), c)
    m, mr = oracle.masses(lab), oracle.masses(lab_ref)
    sc = scores_from_masses(m, mr, F, R)
    fp = {**sc.as_dict(), "n": N_FINAL, "mass": m.tolist(), "mass_ref": mr.tolist()}

    T: List[int] = []
    target_removed = target_removed_tp = None
    if target_clusters:
        T = sorted({md for cl in target_clusters for md in oracle.target_modes(cell, _uids_of(cl))})
        if T:
            target_removed = 1.0 - float(m[T].sum() / max(mr[T].sum(), 1e-12))
            mt = np.array([tp[f"mass_m{j}"] for j in T]).sum()
            # pi_ref masses on the toypref bank: tv_to_ideal_ref is there but per-mode ref masses
            # are not; recompute them once from the bank labels
            target_removed_tp = 1.0 - float(mt / max(_toypref_ref_mass(cell, T), 1e-12))
    # PLAN collateral on the toypref bank too
    mtp = np.array([tp[f"mass_m{j}"] for j in range(oracle.N_MODES)] + [tp["mass_off"]])
    coll_plan = scores_from_masses(mtp, _toypref_ref_masses(cell), F, R).collateral
    return FinalResult(cell=cell.key, arm=arm, version=str(getattr(version, "id", "?")),
                       D=float(tp["tv_to_ideal"]), removed=1.0 - float(tp["neg_mass_ratio"]),
                       collateral=float(tp["retain_tv"]), mass_off=float(tp["mass_off"]), toypref=tp,
                       target_removed=target_removed, final_pool=fp, collateral_plan=float(coll_plan),
                       target_modes=T, target_removed_toypref=target_removed_tp, eval_s=time.time() - t0)


def _toypref_ref_masses(cell) -> np.ndarray:
    from .oracle import metrics as M
    if cell.grid == "cond":
        return oracle.masses(M._cond_bank(cell).lab_ref)
    return oracle.masses(M._uncond_bank(cell)["lab_ref"])


def _toypref_ref_mass(cell, modes) -> float:
    return float(_toypref_ref_masses(cell)[list(modes)].sum())


def write(res: FinalResult, path: str) -> None:
    """Write one result as JSON (tmp + replace). Skip-if-exists is the caller's choice."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(res), f, indent=1, default=float)
    os.replace(tmp, path)


def _main(argv=None):
    import argparse
    from .cell import load_cell
    ap = argparse.ArgumentParser(prog="python -m toyrepair.eval_final", description="final evaluation of pi_ref (B0)")
    ap.add_argument("--grid", default="cond")
    ap.add_argument("--setting", default="k4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npm", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    cell = load_cell(a.grid, a.setting, a.seed, a.npm)
    r = evaluate(cell, _adapt.base_version(cell), "B0")
    d = {k: v for k, v in asdict(r).items() if k != "toypref"}
    print(json.dumps(d))
    if a.out:
        write(r, a.out)


if __name__ == "__main__":
    _main()
