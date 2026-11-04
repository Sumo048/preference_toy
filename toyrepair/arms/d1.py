"""D1: branch-and-distill: the A3 loop with code in place of the LLM; best primitive -> rescue targets for members and basin supports -> targeted_train (BC + locality penalty); fixed list of 3 variants (set in Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .base import ArmOutput, RepairContext, Services, iterate


@dataclass(frozen=True)
class Variant:
    """One D1 candidate: targeted_train(params, bc to rescue targets with ``support``, steps, anchor)."""
    name: str
    params: Tuple[str, ...]
    steps: int
    anchor: str = "dist"
    support: str = "bsup"


#: the fixed variant list, submitted in this order until one is accepted (docs/DEV_LOG.md, W1-C)
VARIANTS: list = [
    Variant("all_5000", ("all",), 5000),            # frozen-recipe budget on the rescue targets
    Variant("all_1500", ("all",), 1500),            # same, shorter: less drift if the first fails tests 3-5
    Variant("late_5000", ("blocks.2", "out"), 5000),  # last block + head only: locality by parameter subset
]


def rescue_targets(ctx: RepairContext):
    """Apply the cluster's best primitive (branch table, class R) to each member's logged decode
    (xi_i, c_i) under the current version. Keep members whose primitive decode leaves B_i.
    Returns (uids (m,), targets (m, 2)) for the kept members (m may be 0)."""
    cell, svc, prim = ctx.cell, ctx.svc, ctx.table.best_primitive
    rows = np.asarray(ctx.cluster.members, dtype=np.int64)
    xi, c = cell.neg.xi[rows], cell.neg.c[rows]
    a = svc.restore_and_decode(ctx.current, svc.flow_state(x=xi.clone(), k=0, c=c.clone()), prim)
    keep = np.array([not bool(cell.in_bad_basin(int(i), a[j:j + 1])[0]) for j, i in enumerate(rows)])
    return np.asarray(ctx.cluster.uids)[keep], a[torch.from_numpy(keep)]


def repair(ctx: RepairContext):
    svc, io = ctx.svc, ctx.io
    uids, targets = rescue_targets(ctx)
    io.log("note", {"d1_cluster": int(ctx.card.cluster), "primitive": ctx.table.best_primitive.name,
                    "n_members": int(len(ctx.cluster.uids)), "n_rescued": int(len(uids))})
    if len(uids) == 0:
        return
    for var in VARIANTS:
        data = svc.rescue_rows(ctx.cell, uids, targets, support=var.support)
        spec = svc.train_spec(params=var.params, objective="bc", steps=var.steps, anchor=var.anchor)
        vid = io.new_version_id()
        res = svc.targeted_train(ctx.cell, ctx.current, vid, spec, data)
        if res.version is None:
            io.log("edit", {"call": "targeted_train", "variant": var.name, "ok": False, "error": res.record.error})
            continue
        if res.patch is not None and not getattr(res.patch, "id", None):
            res.patch.id = io.new_patch_id()
        io.save_patch(res.patch)
        res.version.meta = {**(res.version.meta or {}), "variant": var.name}
        io.log("edit", {"cluster": int(ctx.card.cluster), "variant": var.name, "record": res.record},
               version=res.version.id)
        yield res.version


def run(cell, run, svc: Optional[Services] = None, max_rounds: Optional[int] = None) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run).
    ``max_rounds=1`` is the pilot (one repair, no recursion)."""
    return iterate("D1", cell, run, svc, repair, needs_branch=True, max_rounds=max_rounds)
