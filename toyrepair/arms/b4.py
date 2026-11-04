"""B4: iterative find-then-fine-tune: per cluster, the frozen recipe on that cluster's negatives only, with the locality penalty; cluster gate scope."""

from __future__ import annotations

from typing import Optional

from .base import ArmOutput, RepairContext, Services, iterate

#: the frozen recipe as a targeted_train call: hopB targets with basin supports, all parameters,
#: 5000 steps, vote-masked density anchor (the locality penalty)
RULE, SUPPORT, STEPS, ANCHOR, PARAMS = "hopB", "bsup", 5000, "dist", ("all",)


def repair(ctx: RepairContext):
    """One candidate per cluster: the frozen recipe trained from the current version on the
    cluster's negatives only (B4 has no variants, so a second check would repeat the first)."""
    svc, io = ctx.svc, ctx.io
    uids = ctx.cluster.uids
    data = svc.toypref_rows(ctx.cell, uids, rule=RULE, support=SUPPORT)
    spec = svc.train_spec(params=PARAMS, objective="bc", steps=STEPS, anchor=ANCHOR)
    vid = io.new_version_id()
    res = svc.targeted_train(ctx.cell, ctx.current, vid, spec, data, patch_id=io.new_patch_id())
    if res.version is None:
        io.log("edit", {"call": "targeted_train", "ok": False, "error": res.record.error,
                        "cluster": int(ctx.card.cluster)})
        return
    io.save_patch(res.patch)
    res.version.meta = {**(res.version.meta or {}), "variant": "frozen_recipe_cluster"}
    io.log("edit", {"cluster": int(ctx.card.cluster), "variant": "frozen_recipe_cluster",
                    "record": res.record}, version=res.version.id)
    yield res.version


def run(cell, run, svc: Optional[Services] = None, max_rounds: Optional[int] = None,
        target_cluster: Optional[int] = None) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run).
    ``max_rounds=1`` is the pilot (one repair, no recursion)."""
    return iterate("B4", cell, run, svc, repair, needs_branch=False, max_rounds=max_rounds,
                   target_cluster=target_cluster)
