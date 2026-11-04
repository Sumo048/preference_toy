"""B5: gated recipe selection: the gate checks the 3 method-map configs with the lowest D on seeds 0-4 for the setting; keep the best accepted one."""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import (ArmOutput, RunIO, Services, first_atlas_uids, gate_record, real_services, toypref_train,
                   version_from_net)

#: top 3 method-map configs per setting on seeds 0-4 (docs/SESSION_PROMPTS.md W1-C; frozen)
CONFIGS: Dict[str, List[str]] = {
    "k1": ["R_hopB_bsup_neg_AMdist", "NPO_b10_neg_AMdist", "PSEUDO_vote_dist"],
    "k2": ["R_hopB_bsup_mine_neg_AMmap", "R_hopB_bsup_neg_AMdist", "NPO_b10_neg_AL"],
    "k4": ["R_hopB_bsup_mine_neg", "R_hopB_bsup_neg_AMdist", "R_hopB_bsup_mine_neg_AMmap"],
    "pairs": ["R_hopB_bsup_mine_neg", "R_hopB_bsup_neg_AMdist", "R_hopB_bsup_mine_neg_AMmap"],
    "b3p1": ["R_hopB_bsup_mine_neg", "R_hopB_bsup_neg_AMdist", "R_hopB_bsup_mine_neg_AMmap"],
    "contig": ["R_hopB_bsup_neg_AMdist", "NPO_b10_neg_AMdist", "PSEUDO_vote_dist"],
    "alt": ["R_hop_bsup_neg", "KTO_b10_neg_AMmap", "NPO_b10_neg_AL"],
}
#: gate number used to rank accepted candidates (lower is better); falls back to -target_fall
RANK_KEY = "D_new"


def _rank(res) -> float:
    n = res.numbers or {}
    return float(n[RANK_KEY]) if RANK_KEY in n else -float(res.target_fall)


def run(cell, run, svc: Optional[Services] = None, train_fn=None) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run).

    Trains each of the setting's 3 configs on all negatives (toypref, as cond_cell), checks each
    with the gate in global scope (target = union of the first atlas's clusters), and keeps the
    accepted candidate with the lowest gate D(new). No accept: the run keeps pi_ref."""
    svc = svc or real_services()
    train_fn = train_fn or toypref_train
    io = RunIO(run, svc)
    base = svc.base_version(cell)
    cfgs = CONFIGS[cell.setting]
    io.log("run", {"arm": "B5", "cell": cell.key, "configs": cfgs})
    uids = first_atlas_uids(cell, base, svc, io)
    out = ArmOutput(arm="B5", version=base.id, final=base, meta={"configs": cfgs, "checks": []})
    best = None
    for i, cfg in enumerate(cfgs):
        pol, info = train_fn(cell, cfg)
        ver = version_from_net(io, base, pol.net, "targeted_train",
                               {"toypref_config": cfg, "source": "toypref.train_arm"},
                               meta={"variant": cfg})
        res = svc.gate_check(cell, base, base, ver, uids, check_index=out.gate_checks, scope="global")
        out.gate_checks += 1
        io.log("gate", {"toypref_config": cfg, **gate_record(res)}, version=ver.id)
        out.meta["checks"].append({"config": cfg, "version": ver.id, "accepted": bool(res.accepted),
                                   "rank": _rank(res)})
        if res.accepted and (best is None or _rank(res) < best[0]):
            best = (_rank(res), ver, cfg)
    if best is not None:
        out.version, out.final, out.accepted = best[1].id, best[1], [best[1].id]
        out.meta["toypref_config"] = best[2]
        out.stop = "accepted"
    else:
        out.stop = "no_accept"
    io.log("note", {"stop": out.stop, **out.to_json()}, version=out.version)
    return out
