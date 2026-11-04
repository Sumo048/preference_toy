"""Fit direction edits on the frozen recipe's data (hopB targets, basin support, vote-masked anchor).

Only the edit's own parameters train; pi_ref's weights stay fixed. The loss is the frozen recipe's
loss (experiment.step_fn for R_hopB_bsup_neg_AMdist), so B1a and these edits differ only in the
structure of the change.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn

import lib
from lib import E, TG

ARM = E.ARMS["R_hopB_bsup_neg_AMdist"]


def ctx_rows(cell: lib.Cell, steps=2000, lr=1e-2, batch=256):
    """(Ctx, Rows) for the frozen recipe on this cell; rows cached on disk."""
    ctx = E.Ctx(ref=cell.ref, seed=cell.seed, ed=cell.ed, pool=None, anchor=cell.anchor,
                steps=steps, lr=lr, batch=batch)
    cp = os.path.join(os.environ["TOYPREF_RUNS"], "E2cache", f"rows_{cell.layout}_s{cell.seed}_n{cell.npm}.pt")
    if os.path.exists(cp):
        rows = torch.load(cp, weights_only=False)
    else:
        rows = TG.build(cell.ref, E.MIX, cell.seed, cell.ed.uid, cell.ed.c, cell.ed.xi, cell.ed.a,
                        ARM.rule, ARM.support, d0=4 * E.MIX.mode_std, forget=None)
        torch.save(rows, cp)
    cv = os.path.join(os.environ["TOYPREF_RUNS"], "E2cache", f"vote_{cell.layout}_s{cell.seed}_n{cell.npm}.pt")
    if os.path.exists(cv):
        ctx._vote = torch.load(cv, weights_only=False)
    else:
        E.get_vote(ctx)
        bank, keep, diag = ctx._vote
        torch.save((None, keep, diag), cv)     # bank not needed for the anchor mask
    return ctx, rows


class Trainable(nn.Module):
    """Holds the edit parameters as nn.Parameters; build(params) -> list of lib.Edit."""

    def __init__(self, params: dict, build):
        super().__init__()
        self.p = nn.ParameterDict({k: nn.Parameter(v.clone().float()) for k, v in params.items()})
        self.build = build


def train_edit(cell: lib.Cell, tr: Trainable, steps=2000, lr=1e-2, loss_kind="recipe", log=None,
               wd=0.0):
    """Train tr's parameters with the frozen recipe's loss on a policy pi_ref + edits."""
    ctx, rows = ctx_rows(cell, steps=steps, lr=lr)
    fn = E.step_fn(ARM, ctx, rows)
    torch.manual_seed(cell.seed)
    g = E.gen(cell.seed, E.P_TRAIN)
    opt = torch.optim.Adam(tr.parameters(), lr=lr, weight_decay=wd)
    for step in range(steps):
        lr_s = E.cosine(step, steps, min(200, steps // 10), lr, 0.1 * lr) if hasattr(E, "cosine") else lr
        for pg in opt.param_groups:
            pg["lr"] = lr_s
        pol = lib.EditedFlow(cell.ref, tr.build(tr.p))
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
        opt.step()
        if log is not None and (step % 500 == 0 or step == steps - 1):
            log(f"  step {step} loss {float(loss):.4f}")
    with torch.no_grad():
        return [e for e in tr.build({k: v.detach() for k, v in tr.p.items()})]
