"""B1a: the frozen recipe (toypref R_hopB_bsup_neg_AMdist, 5000 steps) on all negatives in one training run; global gate scope."""

from __future__ import annotations

from typing import Optional

from .base import ArmOutput, Services, toypref_arm

CONFIG = "R_hopB_bsup_neg_AMdist"


def run(cell, run, svc: Optional[Services] = None, measure_gate: bool = False) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run).

    One toypref training run, bit-identical to ``scripts/cond_cell.py --arms R_hopB_bsup_neg_AMdist``.
    The result is kept whatever the gate says; ``measure_gate`` logs one global-scope check."""
    return toypref_arm("B1a", CONFIG, cell, run, svc, measure_gate=measure_gate)
