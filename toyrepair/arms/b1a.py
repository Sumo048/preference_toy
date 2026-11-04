"""B1a: the frozen recipe (toypref R_hopB_bsup_neg_AMdist, 5000 steps) on all negatives in one training run; global gate scope."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError
