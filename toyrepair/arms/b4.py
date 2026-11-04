"""B4: iterative find-then-fine-tune: per cluster, the frozen recipe on that cluster's negatives only, with the locality penalty; cluster gate scope."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError
