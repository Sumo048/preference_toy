"""R0: iterative sampling-time controller: per class-R cluster, a logistic detector on flow states at k_commit (decodes ending in the cluster's basins vs the rest, no labels) runs the best primitive in code when it fires; gate applies."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError
