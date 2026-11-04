"""B0: pi_ref, no change."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError
