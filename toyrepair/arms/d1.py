"""D1: branch-and-distill: the A3 loop with code in place of the LLM; best primitive -> rescue targets for members and basin supports -> targeted_train (BC + locality penalty); fixed list of 3 variants (set in Phase 2)."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError


#: the fixed variant list (Phase 2 sets it; PREREG freezes it)
VARIANTS: list = []
