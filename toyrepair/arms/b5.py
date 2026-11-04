"""B5: gated recipe selection: the gate checks the 3 method-map configs with the lowest D on seeds 0-4 for the setting; keep the best accepted one."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError
