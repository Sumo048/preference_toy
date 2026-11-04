"""B1b: the method-map config with the lowest D on seeds 0-4 for the setting (reference/method_map.json). Oracle-selected, optimistic."""

from __future__ import annotations

from .base import ArmOutput


def run(cell, run) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run)."""
    raise NotImplementedError


def best_configs(setting: str, n: int = 1, seeds: str = "values") -> list:
    """The n toypref arm names with the lowest mean D on seeds 0-4 ("values") for ``setting``,
    from reference/method_map.json."""
    raise NotImplementedError
