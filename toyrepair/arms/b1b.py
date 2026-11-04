"""B1b: the method-map config with the lowest D on seeds 0-4 for the setting (reference/method_map.json). Oracle-selected, optimistic."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from typing import Optional

from .base import ArmOutput, Services, toypref_arm

#: method-map configs that are not runnable toypref conditional arms (SDPO lives on another branch)
_NOT_RUNNABLE_PREFIXES = ("SDPO_",)


@lru_cache(maxsize=None)
def _method_map() -> dict:
    from toyrepair import REPO
    with open(os.path.join(REPO, "reference", "method_map.json")) as fh:
        return json.load(fh)["conditional"]


def best_configs(setting: str, n: int = 1, seeds: str = "values") -> list:
    """The n toypref arm names with the lowest mean D on seeds 0-4 ("values") for ``setting``,
    from reference/method_map.json. Skips the B0 row and configs that the compact repo cannot run
    (SDPO); ties break by name."""
    from toypref.cond.experiment import ARMS
    vals = _method_map()[seeds][setting]
    cand = [(v["tv"], k) for k, v in vals.items()
            if k != "B0_frozen" and k in ARMS and not k.startswith(_NOT_RUNNABLE_PREFIXES)]
    return [k for _, k in sorted(cand)[:n]]


def run(cell, run, svc: Optional[Services] = None, measure_gate: bool = False) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run).

    Trains ``best_configs(cell.setting, 1)[0]`` with toypref ``train_arm`` exactly as
    ``scripts/cond_cell.py`` does. One-shot: no gate decides (the gate check is measured when
    ``run`` asks for it, see ``base.toypref_arm``)."""
    cfg = best_configs(cell.setting, 1)[0]
    return toypref_arm("B1b", cfg, cell, run, svc, measure_gate=measure_gate)
