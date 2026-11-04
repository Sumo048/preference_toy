"""Shared arm types and the code-arm registry. Owner: W1-C."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

#: arm name -> module under toyrepair.arms
CODE_ARMS: Dict[str, str] = {"B0": "b0", "B1a": "b1a", "B1b": "b1b", "B4": "b4", "B5": "b5",
                             "R0": "r0", "D1": "d1"}


@dataclass
class ArmOutput:
    """arm; final version id; accepted version ids in order; gate_checks used; decodes used;
    stop rule (or "single" for one-shot arms); meta (e.g. the toypref config chosen)."""
    arm: str
    version: str
    accepted: List[str] = field(default_factory=list)
    gate_checks: int = 0
    decodes: float = 0.0
    stop: str = "single"
    meta: Dict = field(default_factory=dict)


def get(name: str) -> Callable:
    """The ``run`` function of a code arm."""
    return importlib.import_module(f"toyrepair.arms.{CODE_ARMS[name]}").run
