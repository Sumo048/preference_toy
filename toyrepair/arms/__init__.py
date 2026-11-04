"""Code arms (docs/PLAN.md 9.2): B0, B1a, B1b, B4, B5, R0, D1. LLM arms A1-A3 run through
``toyrepair.orchestrate``.

Every arm module exposes ``run(cell, run) -> ArmOutput`` (``run`` is a ``toyrepair.orchestrate.Run``).
"""

from __future__ import annotations

from .base import ArmOutput, CODE_ARMS, get

__all__ = ["ArmOutput", "CODE_ARMS", "get"]
