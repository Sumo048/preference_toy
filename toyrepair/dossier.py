"""Dossier builder (docs/PLAN.md 7.3).

Static part (<= 4000 tokens): the policy interface (2-D action, scalar condition, 20-step
decode, layer names and sizes), the CLI and edit API reference. No ring geometry, no mode count.
Variable part (<= 6000 tokens): the target cluster's failure card, its branch table, the ledger
summary, and the remaining gate checks and cost.

Owner: W1-D.
"""

from __future__ import annotations

from dataclasses import dataclass

STATIC_TOKEN_CAP = 4000
VARIABLE_TOKEN_CAP = 6000


@dataclass(frozen=True)
class Dossier:
    """static: text of the static part; variable: text of the variable part; arm: arm name
    (decides which commands are listed); tokens_static / tokens_variable: estimates."""
    static: str
    variable: str
    arm: str
    tokens_static: int = 0
    tokens_variable: int = 0

    def text(self) -> str:
        return self.static + "\n\n" + self.variable


def estimate_tokens(text: str) -> int:
    """Rough token count (chars / 3.5, rounded up)."""
    return -(-len(text) * 2 // 7)


def static_part(arm: str) -> str:
    raise NotImplementedError


def variable_part(run, session) -> str:
    raise NotImplementedError


def build(run, session) -> Dossier:
    raise NotImplementedError
