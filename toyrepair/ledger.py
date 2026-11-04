"""Ledger: append-only JSONL per run, ``$TOYPREF_RUNS/repair/<run id>/ledger.jsonl`` (docs/PLAN.md 6.9).

One record per session, edit, gate check, and tool call. Never rewrite or delete a record; mark
invalid records with a later ``invalidate`` record.

Owner: W1-A2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

KINDS = ("run", "atlas", "triage", "session", "tool", "edit", "gate", "quick_check", "branch",
         "invalidate", "final", "note")
SUMMARY_TOKEN_CAP = 2000


@dataclass
class LedgerRecord:
    """kind in KINDS; run: run id; ts: ISO-8601 UTC; session: session id or None;
    version: version id or None; data: JSON-safe payload (gate records hold every number)."""
    kind: str
    run: str
    ts: str
    data: Dict = field(default_factory=dict)
    session: Optional[str] = None
    version: Optional[str] = None


class Ledger:
    def __init__(self, path: str):
        self.path = path

    def append(self, rec: LedgerRecord) -> None:
        """Append one JSON line with flush + fsync."""
        raise NotImplementedError

    def records(self, kind: Optional[str] = None) -> Iterator[LedgerRecord]:
        raise NotImplementedError

    def summary(self, max_tokens: int = SUMMARY_TOKEN_CAP) -> str:
        """Recent attempts and all accepted edits for the dossier (no oracle numbers beyond
        gate pass/fail and test 1's number)."""
        raise NotImplementedError
