"""Ledger: append-only JSONL per run, ``$TOYPREF_RUNS/repair/<run id>/ledger.jsonl`` (docs/PLAN.md 6.9).

One record per session, edit, gate check, and tool call. Never rewrite or delete a record; mark
invalid records with a later ``invalidate`` record whose data is
``{"line": <0-based line number of the record>, "reason": "..."}`` (or ``{"run": true}`` to
invalidate the whole run).

Record payload conventions (what ``summary`` reads; every other key is free-form):
  edit:    data = ``edits.EditRecord`` as a dict (call, args, parent, version, ok, error, ...)
  gate:    data = ``gate.GateResult.ledger_data()`` (holds "surgeon_view"); ``version`` = new id
  session: data = {"summary": one line, "accepted": bool, "cluster": id, ...}

``summary`` uses only ``surgeon_view`` from gate records (pass/fail and test 1's number), never
the other gate numbers, so it is safe for the dossier.

This module does not import the oracle, toypref or torch.

Owner: W1-A2.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, Iterator, List, Optional

KINDS = ("run", "atlas", "triage", "session", "tool", "edit", "gate", "quick_check", "branch",
         "invalidate", "final", "note")
SUMMARY_TOKEN_CAP = 2000
#: chars per token used for caps here; W1-D's dossier measures the real ratio and may pass its own
CHARS_PER_TOKEN = 3.5
RECENT_ATTEMPTS = 6


def now_ts() -> str:
    """ISO-8601 UTC timestamp with microseconds."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(x):
    """Replace nan/inf floats by None (strict JSON)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


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

    # ---------------------------------------------------------------- writing
    def append(self, rec: LedgerRecord) -> None:
        """Append one JSON line with flush + fsync."""
        if rec.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {rec.kind!r}")
        d = {"kind": rec.kind, "run": rec.run, "ts": rec.ts, "session": rec.session,
             "version": rec.version, "data": rec.data}
        line = json.dumps(_clean(json.loads(json.dumps(d, default=_json_default))), allow_nan=False)
        if "\n" in line:  # pragma: no cover  (json.dumps never emits raw newlines)
            raise ValueError("record serialises to more than one line")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write(self, kind: str, run: str, data: Optional[Dict] = None, session: Optional[str] = None,
              version: Optional[str] = None) -> LedgerRecord:
        """Build a record stamped now and append it."""
        rec = LedgerRecord(kind=kind, run=run, ts=now_ts(), data=dict(data or {}), session=session,
                           version=version)
        self.append(rec)
        return rec

    def invalidate(self, run: str, line: Optional[int], reason: str) -> LedgerRecord:
        """Mark record ``line`` (0-based) invalid, or the whole run if line is None."""
        data = {"reason": reason, **({"run": True} if line is None else {"line": int(line)})}
        return self.write("invalidate", run, data)

    # ---------------------------------------------------------------- reading
    def _lines(self) -> Iterator[Dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    # a torn last line after a crash: skip it, keep every complete record
                    continue

    def records(self, kind: Optional[str] = None) -> Iterator[LedgerRecord]:
        for d in self._lines():
            if kind is None or d.get("kind") == kind:
                yield LedgerRecord(kind=d["kind"], run=d["run"], ts=d["ts"], data=d.get("data") or {},
                                   session=d.get("session"), version=d.get("version"))

    def invalid_lines(self) -> Dict[int, str]:
        """{0-based line: reason} of invalidated records; key -1 means the whole run."""
        out = {}
        for d in self._lines():
            if d.get("kind") == "invalidate":
                data = d.get("data") or {}
                out[-1 if data.get("run") else int(data.get("line", -2))] = data.get("reason", "")
        return out

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.records():
            out[r.kind] = out.get(r.kind, 0) + 1
        return out

    # ---------------------------------------------------------------- summary
    def summary(self, max_tokens: int = SUMMARY_TOKEN_CAP, chars_per_token: float = CHARS_PER_TOKEN,
                recent: int = RECENT_ATTEMPTS) -> str:
        """Recent attempts and all accepted edits for the dossier (no oracle numbers beyond
        gate pass/fail and test 1's number). Accepted edits are listed first and in full; then
        the most recent rejected attempts, newest first, until the cap. States how many rows
        were cut."""
        recs = list(self._lines())
        bad = self.invalid_lines()
        edits = {}
        for i, d in enumerate(recs):
            if d.get("kind") == "edit" and i not in bad and d.get("version"):
                edits[d["version"]] = d.get("data") or {}
        accepted, rejected = [], []
        for i, d in enumerate(recs):
            if d.get("kind") != "gate" or i in bad:
                continue
            sv = (d.get("data") or {}).get("surgeon_view")
            if not sv:
                continue
            line = _attempt_line(d, sv, edits.get(d.get("version"), {}))
            (accepted if sv.get("accepted") else rejected).append(line)
        n_sessions = sum(1 for i, d in enumerate(recs) if d.get("kind") == "session" and i not in bad)
        failed_edits = [d for i, d in enumerate(recs) if d.get("kind") == "edit" and i not in bad
                        and not (d.get("data") or {}).get("ok", True)]
        head = [f"Ledger summary: {n_sessions} sessions, {len(accepted)} accepted edits, "
                f"{len(rejected)} rejected gate checks, {len(failed_edits)} edit calls refused."]
        body = ["Accepted edits (oldest first):"] + (accepted or ["  none"])
        rej = list(reversed(rejected))[:recent]
        body += [f"Recent rejected attempts (newest first, {len(rej)} of {len(rejected)}):"] + (rej or ["  none"])
        cap = int(max_tokens * chars_per_token)
        out, cut = [], 0
        for ln in head + body:
            if sum(len(x) + 1 for x in out) + len(ln) + 1 > cap - 40:
                cut += 1
                continue
            out.append(ln)
        if cut:
            out.append(f"[{cut} rows cut to fit the {max_tokens}-token cap]")
        return "\n".join(out)


def _attempt_line(d: Dict, sv: Dict, edit: Dict) -> str:
    passed = sv.get("passed") or {}
    fails = [k for k, v in passed.items() if not v]
    ci = sv.get("target_fall_ci") or [None, None]
    fmt = lambda x: "nan" if x is None else f"{x:.4f}"  # noqa: E731
    call = edit.get("call", "?")
    args = edit.get("args") or {}
    arg_s = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3] if not isinstance(v, (dict, list)))
    who = f"session {d.get('session')}" if d.get("session") else "code"
    verdict = "ACCEPT" if sv.get("accepted") else "reject"
    s = (f"  {d.get('version')} <- {edit.get('parent', '?')}: {call}({arg_s[:80]}) [{who}] {verdict}; "
         f"target_fall {fmt(sv.get('target_fall'))} CI [{fmt(ci[0])}, {fmt(ci[1])}]")
    if fails:
        s += "; failed: " + ",".join(fails)
    return s
