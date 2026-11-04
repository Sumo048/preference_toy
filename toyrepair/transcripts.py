"""Token usage, cost, and the isolation audit from Claude Code transcripts (docs/PLAN.md 6.11, 7.5, 7.6).

Stdlib only. Never imports toypref.

Transcript layout (verified on a cloud session, 2026-11-04):

    ~/.claude/projects/<project-dir>/<session-uuid>.jsonl                     main agent
    ~/.claude/projects/<project-dir>/<session-uuid>/subagents/agent-<id>.jsonl  one subagent invocation
    ~/.claude/projects/<project-dir>/<session-uuid>/subagents/agent-<id>.meta.json  {agentType, description, model, ...}

Each line is one JSON event. Assistant events have ``type == "assistant"``, ``requestId``,
``message.id``, ``message.model`` and ``message.usage`` with the fields ``input_tokens``,
``cache_creation_input_tokens``, ``cache_read_input_tokens``, ``output_tokens``. One API response
can be written as several events (one per content block) that repeat the same usage, so usage is
deduplicated by ``message.id`` (fallback ``requestId``), keeping the largest output count.

A fork subagent transcript starts with a ``{"type": "fork-context-ref", "parentSessionId": ...}``
line and repeats the parent's spawning message; assistant messages whose id also appears in the
parent transcript are excluded, so they are not double counted.

``spend`` appends the CUMULATIVE totals of the main transcript at call time. Consumers take the
latest record per (session, transcript), not the sum of records.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECTS = Path(os.path.expanduser("~/.claude/projects"))
REPO = Path(__file__).resolve().parent.parent
SPEND_FILE = REPO / "results" / "spend.jsonl"

# USD per million tokens: (input, cache write, cache read, output). docs/PLAN.md Section 2.
PRICES = {
    "opus": (4.0, 5.0, 0.20, 20.0),
    "sonnet": (2.0, 2.50, 0.10, 10.0),
    "haiku": (1.0, 1.25, 0.05, 5.0),     # not in the plan: Sonnet / 2 (assumption)
}
FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def price_family(model: str) -> (str, bool):
    """(family, known). Unknown models are priced as Opus and flagged."""
    m = (model or "").lower()
    for fam in ("opus", "sonnet", "haiku"):
        if fam in m:
            return fam, True
    return "opus", False


@dataclass
class ModelUsage:
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    n_messages: int = 0

    def add(self, u: Dict) -> None:
        for f in FIELDS:
            setattr(self, f, getattr(self, f) + int(u.get(f) or 0))
        self.n_messages += 1


@dataclass
class Usage:
    """Token usage of one transcript, per model id."""
    path: str
    models: Dict[str, ModelUsage] = field(default_factory=dict)
    unknown_models: List[str] = field(default_factory=list)

    def total(self, f: str) -> int:
        return sum(getattr(m, f) for m in self.models.values())

    def to_dict(self) -> Dict:
        return {"path": self.path, "models": {k: asdict(v) for k, v in self.models.items()},
                "unknown_models": self.unknown_models, "cost_usd": round(cost(self), 6)}


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue          # a line being written right now


def _assistant_usages(path: Path) -> Dict[str, (str, Dict)]:
    """message id -> (model, usage), deduplicated (largest output_tokens wins)."""
    out: Dict[str, (str, Dict)] = {}
    for ev in _read_jsonl(path):
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message") or {}
        u = msg.get("usage")
        if not u:
            continue
        key = msg.get("id") or ev.get("requestId") or ev.get("uuid")
        prev = out.get(key)
        if prev is None or int(u.get("output_tokens") or 0) >= int(prev[1].get("output_tokens") or 0):
            out[key] = (msg.get("model") or "unknown", u)
    return out


def _fork_parent(path: Path) -> Optional[Path]:
    for ev in _read_jsonl(path):
        if ev.get("type") == "fork-context-ref" and ev.get("parentSessionId"):
            # .../<session>/subagents/agent-x.jsonl -> .../<parentSession>.jsonl
            cand = path.parent.parent.parent / f"{ev['parentSessionId']}.jsonl"
            return cand if cand.exists() else None
        break
    return None


def usage_from_transcript(path, exclude_ids: Optional[set] = None) -> Usage:
    """Sum deduplicated usage of one transcript per model. For a fork transcript, messages that
    also appear in the parent transcript are excluded automatically."""
    path = Path(path)
    msgs = _assistant_usages(path)
    excl = set(exclude_ids or ())
    parent = _fork_parent(path)
    if parent is not None:
        excl |= set(_assistant_usages(parent))
    usage = Usage(path=str(path))
    for key, (model, u) in msgs.items():
        if key in excl or model == "<synthetic>":
            continue
        usage.models.setdefault(model, ModelUsage()).add(u)
        if not price_family(model)[1] and model not in usage.unknown_models:
            usage.unknown_models.append(model)
    return usage


def cost(usage: Usage) -> float:
    """Estimated API-price cost in USD (docs/PLAN.md Section 2 prices)."""
    total = 0.0
    for model, m in usage.models.items():
        p_in, p_cw, p_cr, p_out = PRICES[price_family(model)[0]]
        total += (m.input_tokens * p_in + m.cache_creation_input_tokens * p_cw
                  + m.cache_read_input_tokens * p_cr + m.output_tokens * p_out) / 1e6
    return total


def find_main_transcript(root: Path = PROJECTS, cwd: Optional[str] = None) -> Optional[Path]:
    """Newest *.jsonl directly inside a project dir. Prefers the project dir of `cwd` (Claude
    Code encodes the path by replacing '/' and '.' with '-')."""
    root = Path(root)
    cwd = cwd or os.getcwd()
    enc = re.sub(r"[/.]", "-", str(Path(cwd).resolve()))
    preferred = root / enc
    dirs = [preferred] if preferred.is_dir() else [d for d in root.glob("*") if d.is_dir()]
    files = [f for d in dirs for f in d.glob("*.jsonl")]
    if not files and preferred.is_dir():
        files = list(root.glob("*/*.jsonl"))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def find_subagent_transcripts(root: Path = PROJECTS) -> List[Path]:
    """All subagent transcripts: ~/.claude/projects/*/*/subagents/*.jsonl, oldest first."""
    return sorted(Path(root).glob("*/*/subagents/*.jsonl"), key=lambda f: f.stat().st_mtime)


def subagent_meta(path) -> Dict:
    """The sibling .meta.json of a subagent transcript ({agentType, description, model, ...})."""
    m = Path(path).with_suffix(".meta.json")
    try:
        return json.loads(m.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


DEFAULT_FORBIDDEN = (r"toyrepair/oracle/", r"toyrepair\.oracle", r"src/toypref/", r"toypref/cond/")


def audit_transcript(path, forbidden_patterns: Iterable[str] = DEFAULT_FORBIDDEN) -> List[str]:
    """Regex hits of forbidden patterns (oracle paths, toypref files, other run ids) in the
    transcript text. Returns one 'line <n>: <pattern>: <excerpt>' string per hit; empty = clean.
    Callers add other runs' ids to `forbidden_patterns` (re.escape them)."""
    pats = [re.compile(p) for p in forbidden_patterns]
    hits = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            for p in pats:
                m = p.search(line)
                if m:
                    s = max(0, m.start() - 40)
                    hits.append(f"line {n}: {p.pattern}: {line[s:m.end() + 40].strip()}")
    return hits


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def spend(session: str, dry_run: bool = False, transcript: Optional[str] = None,
          out: Path = SPEND_FILE) -> Dict:
    path = Path(transcript) if transcript else find_main_transcript()
    if path is None:
        raise SystemExit("no main transcript found under ~/.claude/projects")
    u = usage_from_transcript(path)
    rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "session": session, "transcript": str(path), "git_head": _git_head(),
           "models": {k: asdict(v) for k, v in u.models.items()},
           "unknown_models_priced_as_opus": u.unknown_models,
           "cost_usd": round(cost(u), 4), "kind": "main", "cumulative": True}
    if not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m toyrepair.transcripts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spend", help="append this session's main-agent cumulative cost to results/spend.jsonl")
    s.add_argument("--session", required=True)
    s.add_argument("--transcript", default=None, help="override the auto-detected main transcript")
    s.add_argument("--dry-run", action="store_true")
    u = sub.add_parser("usage", help="print usage and cost of one transcript")
    u.add_argument("path")
    a = sub.add_parser("audit", help="grep a transcript for forbidden patterns")
    a.add_argument("path")
    a.add_argument("--pattern", action="append", default=[])
    sub.add_parser("subagents", help="list subagent transcripts with type and cost")
    args = ap.parse_args(argv)
    if args.cmd == "spend":
        print(json.dumps(spend(args.session, args.dry_run, args.transcript)))
    elif args.cmd == "usage":
        print(json.dumps(usage_from_transcript(args.path).to_dict(), indent=1))
    elif args.cmd == "audit":
        hits = audit_transcript(args.path, list(DEFAULT_FORBIDDEN) + args.pattern)
        print("\n".join(hits) if hits else "clean")
        sys.exit(1 if hits else 0)
    elif args.cmd == "subagents":
        for p in find_subagent_transcripts():
            meta = subagent_meta(p)
            print(f"{p}\t{meta.get('agentType', '?')}\t${cost(usage_from_transcript(p)):.4f}")


if __name__ == "__main__":
    main()
