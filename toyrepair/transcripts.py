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


# ----------------------------------------------------------------------------- W1-D: per-session reader

CONTEXT_LIMIT = 150_000      # docs/PLAN.md 7.2: session context cap, checked from the transcript
#: extra isolation patterns (7.5): the oracle's classes and private fields, toypref setting files
ISOLATION_PATTERNS = DEFAULT_FORBIDDEN + (r"CondMixture", r"\._oracle\b", r"toypref\.cond\b",
                                          r"cond/experiment\.py", r"cond/task\.py", r"reference/method_map")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                out.append(_text_of(b.get("content")))
    return "\n".join(out)


def first_user_text(path) -> str:
    for ev in _read_jsonl(Path(path)):
        if ev.get("type") == "user":
            return _text_of((ev.get("message") or {}).get("content"))
    return ""


def final_assistant_text(path) -> str:
    """Text of the last assistant message that has text (the subagent's final report)."""
    last = ""
    for ev in _read_jsonl(Path(path)):
        if ev.get("type") == "assistant":
            t = _text_of((ev.get("message") or {}).get("content"))
            if t.strip():
                last = t.strip()
    return last


def find_subagent_for_token(token: str, root: Path = PROJECTS) -> Optional[str]:
    """Newest subagent transcript whose first user message holds ``token``."""
    for p in reversed(find_subagent_transcripts(root)):
        try:
            if token in first_user_text(p):
                return str(p)
        except OSError:
            continue
    return None


def context_per_request(path) -> List[int]:
    """Context size of each API request: input + cache write + cache read tokens."""
    return [int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
            + int(u.get("cache_read_input_tokens") or 0)
            for _, u in _assistant_usages(Path(path)).values()]


def isolation_audit(path, run_id: str = "", other_runs: Iterable[str] = ()) -> List[str]:
    """Section 7.5: oracle paths, toypref setting files, and other runs' ids or directories.
    The run's own id is allowed."""
    pats = list(ISOLATION_PATTERNS)
    for rid in other_runs:
        if rid and rid != run_id:
            pats.append(r"repair/" + re.escape(rid) + r"(?![\w.\-])")
            if len(rid) >= 6:     # bare ids only when distinctive (short ids give false hits)
                pats.append(r"(?<![\w.\-])" + re.escape(rid) + r"(?![\w.\-])")
    hits, seen = [], set()
    for h in audit_transcript(path, pats):      # one hit per transcript line
        line = h.split(":", 1)[0]
        if line not in seen:
            seen.add(line)
            hits.append(h)
    return hits


def session_report(path, run_id: str = "", other_runs: Iterable[str] = (), expected_agent: str = "") -> Dict:
    """Usage, cost, context check, audit and final text of one subagent session transcript."""
    u = usage_from_transcript(path)
    meta = subagent_meta(path)
    ctx = context_per_request(path)
    hits = isolation_audit(path, run_id, other_runs)
    return {"transcript": str(path), "agent_type": meta.get("agentType"), "model_meta": meta.get("model"),
            "agent_ok": (not expected_agent) or meta.get("agentType") in (None, expected_agent),
            "tokens": {f: u.total(f) for f in FIELDS}, "models": sorted(u.models),
            "n_requests": sum(m.n_messages for m in u.models.values()),
            "cost_usd": round(cost(u), 6), "max_context": max(ctx) if ctx else 0,
            "context_exceeded": bool(ctx) and max(ctx) > CONTEXT_LIMIT,
            "audit_hits": hits, "final_text": final_assistant_text(path)[:2000]}


def chars_per_token(path) -> Dict:
    """Measure characters per token on a real transcript: for each tool result between two API
    requests, its tokens are (context of request n+1) - (context of request n) - (output tokens
    of request n). Uses requests whose growth comes from exactly one tool result, and sums
    chars and tokens over them (results of at least 200 chars, to damp per-message overhead)."""
    reqs = []        # ordered (msg id, usage)
    seen = set()
    events = list(_read_jsonl(Path(path)))
    order = []       # sequence of ("a", id) and ("r", chars)
    for ev in events:
        if ev.get("type") == "assistant":
            msg = ev.get("message") or {}
            mid = msg.get("id")
            if mid and mid not in seen and msg.get("usage"):
                seen.add(mid)
                order.append(("a", mid))
        elif ev.get("type") == "user":
            content = (ev.get("message") or {}).get("content")
            if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result"
                                                 for b in content):
                order.append(("r", len(_text_of(content))))
    usages = _assistant_usages(Path(path))
    chars = toks = n = 0
    samples = []
    for i in range(len(order) - 2):
        if order[i][0] == "a" and order[i + 1][0] == "r" and order[i + 2][0] == "a":
            u0, u1 = usages[order[i][1]][1], usages[order[i + 2][1]][1]
            c0 = sum(int(u0.get(f) or 0) for f in FIELDS[:3])
            c1 = sum(int(u1.get(f) or 0) for f in FIELDS[:3])
            t = c1 - c0 - int(u0.get("output_tokens") or 0)
            ch = order[i + 1][1]
            if ch >= 200 and t > 0:
                chars += ch
                toks += t
                n += 1
                samples.append((ch, t))
    return {"chars": chars, "tokens": toks, "n": n, "ratio": round(chars / toks, 3) if toks else None,
            "samples": samples[:20]}


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
    r = sub.add_parser("report", help="usage, cost, context check and isolation audit of one session transcript")
    r.add_argument("path")
    r.add_argument("--run", default="")
    r.add_argument("--other", action="append", default=[])
    c = sub.add_parser("ratio", help="measure characters per token on a transcript")
    c.add_argument("path")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(json.dumps(session_report(args.path, args.run, args.other), indent=1))
        return
    if args.cmd == "ratio":
        print(json.dumps(chars_per_token(args.path)))
        return
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
