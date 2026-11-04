"""Budget counters: decodes, gate checks, CLI calls, tokens, cost (docs/PLAN.md 2, 6.10, 7.2, 7.6).

``Budget`` holds the per-run counters (``runs/<run id>/budget.json``); ``SessionCounter`` holds
the per-session CLI-call and gate-submission counts the CLI enforces. Both persist as JSON with
an atomic replace, so a crash never leaves a half-written file.

This module does not import the oracle, toypref or torch.

Owner: W1-A2.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Optional

# Session limits (7.2)
SESSION_CLI_CALLS = 20
SESSION_GATE_SUBMISSIONS = 3
SESSION_CONTEXT_TOKENS = 150_000
# Run limits (6.10, 7.6)
RUN_COST_CAP_USD = 8.0
NO_ACCEPT_STREAK_STOP = 3
COOLDOWN_ITERATIONS = 2
# Prices, USD per million tokens: input, cache write, cache read, output
PRICES = {
    "opus": {"input": 4.0, "cache_write": 5.0, "cache_read": 0.20, "output": 20.0},
    "sonnet": {"input": 2.0, "cache_write": 2.50, "cache_read": 0.10, "output": 10.0},
}


class BudgetExceeded(RuntimeError):
    """A limit would be exceeded (the caller turns it into a CLI error or a stop rule)."""


def gate_check_cap(n_first_atlas_clusters: int) -> int:
    """3 x (number of clusters in the first atlas + 2)."""
    return 3 * (int(n_first_atlas_clusters) + 2)


def cost_usd(model: str, input: int = 0, cache_write: int = 0, cache_read: int = 0, output: int = 0) -> float:
    """API-price cost of token counts; model "opus" | "sonnet" (or an id containing either)."""
    key = "opus" if "opus" in model else "sonnet" if "sonnet" in model else model
    p = PRICES[key]
    return (input * p["input"] + cache_write * p["cache_write"] + cache_read * p["cache_read"]
            + output * p["output"]) / 1e6


def usage_cost(model: str, usage: Dict[str, int]) -> float:
    """Cost of an Anthropic ``usage`` dict (input_tokens, cache_creation_input_tokens,
    cache_read_input_tokens, output_tokens)."""
    return cost_usd(model, input=int(usage.get("input_tokens", 0) or 0),
                    cache_write=int(usage.get("cache_creation_input_tokens", 0) or 0),
                    cache_read=int(usage.get("cache_read_input_tokens", 0) or 0),
                    output=int(usage.get("output_tokens", 0) or 0))


def _atomic_write_json(path: str, obj: Dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".budget.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _from_dict(cls, d: Dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class SessionLimits:
    cli_calls: int = SESSION_CLI_CALLS
    gate_submissions: int = SESSION_GATE_SUBMISSIONS
    context_tokens: int = SESSION_CONTEXT_TOKENS


@dataclass
class SessionCounter:
    """Per-session counts the CLI enforces (7.2). ``use_call`` / ``use_submission`` raise
    ``BudgetExceeded`` when the limit is reached (the call is not counted then)."""
    session: str = ""
    cli_calls: int = 0
    gate_submissions: int = 0
    limits: SessionLimits = field(default_factory=SessionLimits)

    def use_call(self) -> int:
        if self.cli_calls >= self.limits.cli_calls:
            raise BudgetExceeded(f"session limit: {self.limits.cli_calls} CLI calls used")
        self.cli_calls += 1
        return self.limits.cli_calls - self.cli_calls

    def use_submission(self) -> int:
        if self.gate_submissions >= self.limits.gate_submissions:
            raise BudgetExceeded(f"session limit: {self.limits.gate_submissions} gate submissions used")
        self.gate_submissions += 1
        return self.limits.gate_submissions - self.gate_submissions

    def remaining(self) -> Dict[str, int]:
        return {"cli_calls": self.limits.cli_calls - self.cli_calls,
                "gate_submissions": self.limits.gate_submissions - self.gate_submissions}

    def save(self, path: str) -> None:
        _atomic_write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str) -> "SessionCounter":
        with open(path) as f:
            d = json.load(f)
        lim = SessionLimits(**d.pop("limits", {}))
        return cls(limits=lim, **{k: v for k, v in d.items() if k in ("session", "cli_calls", "gate_submissions")})


@dataclass
class Budget:
    """Per-run counters, persisted in ``runs/<run id>/budget.json``.

    decodes (float, see policy doc); gate_checks; gate_cap; cost_usd; cost_cap_usd;
    sessions; no_accept_streak. Extra counters: tokens {input, cache_write, cache_read,
    output}; decodes_by {source: decodes} (gate, miner, branch, sandbox, ...); cpu_s."""
    decodes: float = 0.0
    gate_checks: int = 0
    gate_cap: int = 0
    cost_usd: float = 0.0
    cost_cap_usd: float = RUN_COST_CAP_USD
    sessions: int = 0
    no_accept_streak: int = 0
    tokens: Dict[str, int] = field(default_factory=lambda: {"input": 0, "cache_write": 0, "cache_read": 0,
                                                            "output": 0})
    decodes_by: Dict[str, float] = field(default_factory=dict)
    cpu_s: float = 0.0

    # ------------------------------------------------------------ counters
    def add_decodes(self, n: float, source: str = "other") -> None:
        self.decodes += float(n)
        self.decodes_by[source] = self.decodes_by.get(source, 0.0) + float(n)

    def gate_checks_left(self) -> Optional[int]:
        """Remaining gate checks, or None while no cap is set (gate_cap 0 = before the first atlas)."""
        return None if self.gate_cap <= 0 else max(self.gate_cap - self.gate_checks, 0)

    def next_gate_check(self) -> int:
        """Reserve one gate check and return its check index (0, 1, ...). Raises
        ``BudgetExceeded`` when the cap is spent."""
        if self.gate_cap > 0 and self.gate_checks >= self.gate_cap:
            raise BudgetExceeded(f"run limit: {self.gate_cap} gate checks used")
        idx = self.gate_checks
        self.gate_checks += 1
        return idx

    def add_usage(self, model: str, input: int = 0, cache_write: int = 0, cache_read: int = 0,
                  output: int = 0) -> float:
        """Add one session's tokens; returns its cost in USD."""
        c = cost_usd(model, input=input, cache_write=cache_write, cache_read=cache_read, output=output)
        for k, v in (("input", input), ("cache_write", cache_write), ("cache_read", cache_read), ("output", output)):
            self.tokens[k] = self.tokens.get(k, 0) + int(v)
        self.cost_usd += c
        return c

    def end_session(self, accepted: bool) -> None:
        self.sessions += 1
        self.no_accept_streak = 0 if accepted else self.no_accept_streak + 1

    def cost_spent(self) -> bool:
        return self.cost_usd >= self.cost_cap_usd

    def stop_reason(self, n_open_clusters: Optional[int] = None) -> Optional[str]:
        """The first budget stop rule that holds (6.10), or None. ``n_open_clusters`` adds the
        "no cluster is open" rule."""
        if n_open_clusters is not None and n_open_clusters == 0:
            return "no_open_cluster"
        if self.no_accept_streak >= NO_ACCEPT_STREAK_STOP:
            return "no_accept_streak"
        if self.gate_cap > 0 and self.gate_checks >= self.gate_cap:
            return "gate_cap"
        if self.cost_spent():
            return "cost_cap"
        return None

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        _atomic_write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str) -> "Budget":
        with open(path) as f:
            return _from_dict(cls, json.load(f))

    @classmethod
    def load_or_new(cls, path: str, **kw) -> "Budget":
        return cls.load(path) if os.path.exists(path) else cls(**kw)
