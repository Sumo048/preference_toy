"""Budget counters: decodes, gate checks, CLI calls, tokens, cost (docs/PLAN.md 2, 6.10, 7.2, 7.6).

Owner: W1-A2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def gate_check_cap(n_first_atlas_clusters: int) -> int:
    """3 x (number of clusters in the first atlas + 2)."""
    return 3 * (int(n_first_atlas_clusters) + 2)


def cost_usd(model: str, input: int = 0, cache_write: int = 0, cache_read: int = 0, output: int = 0) -> float:
    """API-price cost of token counts; model "opus" | "sonnet" (or an id containing either)."""
    key = "opus" if "opus" in model else "sonnet" if "sonnet" in model else model
    p = PRICES[key]
    return (input * p["input"] + cache_write * p["cache_write"] + cache_read * p["cache_read"]
            + output * p["output"]) / 1e6


@dataclass
class SessionLimits:
    cli_calls: int = SESSION_CLI_CALLS
    gate_submissions: int = SESSION_GATE_SUBMISSIONS
    context_tokens: int = SESSION_CONTEXT_TOKENS


@dataclass
class Budget:
    """Per-run counters, persisted in ``runs/<run id>/budget.json``.

    decodes (float, see policy doc); gate_checks; gate_cap; cost_usd; cost_cap_usd;
    sessions; no_accept_streak."""
    decodes: float = 0.0
    gate_checks: int = 0
    gate_cap: int = 0
    cost_usd: float = 0.0
    cost_cap_usd: float = RUN_COST_CAP_USD
    sessions: int = 0
    no_accept_streak: int = 0

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Budget":
        raise NotImplementedError
