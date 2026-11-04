"""The code half of the loop (docs/PLAN.md 6.10, 7.4).

``python -m toyrepair.orchestrate next --run <id>`` prints the next action:
  TRIAGE <token>          invoke the ``triage`` subagent with this token
  SESSION <agent> <token> invoke ``surgeon`` or ``surgeon-sonnet`` with this token
  CODE <what>             code work that ran (code arms)
  DONE <stop rule>        the run stopped
``python -m toyrepair.orchestrate record --token <token> [--summary "<line>"]`` reads the
subagent transcript, sums tokens and cost, audits it and writes the ledger record.
``python -m toyrepair.orchestrate status`` prints one line per run.
``python -m toyrepair.orchestrate init --run <id> --arm <arm> --grid cond --setting k4 --seed 0 --npm 4``
creates the run directory.

Stop rules: no open cluster; 3 sessions in a row without an accept; gate checks reach
3 x (first-atlas clusters + 2); cost cap spent.

Owner: W1-D.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ARMS = ("B0", "B1a", "B1b", "B4", "B5", "R0", "D1", "A1", "A2", "A3",
        "A3-nobranch", "A3-sonnet", "A3-nogate")
LLM_ARMS = ("A1", "A2", "A3", "A3-nobranch", "A3-sonnet", "A3-nogate")
STOP_RULES = ("no_open_cluster", "no_accept_streak", "gate_cap", "cost_cap")


@dataclass
class SessionToken:
    """token: random url-safe string (letters, digits, - _); run: run id; session: session id
    (e.g. ``s003``); role: "surgeon" | "triage"; agent: subagent name; cluster: target cluster
    id (None for triage); calls / submissions: counters; closed: True after record."""
    token: str
    run: str
    session: str
    role: str
    agent: str
    cluster: Optional[int] = None
    calls: int = 0
    submissions: int = 0
    closed: bool = False


@dataclass
class Run:
    """run id; dir ``$TOYPREF_RUNS/repair/<id>`` (ledger.jsonl, budget.json, patches/, scratch/); arm; cell key (grid, setting, seed, npm); current version id;
    first-atlas cluster count; cool-down {cluster: iterations left}; scratch dir."""
    id: str
    dir: str
    arm: str
    grid: str
    setting: str
    seed: int
    npm: int
    current: str = "v0"
    n_first_clusters: Optional[int] = None
    cooldown: Dict[int, int] = field(default_factory=dict)

    @property
    def scratch(self) -> str:
        return self.dir + "/scratch"


def init_run(run_id: str, arm: str, grid: str, setting: str, seed: int, npm: int) -> Run:
    raise NotImplementedError


def load_run(run_id: str) -> Run:
    raise NotImplementedError


def next_action(run: Run) -> str:
    raise NotImplementedError


def status(runs_root: Optional[str] = None) -> List[str]:
    """One line per run under ``$TOYPREF_RUNS/repair``: id, arm, cell, current version,
    sessions, gate checks, cost, stop rule or "running"."""
    raise NotImplementedError


def record(token: str, summary: str = "") -> Dict:
    raise NotImplementedError


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m toyrepair.orchestrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    for a in ("--run", "--arm", "--grid", "--setting"):
        p.add_argument(a, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--npm", type=int, required=True)
    sub.add_parser("next").add_argument("--run", required=True)
    sub.add_parser("status")
    p = sub.add_parser("record")
    p.add_argument("--token", required=True)
    p.add_argument("--summary", default="")
    args = ap.parse_args(argv)
    raise NotImplementedError(args.cmd)


if __name__ == "__main__":
    sys.exit(main())
