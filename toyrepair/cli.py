"""``toyrepair-tool``: the only interface of the surgeon and triage roles (docs/PLAN.md 7.2-7.5).

Wave 0: ``--help`` only; every subcommand prints "not implemented" and exits 2.

Import rule: this module imports neither ``toypref`` nor ``toyrepair.oracle`` at import time.
Commands load what they need lazily, and never the oracle (gate results come back through
``toyrepair.gate.GateResult.surgeon_view``).

Every command takes ``--token``; the CLI maps it to (run, session), enforces the session limits
(20 calls, 3 submissions) and checks that every path argument is inside the run's scratch dir.
Arms remove commands (Section 9.2): a removed command returns an error.

Owner: W1-D.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

#: command -> one-line help
COMMANDS = {
    "prompt": "print the static prompt and your dossier (first call of a session)",
    "dossier": "print the variable part: failure card, branch table, ledger summary, budget",
    "atlas": "print the failure cards of the current version",
    "triage-input": "print the atlas and the ledger summary (triage only)",
    "branch": "run the branch tool on a cluster (--cluster, --steps, --M, --primitives)",
    "inspect": "one decode of one negative, <= 21 rows (--neg, --units)",
    "sandbox": "run Python on the export (--stdin heredoc or --file)",
    "quick_check": "oracle-free check of a candidate version (--version)",
    "graft_gated_path": "edit: gated hidden units beside a layer",
    "rank1": "edit: W <- W + s u v^T",
    "ablate_direction": "edit: project a direction out of a layer (side pre|post)",
    "set_units": "edit: write selected units' weights",
    "targeted_train": "edit: targeted training with the locality penalty",
    "submit_edit": "send a candidate version to the gate (max 3 per session)",
    "contrast": "activation contrast at step k for a cluster (Phase 4)",
    "unit_card": "correlates of one unit (Phase 4)",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="toyrepair-tool",
        description="Tool CLI of the toyrepair harness. One command per call; pass --token on every call.")
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    for name, helptext in COMMANDS.items():
        p = sub.add_parser(name, help=helptext, description=helptext)
        p.add_argument("--token", help="session token from your task message")
        p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help()
        return 0
    print(f"toyrepair-tool {args.command}: not implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
