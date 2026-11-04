"""Sandbox: export, audit hook, runner (docs/PLAN.md 6.6).

The sandbox runs ``python -I`` in a subprocess as the unprivileged user ``sandbox``, cwd = the
run's scratch directory. It sees only the export:

  export/weights.pt     current version's state (+ graft arrays)
  export/model.py       self-contained velocity-network class (no toypref import), with
                        ``decode(xi, c)`` that counts decodes into export/decodes.json
  export/negatives.npz  logged negatives by uid (c, xi, a), public order
  export/basins.npz     each negative's basin points, labels, eps, bad id

The audit hook (``sys.addaudithook``) blocks imports of ``toypref`` and ``toyrepair.oracle``,
socket/network events and subprocess/os.system/exec events. Output is capped at 2000 tokens.
The sandbox has no write access to weights; it writes ``.npy`` arrays into scratch.

Owner: W1-B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SANDBOX_USER = "sandbox"
OUTPUT_TOKEN_CAP = 2000
TIMEOUT_S = 300
BLOCKED_MODULES = ("toypref", "toyrepair.oracle", "toyrepair")
BLOCKED_EVENTS = ("socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.exec",
                  "os.posix_spawn", "os.fork")


@dataclass
class SandboxResult:
    """stdout/stderr (truncated to the cap); returncode; decodes counted; timed_out; truncated."""
    stdout: str
    stderr: str
    returncode: int
    decodes: int = 0
    timed_out: bool = False
    truncated: bool = False


def export(cell, version, scratch: str) -> str:
    """Write the export for ``version`` into ``<scratch>/export``. Returns that path."""
    raise NotImplementedError


def audit_hook_source() -> str:
    """Python source of the audit hook prepended to every sandbox script."""
    raise NotImplementedError


def run(code: str, scratch: str, timeout_s: int = TIMEOUT_S, user: Optional[str] = SANDBOX_USER) -> SandboxResult:
    """Run ``code`` in the sandbox (``user=None`` only in tests where the user does not exist)."""
    raise NotImplementedError
