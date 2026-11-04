#!/usr/bin/env python3
"""PreToolUse hook for the surgeon and triage subagents (TOY_PLAN.md Section 7.5).

Allows exactly one `toyrepair-tool` call per Bash command. Everything else is
blocked with exit code 2, and the reason goes to stderr so the subagent sees it.

Allowed forms:
    toyrepair-tool <cmd> --token <tok> [args...]
    toyrepair-tool sandbox --token <tok> --stdin <<'PY'
    <python code>
    PY
Arguments may use only letters, digits, spaces, and the characters _ - . = : / , [ ]
"""
import json
import re
import sys

ARG_LINE = re.compile(r"^toyrepair-tool(?: [A-Za-z0-9_\-\.=:/,\[\]]+)*$")
HEREDOC_LINE = re.compile(r"^(toyrepair-tool sandbox(?: [A-Za-z0-9_\-\.=:/,\[\]]+)*) <<'PY'$")


def block(reason: str) -> None:
    print(f"Blocked: {reason} Use a single toyrepair-tool command per Bash call.", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        block("hook input did not parse.")
    if data.get("tool_name") not in (None, "Bash"):
        block("only Bash is allowed.")
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not isinstance(cmd, str) or not cmd.strip():
        block("empty command.")
    lines = cmd.strip("\n").split("\n")
    first = lines[0].strip()

    if len(lines) == 1:
        if ARG_LINE.match(first):
            sys.exit(0)
        block("the command is not a plain toyrepair-tool call.")

    m = HEREDOC_LINE.match(first)
    if not m:
        block("multi-line commands are allowed only for `toyrepair-tool sandbox ... --stdin <<'PY'`.")
    if "--stdin" not in m.group(1).split():
        block("the sandbox heredoc form needs --stdin.")
    if lines[-1].strip() != "PY":
        block("the heredoc must end with a line that is exactly PY.")
    if any(line.strip() == "PY" for line in lines[1:-1]):
        block("the heredoc terminator PY may appear only once, on the last line.")
    sys.exit(0)


if __name__ == "__main__":
    main()
