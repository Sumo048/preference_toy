#!/usr/bin/env python3
"""Project-level PreToolUse hook (W0 workaround). Agent-frontmatter `hooks:` did not fire in the
cloud session, so .claude/settings.json runs this for every tool call. It applies the unchanged
tools/surgeon_hook.py only when the call comes from the surgeon, surgeon-sonnet, or triage subagent,
and passes every other call (main agent, other subagents) through. Hook input is logged
to ~/.claude/role_hook_log.jsonl (keys only) so the dispatch can be audited."""
import json, os, subprocess, sys, time

ROLES = {"surgeon", "surgeon-sonnet", "triage"}
# A3LF session: its Claude Code session started on a branch without .claude/agents, so the role
# subagents are not registered. It runs surgeon and triage as the built-in `claude` agent type
# (model opus / sonnet, the role definition's text in the task message) and uses that type for
# nothing else, so the same filter applies (docs/A3LF_PREREG.md, deviations).
ROLES |= {"claude"}
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception:
    data = {}
root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
agent = data.get("agent_type") or data.get("agentType") or ""
try:
    with open(os.path.expanduser("~/.claude/role_hook_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), "keys": sorted(data), "agent_type": agent,
                            "tool": data.get("tool_name")}) + "\n")
except Exception:
    pass
if agent not in ROLES:
    sys.exit(0)
if data.get("tool_name") == "SubagentHandback":   # returning the final report
    sys.exit(0)
if data.get("tool_name") != "Bash":
    print("Blocked: only Bash is allowed.", file=sys.stderr)
    sys.exit(2)
r = subprocess.run([sys.executable, "-I", os.path.join(root, "tools", "surgeon_hook.py")], input=raw,
                   text=True, capture_output=True)
sys.stderr.write(r.stderr)
sys.exit(r.returncode)
