# STATE_W1-D

Branch: `art/llm` (from `agentic-repair-toy` be38656).

Done:
- `toyrepair/prompts/surgeon.md`, `triage.md` (static texts)
- `toyrepair/cli.py`: tokens, limits, arm command sets, path checks, per-call ledger records, caps; RealBackend + MockBackend
- `toyrepair/dossier.py`: static/variable parts with caps, renderers, ledger-summary fallback
- `toyrepair/orchestrate.py`: init / next / record / status, stop rules, cool-down, triage fallback, locks
- `toyrepair/transcripts.py`: session_report, find_subagent_for_token, isolation_audit, chars_per_token
- tests: `tests/toyrepair/test_llm_harness.py` (20 pass)

Running: none.

Next: real triage + surgeon subagent on a mock run; measure chars/token; DONE file.
