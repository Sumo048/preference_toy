# STATE W3-1 (share 1 of PREREG.md)

Branch: local `art/grid-1` from origin/agentic-repair-toy 4fca1dd (DONE_W2b). Push HEAD to BOTH `art/grid-1` and
`claude/session-w3-1-22i3c7`. Hard deadline Wed 23:00 ET (04:00 UTC Thu).

## How the loop works (read after compaction)
- Driver `scripts/grid_share1.py`. State: `results/grid/_share1/llm_state.json` (inflight token per run, done list).
- Cycle: `python scripts/grid_share1.py tick --slots 8` prints `NEW <triage|surgeon> <token> <run>`; launch an Agent
  (subagent_type triage or surgeon, run_in_background default) with message
  "Run <run>. Session token: <tok>. Start with `toyrepair-tool prompt --token <tok>`." (triage: "Run
  `toyrepair-tool triage-input --token <tok>` and answer with the one-line JSON.").
- When an agent reports: `python scripts/grid_share1.py record --token <tok> --summary '<JSON or RESULT line>'`, then tick.
- `DONE` runs get final eval automatically (background, log results/grid/_share1/logs/eval_<run>.log).
- Agent id <-> token map: scratchpad agents.tsv. Surgeon "waiting on background work" > 10 min: SendMessage
  "Your background command finished; continue in the foreground." Log in scratchpad resume_notes.txt
  (copied to results/grid/_share1/resume_notes.txt at commits).
- Code arms: `python scripts/jobs.py ensure results/grid/_share1/code.jsonl --workers 3` (timeout 5400 set at run).
- Baselines B0/B1a/B1b seeds 5-9 only on origin/art/arms (2d29e95); for `table` set
  BASELINES=<scratchpad>/base/results/baselines.
- Commit hourly with `python -m toyrepair.transcripts spend --session W3-1`.

## Status (15:45 UTC)
- LLM runs done: k4 s6 n4 A3 (D 0.116), k4 s5 n4 A3/A2/A1, k4 s5 n16 A1. In flight: k4 s5 n16 A3/A2, k4 s6 n4 A2/A1,
  k4 s6 n16 A3/A2/A1, k4 s7 n4 A3. Code arms ~25/80 done. All audits clean.

## Next
- Continue through share order (k4 s5..s9 n4,n16 then alt), each cell A3, A2, A1. After share, `tick --rep`.
- At ~03:30 UTC: stop issuing new sessions, table, DONE_W3-1.md, push.
