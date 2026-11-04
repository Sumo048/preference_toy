# STATE A3LF

Branch `art/a3-lf` (mirror pushes to `claude/a3-lf-label-free-gate-ti6cdq`). Deadline Thu 02:30 ET (07:30 UTC).
Prereg: `docs/A3LF_PREREG.md`, pushed at cb7575c before any seed 5-9 cell was loaded.

## Loop (read after compaction)
- `python scripts/a3lf.py tick --slots 8` prints `NEW <triage|surgeon> <token> <run>`.
- Launch Agent subagent_type `claude`, model `sonnet` (triage) or `opus` (surgeon). The message is the role
  definition text (.claude/agents/triage.md or surgeon.md body), then for a surgeon "Run <run>. Session token: <tok>.
  Start with `toyrepair-tool prompt --token <tok>`." and for triage "Run `toyrepair-tool triage-input --token <tok>`
  and answer with the one-line JSON."
- When an agent reports: `python scripts/a3lf.py record --token <tok> --summary '<JSON or RESULT line>'`, then tick.
- DONE runs get `eval` in the background: final eval + shadow oracle gate + purity, written to results/a3lf/...
- Agent id <-> token map: scratchpad agents.tsv.
- Caches: `python scripts/jobs.py ensure results/a3lf/_driver/caches.jsonl`.
- After the 20 main cells: `tick --extra` (k1, contig, b3p1 at npm 4).

## Status
- 19:20 UTC: done k4 s8 n16 (D 0.517), k4 s8 n4 (D 0.082), k4 s7 n4 (eval running). Running: k4 s5,s6,s7n16,s9 + alt s5 n4.
