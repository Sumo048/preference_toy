# STATE W3-3 (share 3, PREREG.md Section 6)

Branch: `art/grid-3` from origin/agentic-repair-toy 4fca1dd (DONE_W2b). Push to art/grid-3 and claude/session-w3-3-cghp65.
Deadline: Wed 23:00 ET. TOYPREF_RUNS=$PWD/runs.

## Done
- Waited for DONE_W2b (ready 07:41 ET). cloud_setup.sh OK (31 s; pandas 2.2.3 for py3.13, numpy 2.2.6 pinned).
- Driver `scripts/grid3.py` (manifest / warm / init / queue / eval / report). No method changes.
- Code arms: `results/grid/manifest_share3.jsonl` (160 units: b3p1, k4/alt n1/n64, k2; B4,R0,D1,B5),
  `jobs.py run ... --workers 2 --timeout 5400`. Results -> results/grid/<s>/s<seed>/n<npm>/<arm>/result.json.
- LLM runs: 55 initialised (`grid3.py init`, run ids g3_cond_<s>_s<seed>_n<npm>_<arm>), --phase4.

## Loop (per LLM run, up to 8 at a time, in `grid3.py queue` order)
`orchestrate next --run R` -> TRIAGE: Agent(triage, "token T; run toyrepair-tool triage-input --token T") ->
`record --token T --summary '<json>'`; SESSION: Agent(surgeon|surgeon-sonnet, "run R, token T, start with
toyrepair-tool prompt --token T") -> `record --token T --summary '<RESULT line>'`; DONE -> `grid3.py eval --run R`.
Idle surgeon > 10 min with calls left: send "your background command finished; continue in the foreground".

## Driving shortcut
`python scripts/grid3.py step --run R --summary '<triage JSON | surgeon RESULT line>'` records the run's pending
token, then prints the next action; on DONE it starts `grid3.py eval` in the background (log results/grid/eval_logs.txt).
Fresh run: `grid3.py step --run R` (no summary). Pending tokens: `orchestrate status` + runs/repair/R/run.json.
Notes on resumes / flags: results/grid/notes_share3.md.

## Main loop (use this)
`python scripts/grid3.py auto [--rep]`: records every pending run whose subagent handed back (hand-back text
read from the transcript = the --summary), advances it, starts fresh runs up to 8 active, prints
`LAUNCH <agent> <run> <token>` (invoke Agent subagent_type=<agent> with "Run R. Your (session) token is T.
Start with `toyrepair-tool prompt --token T`" / triage: "Run `toyrepair-tool triage-input --token T`, then reply
with the one-line JSON."), `WAIT <run> agent=<id> idle=Nm` (SendMessage resume if idle >10 m and its tool
process is gone), `DONE` (eval auto-started), `NOEVAL` (eval missing: run `grid3.py eval --run R`).

## Next
- Drive LLM runs; `jobs.py ensure` while polling; commit + spend at milestones; replicates (`init --rep`) after share.

## Patch stacks (user request, 16:50 UTC)
Each finished LLM run's final patch stack (p*.json + p*.npz of versions[current].patches, plus stack.json with the
accepted chain; no caches/scratch) goes to `results/grid/<setting>/s<seed>/n<npm>/<arm>/patches/`.
`grid3.py eval` exports it automatically after the evaluation; `python scripts/grid3.py patches` re-exports every
finished run (run it before each commit).
