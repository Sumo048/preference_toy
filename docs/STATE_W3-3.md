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

## Next
- Drive LLM runs; `jobs.py ensure` while polling; commit + spend at milestones; replicates (`init --rep`) after share.
