# STATE W3-1 (share 1 of PREREG.md)

Branch: local `art/grid-1` from origin/agentic-repair-toy 4fca1dd (DONE_W2b). Push to `art/grid-1` and the
session branch `claude/session-w3-1-22i3c7`. Hard deadline Wed 23:00 ET.

## Done
- Waited for DONE_W2b (PASS-WITH-ISSUES, nothing blocking). cloud_setup ok.
- Driver `scripts/grid_share1.py` (caches, manifest, code, tick, record, eval, table). Run ids
  `grid_cond_<setting>_s<seed>_n<npm>_<arm>[_rep1]`; results `results/grid/<setting>/s<seed>/n<npm>/<arm>/`.
- Baselines B0/B1a/B1b for seeds 5-9 are on origin/art/arms (2d29e95), not in agentic-repair-toy; extracted to
  scratchpad, `BASELINES=<scratch>/base/results/baselines` for `table`.

## Running
- jobs: results/grid/_share1/caches.jsonl (cell + track caches, 20 cells)

## Next
1. Code arms: `python scripts/jobs.py run results/grid/_share1/code.jsonl --workers 3 --timeout 5400`.
2. LLM loop: `tick` -> invoke surgeon/triage (background) -> `record --token T --summary "<line>"` -> tick.
   Resume a surgeon idle > 10 min ("your background command finished; continue in the foreground").
3. When share is done: `tick --rep` (A3_rep1) until 23:00 ET. Then table, DONE_W3-1.md, push.
