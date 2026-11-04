# STATE W1-C

Branch `art/arms` (from `origin/agentic-repair-toy` be38656), mirrored to `claude/session-w1-c-rwpo6t`.

## Done
- Arms B0, B1a, B1b, B4, B5, R0, D1 in `toyrepair/arms/` (Services injection; shared loop in `base.py`).
- `scripts/run_arm.py`, `scripts/phase1_repro.py`; tests `tests/arms/` (18 pass: fakes + real-cell bit-identity vs cond_cell).
- `docs/DEV_LOG.md` (D1 variants, R0 design), `docs/INTERFACE_REQUESTS_W1-C.md`.
- Env: torch 2.14.0 from PyPI (CUDA wheel, ~10 min to install), `cloud_setup.sh` ok.

## Running
- Baselines seeds 5-9: `python scripts/jobs.py ensure results/baselines/manifest.jsonl` (240 units, started 03:52 UTC).
- Phase 1 done (exact reproduction), integration check done (results/arms_integ), DONE_W1-C.md written.

## Next
1. Keep polling baselines (sleep 90 + ensure); commit results/baselines every ~hour.
2. When done: `python scripts/phase1_repro.py report baselines`, append the summary to DONE_W1-C.md, push.
