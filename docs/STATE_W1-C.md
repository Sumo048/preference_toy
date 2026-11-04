# STATE W1-C

Branch `art/arms` (from `origin/agentic-repair-toy` be38656). Also mirrored to `claude/session-w1-c-rwpo6t`.

## Done
- Arms B0, B1a, B1b, B4, B5, R0, D1 in `toyrepair/arms/` (Services injection, shared loop in `base.py`).
- `scripts/run_arm.py`, `scripts/phase1_repro.py`, tests `tests/arms/test_arms.py` (fakes).
- `docs/DEV_LOG.md` (D1 variants, R0 design), `docs/INTERFACE_REQUESTS_W1-C.md`.

## Running
- (see below)

## Next
1. torch install -> `pip install -e .` -> tests.
2. Phase 1 manifest: `python scripts/phase1_repro.py manifest phase1 && python scripts/jobs.py run results/phase1/manifest.jsonl`.
3. Report: `python scripts/phase1_repro.py report phase1`.
4. Baselines seeds 5-9: same with `baselines` (140 units).
