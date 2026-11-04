# STATE W1-C

Branch `art/arms` (from `origin/agentic-repair-toy` be38656), mirrored to `claude/session-w1-c-rwpo6t`.

## Done
- Arms B0, B1a, B1b, B4, B5, R0, D1 in `toyrepair/arms/` (Services injection; shared loop in `base.py`).
- `scripts/run_arm.py`, `scripts/phase1_repro.py`; tests `tests/arms/` (18 pass: fakes + real-cell bit-identity vs cond_cell).
- `docs/DEV_LOG.md` (D1 variants, R0 design), `docs/INTERFACE_REQUESTS_W1-C.md`.
- Env: torch 2.14.0 from PyPI (CUDA wheel, ~10 min to install), `cloud_setup.sh` ok.

## Running
- Phase 1: `python scripts/jobs.py ensure results/phase1/manifest.jsonl` (80 units, one per cell x arm group;
  started 02:57 UTC). The first 4 k4 n64 cells ran as whole-cell units (19.5 min each, too close to the cap -> split).

## Next
1. `python scripts/phase1_repro.py report phase1` -> results/phase1/repro.md.
2. Smoke `scripts/run_arm.py --arm B1a --setting k4 --seed 0 --npm 4` (compare D with cond_cell).
3. Baselines: `python scripts/jobs.py run results/baselines/manifest.jsonl` (240 units), then `report baselines`.
4. DONE file.
