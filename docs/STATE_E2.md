# STATE E2 (direction edits)

Branch: `claude/session-e2-ivzni2` (harness-assigned), mirrored to `art/explore-E2` if the push is allowed.
Code: `explore/E2/` (lib.py, fit.py, run_edit.py, probe*.py, clusters.py, queue.sh). Results: `explore/E2/results/<cell>/<method>.json`.
Baselines: `explore/E2/run_baselines.sh` -> `runs/cond/cells*/` (not committed; numbers summarised in docs/explore/E2.md).

## Done
- Env: torch 2.14.0 from PyPI (cu130 build, CPU use); pandas 2.2.2 has no py3.13 wheel, newer pandas is used.
- B0 + B1a on k4/alt x npm 4/16 x seeds 0-4 (running/near done).
- Global probe k4 s0 n16: AUC 0.64-0.72 -> one global direction separates bad trajectories poorly.
- r1_out_r2 (s1000) k4 s0 n16: D 0.237, Rem 0.73, Col 0.04, off 0.10 (B1a 0.139).

## Running
- q1 sweep of edit structures on k4 s0 n16; per-cluster probe.

## Next
- Pick best structures, run on 12 cells (k4/alt, npm 4/16, seeds 0-2), best on seeds 3-4; write docs/explore/E2.md, DONE_E2.
