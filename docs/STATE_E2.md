# STATE E2 (direction edits)

Branch: `claude/session-e2-ivzni2` (harness-assigned), mirrored to `art/explore-E2` if the push is allowed.
Code: `explore/E2/` (lib.py, fit.py, run_edit.py, probe*.py, clusters.py, queue.sh). Results: `explore/E2/results/<cell>/<method>.json`.
Baselines: `explore/E2/run_baselines.sh` -> `runs/cond/cells*/` (not committed; numbers summarised in docs/explore/E2.md).

## Done
- Env: torch 2.14.0 from PyPI; pandas 2.2.2 has no py3.13 wheel, newer pandas used.
- B0 + B1a on k4/alt x npm 4/16 x seeds 0-4 (runs/cond, via cond_cell.py). k4 B1a 0.281, alt 0.446 (s0-4).
- Probes: global direction AUC 0.64-0.72; per-cluster (miner affinity) AUC 0.95-0.99 already at step 0 (b0.z).
- Sweep k4 s0 n16 (q1-q4): ablation fails (abl no effect; ablc destroys -> off mass); learned rank-k works;
  early t-gate cuts between-mode mass; best: r1_fc2.0+fc2.1+fc2.2_r4_t0.4-1 D 0.142 (B1a 0.139).
- 12 cells (q5): best edit matches B1a (k4 0.258 vs 0.261, alt 0.426 vs 0.421); r1pc (probe-fixed) worse.
- B1a SVD k4 s0 n16: truncating B1a's change to rank 4/layer gives D 0.124 < 0.139 (less collateral).

- Seeds 3-4, AL full-weight control, full_w1-0.5 control, t-gated B1a (19/20 cells better): all done.
- docs/explore/E2.md and docs/DONE_E2.md written.

## Running
- nothing.

## Next
- nothing; session complete.
