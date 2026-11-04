# STATE_W2a (session branch claude/session-w2a-nbkrjw)

Note: a second W2a session exists (claude/session-w2a-5zzqkv). The user said to continue anyway.

## Done
- Waited for inputs; all five DONE_W1-* found (A1 art/harness-sim, A2 art/harness-gate, B art/edits,
  C art/arms, D art/llm). Merged them one at a time into local `agentic-repair-toy` (only spend.jsonl conflicts; union).
- Env: pandas 2.2.3 + numpy 2.2.6 by hand (pandas 2.2.2 has no cp313 wheel); cloud_setup.sh then passes.
- Interface requests resolved: A1-1 purity adapter removed; A1-2/B-1 policy capture on EditedNet via
  edits.run_forward, layers() lists grafts; C-1 apply_patch accepts toypref_train; C-8 unit_tests delegates
  to controller_unit_tests; D-2 ledger summary uses 4_other labels; D-4 CLI parses --units for inspect;
  D-8 EDIT_REFERENCE.md rewritten to 2,695 chars (static A3+P4 = 3,988 tokens); CHARS_PER_TOKEN 1.8 everywhere.
  Others already satisfied by the merged code.
- Fixes: CLI patch-id rename broke graft rebuild (sha mismatch) -> patch id passed into edit calls.
  Cell cache (runs/cells/<key>/cell_cache.pt) so each CLI call does not reload the cell (~40 s).
- Phase 2 exit: hand graft on uncond base_k2 through orchestrator/CLI/gate/ledger: ACCEPT (results/phase2_exit/).
- Miner rule -> agglomerative (DEV_LOG). Pilot mode: run_arm --cluster/--targets; orchestrate --pilot_cluster/--max_sessions.
- scripts/pilot.py (targets, manifest, eval_llm, report).

- Integration smoke (k4 s0 n4): code arms ran; A1 and A2 one session each ($0.84, $0.93), no accept.
- Gate calibration: B4 0/12 at dev thresholds; loosened once (0.15/0.02/0.02) -> 1/12 (results/pilot/calibration.md).
- Pilot r1 done: Table 2 in results/pilot/r1/table2.md. Rule FAIL (A1 0/12, A2 2/12; A2 4 accepts).
- Improvement round r2 prompt changes (DEV_LOG); r2 running (code arms reused from r1, deterministic).

## Running
- r2 surgeon sessions (pilot_runs_r2.txt in scratchpad; 16 issued of 24).

## Next
1. Finish r2 (sessions 17-24), eval (scripts/pilot_eval_manifest.py r2 + jobs), report --round r2 (copy code arms).
2. results/pilot.md: Table 2 both rounds, cost per session/run. Full test suite, push agentic-repair-toy. DONE_W2a.md.
- Hard deadline: Wed 12:00 ET (16:00 UTC).
