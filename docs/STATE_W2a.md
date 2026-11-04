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

## Running
- Full test suite (scratchpad/pytest_full.log). Pilot targets: results/pilot/manifest_targets.jsonl (jobs.py).

## Next
1. Tests green -> push agentic-repair-toy.
2. Integration smoke k4 s0 n4: B4/R0/D1 + one A1 and one A2 session; cost.
3. Gate calibration: B4 first repair on 12 cells; loosen 3-5 once if < 6 pass.
4. Pilot r1: code arms via jobs.py; A1/A2 surgeons (<= 8 at a time). report -> results/pilot.md.
5. If decision rule fails: one improvement round (prompts/tools), r2. DONE_W2a.md.
- Hard deadline: Wed 12:00 ET (16:00 UTC).
