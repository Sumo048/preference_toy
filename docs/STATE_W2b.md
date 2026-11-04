# STATE_W2b

Branch: local `agentic-repair-toy` (from origin/agentic-repair-toy, W2a DONE commit). Push to `agentic-repair-toy`
and mirror to the session branch `claude/session-w2b-uef5f7`.
Hard deadline Wed 17:00 ET. Time box 5 h from 02:50 ET (DONE_W2a seen).

## Done
- Waited for DONE_W2a (PASS-WITH-ISSUES: pilot rule failed both rounds; Amendment 1 -> continue).
- Read docs/explore/E1.md, E2.md, E3.md (E1 art/explore-E1, E2 art/explore-E2, E3 art/explore-E3) and the
  pilot sandbox code in results/pilot/smoke_llm/*_ledger.jsonl (the r1/r2 run dirs were not committed).
- Phase 4 tools: toyrepair/tracks.py (label-free behaviour tracks, from E3/E1), toyrepair/act_tools.py
  (contrast, unit_card), wired into cli.RealBackend; static prompt A3+P4 = 4000 tokens (cap 4000).
- Code arms B4/R0/D1 are already iterative (arms/base.iterate with max_rounds=None).

## Running
- cloud_setup (torch install).

## Next
1. Tests for tracks/act_tools; smoke contrast/unit_card through the CLI on k4 s0 n4.
2. Phase 4 runs: A3 (phase4) + B4/R0/D1 iterative on k4 npm 4,16 seeds 0-4; final eval.
3. PREREG.md (Section 10.5, Amendment 1 grid, 3 shares); DONE_W2b.md; push.
