# STATE_W2b

Branch: local `agentic-repair-toy` (from the W2a DONE commit). Pushes: `claude/session-w2b-uef5f7` works;
`agentic-repair-toy` was accepted once (4ec5628) and then refused by the remote ("remote rejected ... failure").

## Done
- Waited for DONE_W2a; read E1/E2/E3 write-ups and the smoke-ledger sandbox code.
- Phase 4 tools: toyrepair/tracks.py, toyrepair/act_tools.py (contrast, unit_card), CLI wiring, tests (4 new;
  46 relevant tests pass). Static A3+P4 prompt 4000 tokens.
- Phase 4 runs on k4 npm 4/16 seeds 0-4: code arms B4/R0/D1 iterative (30 units, results/phase4), A3 recursive
  (10 runs, all ended at a stop rule, evaluated).
- DEV_LOG: tool design, oracle diagnostic of track labels, triage hand-back bug (pass --summary), surgeon
  background stalls (send a resume message), observed failure modes.
- PREREG.md filled (scripts/prereg_shares.py, scripts/prereg_mdd.py).

## Next
- Done. DONE_W2b.md written and pushed.
