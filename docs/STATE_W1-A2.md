# STATE W1-A2 (gate and evaluation), branch art/harness-gate

Done:
- oracle/ (labels, target_modes, purity, metrics incl. PLAN collateral + retain_tv, toypref_compute cond+uncond)
- oracle/_adapt.py: local adapters (decode -> policy.decode or local Euler; base_version; edits.unit_tests stub)
- gate.py: gate_check (both scopes, six tests, joint-category paired bootstrap), noop_check, power_check, recipe_check
- ledger.py, budget.py, eval_final.py; tests/toyrepair/test_gate_oracle.py, test_ledger_budget.py (20 pass)

Running: results/gate_check/manifest.jsonl (noop, power at N_G=16384 on 4 cells; recipe on k4/alt s0 n4)

Next: read results, double N_G if power < 80%, write DONE_W1-A2.md, push art/harness-gate (+ claude/session-w1-a2-6n8qd5).
