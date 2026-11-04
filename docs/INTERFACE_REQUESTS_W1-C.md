# Interface requests from W1-C (code arms)

For W2a to resolve at the merge. The arms use local adapters until then.

1. **W1-B, `patches.apply_patch` / `rebuild`: call `toypref_train`.** B1a, B1b and B5 train with
   toypref's `train_arm` (bit-exact with `scripts/cond_cell.py`) and store the result as a
   `Patch(call="toypref_train", arrays={<parameter name of the CondVelocityNet>: full tensor})`,
   holding only the tensors that changed. Please make `apply_patch` load such a patch as a full
   tensor overwrite (the same as a `targeted_train` patch that "stores the full changed tensors").
   Adapter: if `patches.save_patch` raises `NotImplementedError`, `RunIO.save_patch` writes
   `<id>.json` and `<id>.npz` itself.
2. **W1-A2, `GateResult.numbers["D_new"]`.** B5 ranks accepted candidates by the gate's D of the
   new version. Please store it under `numbers["D_new"]` (and `D_current`, `D_ref`). Fallback in
   B5: rank by the test-1 fall.
3. **W1-A2, `ledger.Ledger`.** The arms write `LedgerRecord(kind, run, ts, data, session, version)`
   through `Ledger.append`. Adapter: on `NotImplementedError` they append the same fields as JSON
   lines to `ledger.jsonl`.
4. **W1-A2, `eval_final.evaluate`.** `scripts/run_arm.py` calls `evaluate(cell, version, arm)`.
   Adapter: toypref `Bank` + `compute` on the version's net (same eval bank, so the same numbers).
5. **W1-A1, `policy.decode` with a stateful wrapper net.** R0's `ControlledNet` decides at step k
   and holds the decision for steps k+1..k+h-1. This needs `decode` to call the net once per step,
   chunk by chunk, in step order inside a chunk (the toypref sampler order, which the docstring
   already promises). The gate and quick_check must decode through `version.net`.
6. **W1-A1, `flowstate.restore_and_decode` from a step-0 state with a primitive at a later k.**
   D1 builds rescue targets with `restore_and_decode(version, FlowState(x=xi, k=0, c=c), best_primitive)`
   and expects the primitive to act at its own k (push) or at step 0 (hop).
7. **W1-D, `orchestrate.Run`.** `scripts/run_arm.py` builds `Run(id, dir, arm, grid, setting, seed, npm)`
   directly under `$TOYPREF_RUNS/repair/<id>` instead of `init_run`. For W2a: let `next_action`
   call `toyrepair.arms.get(arm)(cell, run, max_rounds=...)` for code arms, or run them through
   `scripts/run_arm.py` and `scripts/jobs.py`.
