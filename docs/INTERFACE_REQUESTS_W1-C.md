# Interface requests from W1-C (code arms)

For W2a to resolve at the merge. The arms use local adapters until then.

1. **Resolved (checked against `origin/art/edits`).** B1a, B1b and B5 train with toypref's
   `train_arm` (bit-exact with `scripts/cond_cell.py`) and store the result as
   `Patch(call="targeted_train", arrays={<parameter name>: full tensor})` with only the changed
   tensors; W1-B's `edits.apply_call` already applies such a patch as a full-tensor overwrite.
2. **Resolved (`origin/art/harness-gate`).** B5 ranks accepted candidates by `GateResult.numbers["D_new"]`,
   which W1-A2's gate already writes.
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
8. **W1-B / W2a, `edits.unit_tests` for R0 controllers (needed for R0 to ever pass the gate).** R0's
   versions wrap the current net in `toyrepair.arms.r0.ControlledNet` (code at sampling time, no
   weight change). `edits.unit_tests` rejects it today (parameter names gain an `inner.` prefix,
   `run_forward` and the type check do not know it). Please add at the top of `unit_tests`:

   ```python
   hook = getattr(new.net, "controller_unit_tests", None)
   if hook is not None:
       r = hook(current)
       return UnitTestResult(passed=r["passed"], checks=r["checks"], detail=r["detail"])
   ```

   The hook checks the detector source and width, the parent's parameter shapes, finite outputs,
   and bit-identity to the parent with the detector forced off (10,000 mixed-step states).
