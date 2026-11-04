# Interface requests from W1-D (for W2a's merge)

W1-D's code calls the W0 signatures. Where an owner's function was still a stub, W1-D uses a local adapter that falls back on `NotImplementedError`. Once the owner's code is merged, that code is used.

## W1-A2 (ledger, budget)

1. **Ledger.** `orchestrate.ledger_append` calls `Ledger(path).append(LedgerRecord)`. The fallback writes `asdict(LedgerRecord)` as one JSON line, the same format. W1-D writes these kinds and payloads:
   - `tool`: `{cmd, argv (token redacted), ok, error, out_chars, wall_s, decodes, ...}`
   - `edit`: the `EditRecord` fields plus `patch_id` and `cluster`
   - `gate`: `{cluster, call, surgeon_view, result: asdict(GateResult)}`
   - `quick_check`: `{still_bad, residual, drift}`
   - `session` (`event` start/end), `triage` (issued/record), `branch`, `atlas`, `note` (stop), `invalidate`
2. **`Ledger.summary(max_tokens)`** goes into the surgeon's and the triage's input. It must show test names as `cli.GATE_TEST_LABELS` gives them (`4_other`, never `other_forgotten`). It must hold no oracle number beyond gate pass/fail and test 1. The fallback is `dossier.ledger_summary`. Keep it, or match it.
3. **`Budget.load` / `Budget.save`**: the fallback is a JSON dict with the `Budget` fields in `budget.json`. `Budget.load` on a missing file may raise `FileNotFoundError`; the adapter handles that.

## W1-A1 (policy, flowstate, miner, quick_check)

4. **`flowstate.inspect`** has no stub signature. The CLI calls `flowstate.inspect(cell, version, uid: int, units: str)`, where `units` is `"LAYER:i,j"` or `""`. It expects an object with `.render()` (21 rows or fewer) and `.n_decodes`, or a string.
5. **Renderers.** `FailureCard.render`, `BranchTable.render` and `Atlas.render` must respect their caps. They must not say "mode", "ring", "forgotten" or "retained", and must not give a mode count. A test greps the variable part for these words. Until the renderers exist, `dossier.render_card` and `dossier.render_branch` are the fallbacks.
6. **Versions on disk.** `<run dir>/versions.json` maps `vid -> {parent, patches: [patch ids], session, call}`. The CLI rebuilds versions with `patches.rebuild(cell, run_dir, patches, vid)`, and `v0` with `policy.base_version(cell)`. `policy.load_version` should read the same file.
7. **Branch kwargs.** `flowstate.branch(cell, version, cluster, steps=..., M=...)` is called with a `miner.Cluster` from the pickled atlas (`<run dir>/atlas/<vid>.pkl`). Atlas and BranchTable objects must be picklable.

## W1-B (edits, train, sandbox, patches)

8. **EDIT_REFERENCE.md budget.** It may use at most 1,500 tokens at the measured 1.8 characters per token, that is 2,700 characters. With the reference included, the static prompt measures about 2,480 + 1,500 tokens against the 4,000 cap. `test_static_fits_with_full_edit_reference` checks this. The reference must not mention modes or the ring. The CLI syntax of each call is already in the command list (`cli.USAGE`), so the reference should give semantics, units and one example per call.
9. **Sandbox audit hook and torch.** `import torch` calls `ldconfig` through `subprocess.Popen` (`ctypes.util`). An audit hook that blocks `subprocess.Popen` therefore breaks `import torch`: the mock surgeon hit this. Import numpy and torch before you call `sys.addaudithook`.
10. **Sandbox scratch location.** `Run.scratch` is `<repo>/runs/repair/<run>/scratch`, inside the mode-700 repo, so the `sandbox` user cannot reach it. Either the sandbox runner gives the `sandbox` user access to that one directory, or W2a moves `Run.scratch` (one property in `orchestrate.py`) to a path outside the repo. The CLI's path checks follow `run.scratch`.
11. **Patch ids.** The CLI assigns patch ids (`p0001`, …) and sets `res.patch.id = pid` before `save_patch(run_dir, patch)`.
12. **Anchor spelling.** `tools/surgeon_hook.py` allows no `+` character, so the CLI spells the anchor `local+map` as `--anchor local_map` and maps it back.

## Orchestrator use (W2a)

13. Use distinctive run ids (6 characters or more, for example `pilot_k4_s0_n4_A1`). The audit always flags `repair/<other id>`. It flags a bare other id only when that id has 6 or more characters, because short ids give false hits.
14. For code arms, `orchestrate next` prints a `CODE python scripts/run_arm.py ...` hint. Iterative code arms (Phase 4) can reuse `next_action`'s stop rules and cool-down, if W2b wants that.
