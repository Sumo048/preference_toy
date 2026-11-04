# Interface requests from W1-B (edits, patches, train, sandbox)

No W0 signature changed. These are notes and requests for the merge (W2a).

## To W1-A1 (`policy.py`)

1. **Grafted versions.** After a graft, `Version.net` is `toyrepair.edits.EditedNet`: `.base` is the toypref module and `.grafts` the gated paths. `net(x, t, c)` works as usual. For activation capture, call `toyrepair.edits.run_forward(net, x, t, c, capture=names)`. It works on plain and grafted nets, conditional and unconditional. On a plain net it is bit-identical to `net(x, t, c)` (tested). Its names are `CAPTURE_LAYERS` plus `x`, `t`, `c`, `inputs`, and `blocks.{i}.fc1.in`. Request: have `policy.velocity` and `policy.decode(capture=...)` use it, or hooks that give the same values. Module hooks on `net.blocks...` will not find the layers inside an `EditedNet`; use `edits.core_of(net)`.
2. `edits.base_version(cell)` calls `policy.base_version` and falls back to a local deep copy while that raises `NotImplementedError`. Keep the same semantics: v0 holds a deep copy of `cell.ref.net`, frozen.
3. `policy.load_version(run_dir, vid, cell)` can call `patches.rebuild(cell, run_dir, list_of_patch_ids, vid)`.
4. `policy.layers()`: for grafts, use `edits.grafts_of(net)`, where each graft has `.spec()`: layer, src, n_units, detector.
5. `edits.decode_actions(net, xi, c)` is the local decode adapter. It is bit-identical to `CondFlow.sample`. Replace it with `policy.decode` at the merge if you like.

## To W1-A2 (`gate.py`)

- Test 6: call `edits.unit_tests(new_version, current_version)`. It returns `UnitTestResult(passed, checks{shapes, finite, legal_detectors, graft_identity}, detail)` and takes about 0.5 s.

## To W1-D (`cli.py`)

1. **Edit calls.** Each call returns an `EditResult`. When `record.ok`:
   - `patches.save_patch(run_dir, res.patch)`
   - log `res.record` (an `EditRecord`) to the ledger
   - keep `res.version` (or rebuild it later)

   Rejected calls give `version None` and `record.error`; show that message to the surgeon. Patch ids default to `p_<new_id>`; pass `patch_id=` to change them.
2. **targeted_train data**:
   - `train.toypref_rows(cell, uids, rule, support, mine=False)`
   - `train.rescue_rows(cell, uids, targets (m,2), support)`, where targets come from a sandbox `.npy`
   - `train.negatives(cell, uids)`, for NPO and KTO

   All three take uids, not row positions. The CLI should map cluster ids to uids.
3. **Sandbox**: `sandbox.export(cell, version, scratch)` once per current version, then `sandbox.run(code, scratch)`. The runner copies scratch into a private work dir under `/var/tmp/tr_sbx/`, because the sandbox user cannot enter the repo, which holds `runs/`. It runs the code there as user `sandbox` and copies new `.npy/.npz/.json/.txt/.csv/.py` files back. `SandboxResult.decodes` is the decode count for the budget.
4. **Output cap**: the 2000-token cap uses `sandbox.CHARS_PER_TOKEN = 4.0`. Set it to your measured ratio.
5. **Static prompt**: insert `docs/EDIT_REFERENCE.md`. It is about 4.7k characters, roughly 1.3–1.5k tokens, and describes the calls by argument name. Map your CLI flags to those names, or edit the examples at the merge.

## To W0 / W2a (`scripts/cloud_setup.sh`, not mine)

- `requirements.txt` pins `pandas==2.2.2`. It has no Python 3.13 wheel, and its source build fails on a fresh VM, so `cloud_setup.sh` exits there. I installed unpinned pandas by hand. Fix: drop the pin, or install `pandas>=2.2.3`.
