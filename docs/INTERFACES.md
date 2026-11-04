# toyrepair interfaces (Wave 0 contract)

Signatures, dataclasses and units are in each module's docstrings; this page is the map. A type is
defined in the module of its owner; other modules use it, never redefine it. Run directory:
`$TOYPREF_RUNS/repair/<run id>/` (`ledger.jsonl`, `budget.json`, `patches/`, `scratch/`).

| Module | Owner | Key types and functions |
|---|---|---|
| `__init__` | W0 | sets `TOYPREF_RUNS=<repo>/runs` if unset; import `toyrepair` before `toypref` |
| `cell` | W0 (done) | `Cell`, `LoggedNegatives`, `load_cell(grid, setting, seed, npm) -> Cell`, `cell_index`, `cell.vote` (toypref `NegBank`), `cell.basins[i]`, `cell.in_bad_basin(i, x)`, `cell.public_order`, `cell.toypref_ctx()` (code arms only) |
| `pools` | W0 seeds (done), W1-A1 rest | `MINING/GATE/FINAL = 1/2/3`, `seed_sequence`, `draw(cell, pool, n, index=0) -> (c (n,), xi (n,2))` float32 |
| `policy` | W1-A1 | `Version`, `Decode`, `LayerInfo`, `CAPTURE_LAYERS`, `EDITABLE_LAYERS`, `base_version(cell)`, `decode(version, xi, c, save_states, capture, start_step)`, `velocity(...)`, `layers(version)`, `load_version` |
| `flowstate` | W1-A1 | `FlowState`, `Primitive` (`push`, `hop`), `BranchRow`, `BranchTable`, `restore_and_decode`, `renoise`, `wilson`, `k_commit`, `branch(cell, version, cluster, ...)`; inspect tool also here |
| `miner` | W1-A1 | `Cluster` (members = row positions, uids), `FailureCard`, `Atlas`, `affinity`, `cluster`, `residual_mass`, `mine(cell, version) -> Atlas`, `log_purity` (calls `oracle.purity`, CSV only) |
| `quick_check` | W1-A1 | `QuickCheckResult(still_bad, residual, drift)`, `quick_check(cell, current, candidate, cluster)` |
| `oracle/` | W1-A2 | `label(cell, a, c) -> (B,) mode or -1`, `masses`, `target_modes(cell, uids)`, `purity(cell, uids) -> float`, `forget_modes`, `retain_modes`; `metrics.Scores`, `scores_from_masses`, `toypref_compute(cell, version)`. Importers: `gate`, `eval_final`, `miner.log_purity` only |
| `gate` | W1-A2 | `GateConfig`, `GateResult` (+ `surgeon_view()`), `gate_check(cell, base, current, new, target_uids, check_index, scope)`; decodes via `policy.decode`; test 6 calls `edits.unit_tests` |
| `ledger` | W1-A2 | `LedgerRecord`, `Ledger(path).append/records/summary` |
| `budget` | W1-A2 | limits, `PRICES`, `cost_usd`, `gate_check_cap`, `SessionLimits`, `Budget` |
| `eval_final` | W1-A2 | `FinalResult`, `evaluate(cell, version, arm, target_clusters)` (toypref eval bank) |
| `edits` | W1-B | `Detector`, `GatedPath`, `EditRecord`, `EditResult`, `UnitTestResult`, `EditRejected`; `graft_gated_path / rank1 / ablate_direction / set_units(parent, new_id, ...) -> EditResult`; `unit_tests(new, current) -> UnitTestResult` (gate hook) |
| `patches` | W1-B | `Patch`, `save_patch`, `load_patch`, `apply_patch`, `rebuild(cell, run_dir, patch_ids, version_id)`, `state_sha256` |
| `train` | W1-B | `TrainSpec`, `TrainData`, `rescue_rows`, `toypref_rows`, `targeted_train(cell, parent, new_id, spec, data) -> EditResult` |
| `sandbox` | W1-B | `SandboxResult`, `export(cell, version, scratch)`, `audit_hook_source`, `run(code, scratch)` |
| `arms/` | W1-C | `ArmOutput`, `CODE_ARMS`, `get(name)`; each arm `run(cell, run) -> ArmOutput` |
| `cli`, `dossier`, `orchestrate` | W1-D | `toyrepair-tool` (no toypref/oracle import at module load), `Dossier`, `SessionToken`, `Run`, `init_run`, `next_action`, `record`, `status` |
| `transcripts` | W0 `spend`, W1-D rest | `python -m toyrepair.transcripts spend --session <S>` |

Conventions: actions and noises are `(B, 2)` float32 tensors; conditions `(B,)` float32 in [0, 1);
negatives are identified by their toypref pool uid; anything shown to an LLM uses uids or
`cell.public_order`, never toypref row order (grouped by forgotten mode). Edit calls are pure: the
caller assigns version ids, saves patches, counts budget and writes the ledger. Decodes are not
batch-size invariant on CPU (~4e-7), so bit-identity tests compare equal batches.
