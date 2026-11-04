STATUS: PASS-WITH-ISSUES — (1) EDIT_REFERENCE.md is 4.7k chars, estimated at 1.3–1.5k tokens, not counted with a real tokenizer; (2) cloud_setup.sh fails on a fresh VM at the pandas==2.2.2 pin (no py3.13 wheel), worked around by hand

# W1-B summary (branch art/edits, also mirrored to claude/session-w1-b-wrmkkf)

1. **Edit API** (`toyrepair/edits.py`): `graft_gated_path` (hard gate via `torch.where`, any layer of inp/blocks.{0,1,2}.fc1/fc2/out, detectors on x/t/c/inputs/cond/any activation, optional k_range), `rank1`, `ablate_direction` (pre/post), `set_units`, and `unit_tests(new, current)`, the gate's test-6 hook (shapes, finite, legal_detectors, graft_identity; 0.5 s).
2. **Tests passed:**
   - Graft identity holds on 10,000 reference flow states with the detector off, for every layer. Forced-off and naturally-off rows are bit-identical.
   - A zero-output graft gives bit-identical decodes.
   - The API rejects illegal detectors: oracle/label names, non-Detector objects, wrong size, NaN, non-causal sources, bad k_range.
3. **Patches** (`patches.py`): JSON + NPZ files with checksums. `apply_patch` is the single implementation of every edit. A 7-patch stack (graft, rank1, ablate pre/post, set_units, a second graft, targeted_train) rebuilds bit for bit, and a tampered patch is detected.
4. **targeted_train** (`train.py`):
   - Objectives: bc (toypref rows for every practical rule/support, `mine`, or your own rescue targets), npo, kto.
   - Anchors: dist, map, local, local+map, none. Parameter subsets include grafts.
   - It matches toypref `train_arm` bit for bit, on all weights, for 10 arm families (150-step checks).
   - Configured as `R_hopB_bsup_neg_AMdist` on k4 s0 npm 4 for 5,000 steps, its weights are bit-identical, and D, Removed, Collateral and mass_off equal the `scripts/cond_cell.py` JSON exactly (D 0.2588). One run takes about 100 s on 1 thread.
5. **Sandbox** (`sandbox.py`): the export holds the weights, a self-contained `model.py` (no toypref), negatives and basins.
   - `model.py` reproduces harness decodes bit for bit, grafts included.
   - It runs as uid `sandbox` in a private `/var/tmp/tr_sbx/<rand>` mirror, because the 700-mode repo holds `runs/`.
   - Tests passed as user sandbox:
     - Reading `task.py` fails, by the audit hook and also by OS permissions alone.
     - Importing toypref or toyrepair(.oracle) fails, including via importlib and sys.path.
     - subprocess, os.system, exec, fork, posix_spawn, sockets, urllib and ctypes fail.
     - A least-squares fit of a graft's `w_out` runs and is grafted successfully.
     - The export is read-only, and symlinks and other file types are not copied back.
     - The 2000-token cap (4 chars/token) and the timeout work.
   - One sandbox call costs about 2 s of overhead.
6. **Tests:** `tests/toyrepair/test_edits.py` (22), `test_train.py` (13, about 13 min in total including the 7 min slow test; set `TOYREPAIR_SKIP_SLOW=1` to skip that one), `test_sandbox.py` (28). Everything else in tests/toyrepair and tests/infra still passes.
7. **Deviations:**
   - NPO/KTO's reference is the parent version, which equals π_ref at v0.
   - Anchors are always built from π_ref and all of the cell's negatives.
   - Rule `own` and transplant are not offered, because they need oracle geometry.
   - `set_units` covers fc1 units only.
8. **Open interface requests:** `docs/INTERFACE_REQUESTS_W1-B.md`.
   - W1-A1: have `policy` use `edits.run_forward` for capture on grafted nets.
   - W1-D: the CLI wiring and the measured chars-per-token ratio.
   - W0/W2a: the pandas pin.
9. **Measured cost:** main agent about $4.8 at the last spend record; no subagents.
