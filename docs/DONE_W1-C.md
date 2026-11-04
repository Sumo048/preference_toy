STATUS: PASS-WITH-ISSUES — R0 needs a 4-line delegation in `edits.unit_tests` (INTERFACE_REQUESTS_W1-C item 8) to pass gate test 6, and it removed almost nothing on the one cell tried

# W1-C summary (branch art/arms, also mirrored to claude/session-w1-c-rwpo6t)

1. **Phase 1 reproduction: exact.** Repo code (`scripts/cond_cell.py` through `scripts/jobs.py`), k4 and alt, seeds 0-4, npm 1/4/16/64; mean D vs `reference/method_map.json` `values`: k4 B0 0.516 (0.516), B1a 0.318 (0.318), B1b `R_hopB_bsup_mine_neg` 0.304 (0.304); alt B0 0.514 (0.514), B1a 0.457 (0.457), B1b `R_hop_bsup_neg` 0.261 (0.261). Every miss is below 0.001. Table: `results/phase1/repro.md`; cells: `results/phase1/<setting>/`.
2. **CPU time per arm run (1 thread, `train_s`, seeds 0-4):** B1a k4 61/74/128/487 s at npm 1/4/16/64, alt 118/75/128/474 s; B1b k4 53/93/175/454 s, alt 39-76 s; B0 2 s. One cell's pool build adds about 60 s. `run_arm.py` B1a k4 s0 n4 end to end: 157 s wall, D 0.2588, the same as cond_cell.
3. **Arms** B0, B1a, B1b, B4, B5, R0, D1 in `toyrepair/arms/`, one entry point `scripts/run_arm.py --arm --setting --seed --npm [--max_rounds 1] [--measure_gate]`. Each arm is `run(cell, run, svc=None)`; `svc` injects the harness functions (tests use fakes). B4, R0 and D1 share `base.iterate`: the PLAN 6.10 schedule with code in place of the LLM, with stop rules and a 2-iteration cool-down. `--max_rounds 1` is the pilot.
4. **D1 variants (fixed, `docs/DEV_LOG.md`):** `all_5000`, `all_1500`, `late_5000` (blocks.2 + out); all use BC to branch rescue targets with `bsup` supports and the `dist` anchor, submitted in that order until accepted. B5 lists equal the method map's top 3 per setting (a test checks this). B1a, B1b and B5 store their nets as `targeted_train` full-tensor patches, which W1-B's `apply_patch` rebuilds.
5. **Tests:** `tests/arms/`, 19 tests: fakes for the loop, stop rules, variants, B5 selection and the R0 controller semantics; real-cell bit identity of B1a's training vs cond_cell; B0. All pass.
6. **Integration check (local merge of art/harness-sim, harness-gate and edits; not pushed),** k4 s0 n4, development gate thresholds:
   - B4 (pilot): target fall 0.118 of 0.122, D 0.518 → 0.262. Fails only test 5 (between-mode rise UB 0.0167).
   - D1 `all_5000`: D → 0.253, fails only test 5 (0.0175). The other two variants also fail drift.
   - B5: all 3 configs fail test 5, two also fail drift. B5 kept π_ref.
   - So the Phase 3 loosening (test 5 to 0.02) decides B4 and D1. Ledgers: `results/arms_integ/`.
7. **R0 issue:**
   - Without item 8, gate test 6 rejects every R0 version, because `ControlledNet` is not a weight edit. With item 8 applied locally, all tests ran.
   - The detector (blocks.2 at k8, cluster-vote labels over all c, 512/4288 positive) and the fixed action-space push `push(d1,1r,k8,h3)` removed only 0.003 of 0.122 target mass.
   - A push in a fixed direction does not transfer across c, because the ring rotates. That is R0 as specified. W2a may want to check the fire rate.
8. **Open interface requests** (`docs/INTERFACE_REQUESTS_W1-C.md`): item 8, the `unit_tests` delegation for R0 (must do), and item 7 (`orchestrate` should call `toyrepair.arms.get(arm)` or use `run_arm.py`). Items 1-2 are resolved, and items 3-6 are already satisfied by the W1 branches.
9. **Setup note:** torch 2.14.0 PyPI wheel plus CUDA libs. The first pip run stalled at ~0.5 MB/s, and a retry ran at ~6 MB/s. Phase 1 cells at npm 64 take ~19.5 min as whole-cell units, so `phase1_repro.py` makes one unit per (cell, arm group).
10. **Baselines (step 5): complete and exact.**
    - Units: 240 = 140 cells × {B0+B1a, B1b}. B1b = B1a for k1 and contig, so those cells have one unit. 0 failed.
    - Table: `results/baselines/summary.md` (+ `.json` with every row); cells: `results/baselines/<setting>/s<seed>_k<k>_n<npm>/`.
    - All 21 (setting × arm) means for seeds 5-9 match `reference/method_map.json` `confirm` (every |miss| < 0.0005).
    - D for B0 / B1a / B1b: k1 0.149 / 0.107 / 0.107; k2 0.270 / 0.166 / 0.162; k4 0.515 / 0.265 / 0.294; contig 0.513 / 0.228 / 0.228; alt 0.516 / 0.449 / 0.262; pairs 0.513 / 0.251 / 0.188; b3p1 0.514 / 0.256 / 0.232.
    - These results landed after DONE was first pushed. **W2b/W3: merge `origin/art/arms` again (or `git checkout origin/art/arms -- results/baselines`).**
11. **Measured cost:** main agent $10.02 (cumulative, `results/spend.jsonl`); no subagents.
