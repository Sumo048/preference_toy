STATUS: PASS-WITH-ISSUES — (1) the pilot decision rule had failed (W2a); work continued per Amendment 1; (2) in the first triage of 6 runs, a hand-back bug silently replaced the triage choice with the fallback; fixed in the driver by passing `record --summary`; (3) the full test suite did not finish within 25 min on this VM; the 46 tests that cover the changed code pass; (4) the remote refused one push to agentic-repair-toy; later pushes went through

# W2b summary (branch agentic-repair-toy; mirrored to claude/session-w2b-uef5f7)

1. **Phase 4 tools:** `contrast` and `unit_card` (`toyrepair/act_tools.py`). They build on label-free behaviour tracks (`toyrepair/tracks.py`, from E3/E1) and on the pilot's sandbox analyses. Outputs: target share per c decile, probe AUC by step, a probe saved as a ready graft detector, top units, and unit correlations. A1, A2 and A3 all get them. The static prompt for A3 with Phase 4 is 4,000 tokens (at the cap) and contains no exploration finding.
2. **Phase 4 (k4, npm 4/16, seeds 0-4, every arm iterative), mean D:** B0 0.516, B1a 0.281, B4 0.464, R0 0.516, D1 0.428, **A3 0.165 ± 0.024**. A3 is below B1a in 8 of 10 cells, with Collateral 0.097 and Removed 0.79. Table: `results/phase4/table.md`.
3. **A3 runs:** all 10 ended at a stop rule (4 no_open_cluster, 5 no_accept_streak, 1 cost_cap). There were 74 surgeon sessions and 42 accepted edits, every one a `graft_gated_path`, mostly on `out` with an activation-probe detector. `targeted_train` was never accepted. The most common rejection was test 2 (progress): a push sent the whole target mass into one neighbouring mode.
4. **Cost:** A3 $39.71 for 10 runs ($3.97 per run, $0.54 per session). This session's main agent: $26.82. All transcript audits were clean (0 hits).
5. **PREREG.md** is committed. It freezes the claims, the gate (0.15/0.02/0.02, N_G 16,384) and the miner (agglomerative, cut 0.7). It splits the Amendment 1 grid into 3 shares (60/50/55 LLM runs, projected $655) and fixes the order of cells and of replicates. Minimum detectable differences: T1a 0.033, T2 0.066.
6. **For W3:**
   - Pass the triage JSON and the surgeon RESULT line to `orchestrate record --summary`.
   - Resume any surgeon idle more than 10 minutes after a backgrounded call; see DEV_LOG.
   - Init every LLM run with `orchestrate init --phase4`.
   - Build the cell and track caches first (`tracks.get`).
   - Expect about 25-30 LLM runs per share by 23:00 ET.
7. **Open interface requests:** none.

## Addendum (2026-11-04, after DONE; user requests)

8. **B6 (PREREG Amendment 1):** E3's track-masked BC ported to `toyrepair/arms/b6_track.py`. It reproduces E3
   exactly on k4 s0 n4 and alt s1 n16. On all 140 cond cells of seeds 5-9 (0 failures): **D 0.024**, against
   B0 0.427, B1a 0.246, B1b 0.210, lower than B1a in 35 of 35 (setting, seed) pairs (`results/b6/summary.md`).
   T1 now includes T1c (A3 vs B6), with Holm over six tests. The surgeon arms do not get the tracker. The full seed
   5-9 baselines were copied from `art/arms` into `results/baselines/`.
9. **Patch stacks:** the final stacks of all 40 Phase 4 runs are in `results/phase4/patches/` (rebuild recipe in
   its README). The pilot A1/A2 stacks were lost with W2a's VM.
