STATUS: PASS-WITH-ISSUES — HDBSCAN (the PLAN 6.2 rule) merges forgotten modes on k4 npm 16 (largest-cluster purity 0.33–0.52); the agglomerative fallback alone keeps them apart. A switch `miner.CLUSTER_RULE` is in place; the rule is unchanged.

# W1-A1 summary (branch art/harness-sim, from agentic-repair-toy be38656; mirrored to claude/session-w1-a1-sxv188)

1. **What works:**
   - `policy` (decode bit-identical to `CondFlow.sample` and the uncond sampler, hook-based capture of all 17 `CAPTURE_LAYERS`, `velocity`, `layers`, `load_version`, decode counter `policy.counter`).
   - `flowstate`: renoise, push/hop, Wilson, k_commit, the branch tool with Wilson top-up and the 40-row cap, and `inspect` (≤ 21 rows).
   - `miner`: affinity, HDBSCAN plus agglomerative fallback, residual mass, cards ≤ 150 tokens, atlas ≤ 1500 tokens, purity CSV.
   - `quick_check`, and `pools` sub-streams (residual noises, renoise, hop dirs).
   - Tests: 31 pass in `tests/toyrepair` (88 s), including branch validity at every k on 10,000 noises (bit-identical).
2. **Smoke test, uncond base_k2 s0:** 2 clusters at npm 4 and 16, purity 1.0, k_commit 12 / 14, class R.
3. **Cluster counts (PLAN rule, HDBSCAN), seeds 0–4:**
   - k4 n4: 5, 6, 5, 5, 6
   - k4 n16: 3, 4, 3, 5, 9
   - alt n4: 5, 6, 6, 7, 5
   - alt n16: 6, 6, 5, 4, 6
4. **Largest cluster: k_commit and class.** k_commit is 12 in 17 of 20 cells (14 in k4 s2 n16 and alt s2 n16; 16 in k4 s4 n4). Class is R in all 20, with a best rescue rate of 0.65–1.00. The best primitive is a 2r push at k8, or a shared hop. Full table: `results/harness_sim/SUMMARY.md`.
5. **Purity issue:**
   - Weighted purity under HDBSCAN: k4 n4 0.86, k4 n16 0.69, alt n4 0.95, alt n16 0.96.
   - On k4 n16 the largest cluster (29–48 of 64 negatives) mixes 2–3 modes, and its cohesion is low (0.25 on s0), so the card shows it without labels.
   - No HDBSCAN setting I tried fixes it (min_cluster_size 2–4 × min_samples 1, 2, default).
   - The fallback alone (average linkage, cut 0.7) gives 7–9 clusters with weighted purity 0.85 on k4 n16, and 0.89 / 0.99 / 0.99 on the other three groups. Largest-cluster k_commit is 12–16 and class is R in all 20 cells (`results/harness_sim/agglomerative/SUMMARY.md`).
   - **Decision for W2a/W2b, before Phase 5:** keep PLAN 6.2, or set `miner.CLUSTER_RULE = "agglomerative"`.
6. **Choices the PLAN left open** (in the docstrings):
   - k_commit: earliest sampled k with p(k′) < 0.2 for every k′ ≥ k; the last step if none qualifies.
   - HDBSCAN noise points become singleton clusters.
   - Push α uses the cluster's median basin radius.
   - Hops: each member's toypref hop exit becomes one primitive shared by all members. A member's own exit is not counted, because it always rescues that member.
   - Rescue: outside B_i and typical, using toypref's `_typical`.
7. **Timing (1 thread):**
   - Load a cond cell, including basins and vote: about 42 s.
   - Mine: 1.7–2.5 s.
   - Branch on a cluster of 4–48 members: 1–5 s.
   - One full report cell: 45–55 CPU s.
8. **Interface requests** (`docs/INTERFACE_REQUESTS_W1-A1.md`):
   - The purity adapter is used until W1-A2's `oracle.purity` merges.
   - Graft modules (W1-B) keep their path, expose `.base`, and are named `graft…` (or provide `forward_capture`).
   - The patch JSON needs the keys `id`, `version`, `parent_version`.
   - CLI signatures for W1-D.
9. **Setup:** pip's torch download stalled at 0.4 MB/s; a fresh curl gets 15 MB/s, so fetch the wheel with curl, then `pip install` it. `pandas==2.2.2` has no cp313 wheel and failed to build; 2.2.3 installs.
10. **Measured cost:** main agent about $3.9 (`results/spend.jsonl`); no subagents.
