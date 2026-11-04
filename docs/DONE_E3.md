STATUS: PASS-WITH-ISSUES — pushed to harness branch claude/session-e3-7loi2v and mirrored to art/explore-E3; pandas unpinned (2.2.2 does not build on Py 3.13)

E3 summary (targeted training and headroom; seeds 0–4 only; details in docs/explore/E3.md):
1. Works: explore/E3/e3.py reproduces scripts/cond_cell.py B1a bit for bit (k4 s0 n4, D 0.259). 540 runs on k4/alt.
2. B1a knobs do not matter: LoRA 4/16, single blocks, fc2, 20k steps, lr 3e-4, anchor weight, dist20k, mined negatives all land within ±0.015 D of B1a (0.341, seeds 0–2, npm 4/16). out/head/cond-only and unmasked/local/map anchors are worse.
3. Privileged floor (oracle BC on 20k retained π_ref samples): D 0.037 (5k steps), 0.025 (20k). Headroom below B1a ≈ 0.31.
4. Best label-free method, track-masked BC (explore/E3/tracks.py): cluster π_ref decodes on a 101-point c grid, link clusters across c into behaviour tracks, let negatives mark bad tracks, then BC on the anchor samples not in bad tracks.
   Seeds 0–4, npm 4/16 (20 cells): D 0.039 (5k) / 0.030 (20k steps, 60k pool) vs B1a 0.363, B0 0.515; lower than B1a in 20/20; Removed 0.98–0.99, Collateral 0.016–0.018. npm 1: D 0.032 (B1a 0.48).
5. Why: the toypref vote mask is noise-only and ignores the sample's c. The ring rotates with c, so on alt the vote flags 84–90% of the pool at ~50% precision and the kept anchor is unbalanced (retained-mode shares 0.01–0.29), which gives B1a's alt Collateral of 0.30. Track mask precision is 1.00 in all cells.
6. With correct data the rerouting term hurts (track + reroute 0.085 vs 0.032), and a rank-4 LoRA is enough (0.034). Output layer alone fails (0.42).
7. Phase 4 tool ideas: tracks(cluster) / track_mask data option for targeted_train; kept-set balance across tracks as a label-free mask check.
8. Caveat: tracks need separated behaviour clusters at each c; global edit, not per-cluster; outside the claims.
9. Open interface requests: none. Measured main-agent cost: $3.83 (results/spend.jsonl, via agentic-repair-toy's toyrepair/transcripts.py run from a scratch copy).
