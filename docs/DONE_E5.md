STATUS: PASS — K10c beats B1a by more than 2 SE in 7/7 settings on fresh seeds 5–9 (70/70 cells lower); B6 beats the graft in every cell

E5 summary (full write-up: docs/E5.md; prereg: docs/E5_PREREG.md, pushed in 111f98b before any seed 5–9 cell):
1. Confirmed: E1's label-free gated graft K10c (E1 code unchanged, chunked resumable fit) on 7 settings × seeds 5–9 × npm {4, 16}.
   One npm-4 fit per (setting, seed), evaluated at npm 4 and 16 (35 fits, 70 cells, 0 failures).
2. Test (graft − B1a in D, paired over 10 cells, PASS iff mean < −2 SE): k4 −0.104 ± 0.015, alt −0.330 ± 0.027, k1 −0.039 ± 0.005,
   contig −0.086 ± 0.019, pairs −0.125 ± 0.016, b3p1 −0.101 ± 0.023, k2 −0.068 ± 0.013. All PASS. The seed-clustered SE (n = 5)
   gives the same verdicts.
3. Graft D: 0.067 (k1) to 0.121 (k4). Removed: 0.82 (alt) to 0.97 (contig). Collateral is below B1a's in every setting.
   Fresh-seed k4 and alt match E1's dev numbers (0.121 / 0.120 vs 0.115 / 0.122).
4. Descriptive: the graft beats the oracle-selected B1b in 65/70 cells. B6 (E3/E4 track-masked BC) beats the graft in 70/70 cells
   (D 0.017–0.035 vs 0.067–0.121; Removed ≈ 0.995 vs 0.82–0.97).
5. Pre-checks on dev seed 0: chunked fit == unchunked bit for bit; full K10c on k4 s0 n4 reproduces E1's stored graft bit for bit
   (D 0.0875).
6. npm-16 rule fired once: on alt s7 n16 the tracker named an extra small bad track, so that cell got its own fit (D 0.115 vs 0.117).
7. Deviation: run_queue.py was made restart-safe after the prereg (scheduling only; method and analysis code hashes unchanged).
8. Artifacts: explore/E5/results/ (per-cell JSON, summary.json, tables.md), explore/E5/grafts/ (36 graft .pt files, which render with
   explore/E1/weights/load.py, checked by explore/E5/render_check.py), explore/E5/logs/.
9. Env: pandas 2.2.3 (2.2.2 has no py3.13 wheel); torch 2.14.0; scikit-learn 1.9.1. Compute: about 4.9 CPU-hours of fitting,
   4 workers, about 3.5 h wall.
