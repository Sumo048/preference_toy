STATUS: PASS — E3's track-masked BC confirmed on seeds 5–9, 7/7 settings; pushed to art/confirm-E4 (mirrored to claude/confirm-e4-label-free-zipbi9); pandas 2.2.3 used (2.2.2 does not build on Py 3.13)

E4 summary (details in docs/E4.md, prereg in docs/E4_PREREG.md, pushed in 069e995 before any seed 5–9 cell was loaded):
1. Frozen config: `a=track60k-w=1-st=20000`, exactly as explore/E3/e3.py runs it (wrapper explore/E4/e4.py; e3.py/tracks.py unchanged, sha256 in prereg). The wrapper reproduced E3's k4 s0 n4 JSONs field for field.
2. Grid: 7 settings (k1, k2, k4, contig, alt, pairs, b3p1) × npm 1/4/16/64 × seeds 5–9 = 140 cells, 420 runs. All complete, all units exit 0, no reruns or exclusions.
3. Test (method beats B1a by > 2 SE, paired over 20 seed×npm cells): PASS in all 7 settings, by 6.4–25 SE. Lower than B1a in 140/140 cells.
4. Method D 0.017–0.035 (B1a 0.107–0.449, B0 0.149–0.516); Removed 0.994–0.998; Collateral 0.011–0.030.
5. Independent of npm: pooled D 0.024 at npm 1, 4, 16, 64 (B1a 0.371 → 0.170).
6. B1a reproduces the method map's seed 5–9 `confirm` D and Removed for all 7 settings (3 decimals).
7. Track flag precision 1.000 in all 140 cells; kept anchor ≤ 0.5% forgotten-mode samples.
8. Weakest settings: contig and b3p1 (D 0.035/0.034, Collateral 0.030/0.026), still 6.5–7.5× below B1a.
9. Caveats unchanged from E3: exploration outside the claims; needs separated behaviour clusters per c; global edit.
