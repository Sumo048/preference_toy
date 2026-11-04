STATUS: PASS-WITH-ISSUES — seed 4 needed a fit fix (lr 1e-3 + grad clip) chosen after seeing it; npm-16 rows reuse the identical npm-4 graft; no `toyrepair.transcripts` on main (own spend reader used)

E1 summary (gated grafts, exploration only; seeds 0–4; full write-up in docs/explore/E1.md):
1. Works: per-bad-track gated paths beside `out` (hard gate = logistic probe on blocks.2 SiLU hidden + x + c features, AND t-window steps 0–9), 128 units each, 2 recursive rounds, fitted by gradient through the decode on added units only. Graft identity test passes bit for bit.
2. Key numbers (mean ± s.e. over 5 seeds, npm 4 and 16): k4 D 0.115 ± 0.011 (B1a 0.281, B0 0.516), Removed 0.85, Collateral 0.041 (B1a 0.133). alt D 0.122 ± 0.017 (B1a 0.446), Removed 0.81, Collateral 0.010 (B1a 0.295). Lower than B1a on 5/5 seeds in both settings.
3. Why: π_ref's activations encode the destination at step 0 (probe AUC 0.96 vs 0.74 on x, c), and k_commit ≈ 10–12, so an early, small push suffices. Per-track detectors, balanced exits (equal fill of the good basins), and a gate window reaching k_commit each matter (ablation table).
4. Label-free fitting: a behaviour tracker links π_ref basins across c from a fixed noise set, and the logged negatives name the bad tracks (precision 1.00 vs the oracle). After that, npm does not matter: npm 4 and 16 give identical grafts.
5. Phase 4 tool ideas: behaviour tracker across c; probe AUC by step × layer; p(k)/k_commit to set the gate window; residual decomposition (gate never fired vs fired-but-failed) plus gate firing rate on retained; good-basin fill as a label-free Collateral predictor.
6. Failed: one detector for all modes; strict threshold p > 0.8; output into the residual stream after block 2; more capacity (no gain); window past k_commit (more off-mode mass). alt Removed saturates near 0.83.
7. Issues: the first sweep (K10, lr 3e-3) diverged on seed 4 (D 0.36 / 0.41); K10c fixes it (0.156 / 0.190); both reported. An early id()-keyed cache bug invalidated the first results (kept in explore/E1/results/invalid/).
8. Caveats: about 1.7 M π_ref decodes plus backprop per cell (CPU only, far above a gate budget); +23% parameters; relies on behaviour continuity in c and at least one negative per forgotten mode.
9. Interface requests: none. Env: requirements.txt pins pandas 2.2.2, which has no py3.13 wheel; install a newer pandas.
10. Measured main-agent cost: about $6.7 (Opus prices, PLAN Sec. 2; results/spend.jsonl). No subagents.
