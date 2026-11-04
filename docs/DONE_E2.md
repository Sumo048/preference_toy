STATUS: PASS — exploration only (seeds 0–4, outside the claims); no direction edit beats B1a on k4

E2 (direction edits) summary. Details and tables: docs/explore/E2.md. Code and per-cell JSONs: explore/E2/.
1. Works: rank-4 learned writes on all three `fc2` layers, t-gated to t ≥ 0.4 (3,084 params), fit on the frozen recipe's loss with π_ref frozen. They tie B1a: D 0.283 vs 0.281 (k4), 0.447 vs 0.446 (alt), seeds 0–4, npm 4/16, 20 cells, with less between-mode mass (0.036/0.041 vs 0.046/0.067).
2. Most robust finding: gating B1a's change by t (π_ref for t < 0.3) lowers D in 19/20 cells (−0.004 ± 0.001 k4, −0.009 ± 0.002 alt). Late-decode changes put mass between modes.
3. Best alt D: the same edit with the repo's local anchor (AL), 0.399 vs B1a 0.446 (9/10 cells, Collateral 0.087 vs 0.295). The gain is the loss's: the full-weight R_hopB_bsup_neg_AL gets 0.408 on the same 6 cells (edit 0.394). AL hurts k4 (0.351 vs 0.261).
4. Probes: a label-free bad-vs-rest direction at matched conditions has AUC 0.64–0.72 globally, 0.95–0.99 per miner cluster from step 0 (block-0 `fc1` output).
5. Failed: every ablation (no effect, or off mass 0.73–0.92); every edit on a fixed probe direction (k4 0.39–0.42, alt 0.52–0.54); ungated edits (more off mass); low-rank truncation of B1a (helps k4 at rank 8, hurts alt).
6. B1a's change is low-rank (50–65% of each block matrix's energy in 4 directions), and a learned low-rank edit matches full training of the same loss. The limit is the recipe's targets and anchor, not edit capacity.
7. Phase 4 tool candidates: per-cluster contrast AUC by step and layer; a t-window sweep of a candidate edit with a label-free between-mode/atypicality rate; the singular spectrum of a trained change with rank-r truncation D.
8. Checks: in-process B1a equals cond_cell.py on all 20 cells. B1a here (cond_cell.py, npm 4/16 only): k4 0.281, alt 0.446 (map, npm 1–64: 0.318, 0.457).
9. Interface requests: none. Env: pandas 2.2.2 has no py3.13 wheel (newer pandas used); torch 2.14.0 from PyPI (CPU index blocked).
10. Branch: claude/session-e2-ivzni2 (harness-assigned), mirrored to art/explore-E2. Spend is in explore/E2/spend.jsonl (not results/spend.jsonl, which this section does not own).
11. Measured main-agent cost: about $7.6 (W0's transcripts.spend, cumulative). No subagents were used.
