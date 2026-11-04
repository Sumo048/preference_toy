# STATE E1 (gated grafts exploration)

Branch: `art/explore-E1` (also pushed to `claude/session-e1-docs-h9n8nb`, this session's designated branch).

Done
- Env: torch 2.14.0 from PyPI (CPU use), numpy 2.2.6, newest pandas (2.2.2 has no py3.13 wheel), scikit-learn.
- Baselines B0/B1a on k4+alt, npm 4/16, seeds 0-4 (`explore/E1/results/baselines.json`).
- Tracker, commit curves, graft fitter, identity test, residual diagnostic.
- Config sweep on k4 s0 n4 (B, E, F, G, F2, K10, K12, Q8); F sweep seeds 0-2; K10 (2 rounds) sweep seeds 0-4.
- Seed 4 failed with K10 (unit fit diverges at lr 3e-3).

Running / next
- K10c (lr 1e-3, grad clip 1) on seed 4 + k4 s0 control; ablations (ablate.sh) on k4 s0 n4.
- If K10c fixes seed 4 without hurting s0: rerun K10c on all 20 cells (targets cached), report both.
- Fill tables in docs/explore/E1.md, write docs/DONE_E1.md, push.
