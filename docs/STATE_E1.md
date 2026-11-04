# STATE E1 (gated grafts exploration)

Branch: `art/explore-E1` (also pushed to `claude/session-e1-docs-h9n8nb`, this session's designated branch).

Done
- Env: torch 2.14.0 from PyPI (CPU use), numpy 2.2.6, newest pandas (2.2.2 has no py3.13 wheel), scikit-learn.
- `explore/E1/common.py`: cell loader, decode with activations (bit-identical to CondFlow.sample), label-free mode tracker.
- `explore/E1/commit.py`: rescue curves p(k) and probe AUC by step x layer.
- `explore/E1/graft.py`: gated graft fitter (detector = logistic probe, units fitted by gradient through decode).
- Baselines B0 / B1a via `scripts/cond_cell.py` on k4 + alt, npm 4/16, seeds 0-4 (`explore/E1/run_baselines.sh`).

Running / next
- balanced-exit grafts on k4/alt s0; then sweep seeds 0-2 on 8 cells; best method on seeds 3-4.
