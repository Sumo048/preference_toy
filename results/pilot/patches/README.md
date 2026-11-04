# Pilot (Phase 3) A1/A2 patch stacks: not available

The 48 pilot runs (rounds r1/r2) ran on W2a's VM; their run directories (`runs/repair/...`, holding `patches/`)
were not committed and are not on W2b's VM, so their patch stacks cannot be exported. What exists:
`results/pilot/r1|r2/<setting>/s<seed>/n<npm>/A1_*.json`, `A2_*.json` (final evaluation, session summaries,
version ids) and the smoke ledgers in `results/pilot/smoke_llm/`. The rebuild recipe for any committed stack is in
`results/phase4/patches/README.md`.
