# toypref (compact) — testbed for agentic policy repair

This is a compact copy of the toypref flow toy. It holds only what `docs/PLAN.md` (Agentic Policy Repair
on the toypref Flow Toy) needs. It runs on CPU in a Claude Code cloud session.

- **Code**: `src/toypref/` is taken unchanged from branch `cond-reroute` at `c48b697` (Toypref Method Map v3).
  Every entry point in PLAN §5.1 is at the path the plan gives. Import it, and do not edit it (PLAN §4.2).
- **Full history and every other study** (robo2d, robol, SDPO, reroute-wins, …): the other branches of this repo.

## Layout

```
src/toypref/          toy code: cond/ (conditional toy, main testbed) + unconditional modules (smoke test)
scripts/              cond_base.py, cond_cell.py (conditional); train_base.py, run_cell.py (unconditional)
tests/                RNG/fairness tests (pytest, ~1 min)
ckpt/                 base checkpoints, laid out like $TOYPREF_RUNS (see below)
reference/            method_map.json = Toypref Method Map data (D, Removed, Collateral per config x setting;
                      `values` = seeds 0-4, `confirm` = seeds 5-9); method_map_tables/*.md = round tables
docs/                 PLAN.md (the task), DESIGN.md, COND_REROUTE_PREREG.md, COND_REROUTE_RESULTS.md
```

## Setup

```bash
pip install -r requirements.txt          # torch (CPU), numpy, pandas
export TOYPREF_RUNS=$PWD/runs            # default is a cluster path; always override
mkdir -p runs && cp -r ckpt/cond_base ckpt/base runs/
python -m pytest -q tests
python scripts/cond_cell.py --seed 0 --n_forget 4 --n_neg 4 --arms B0_frozen,R_hopB_bsup_neg_AMdist
```

Outputs go to `$TOYPREF_RUNS/cond/cells[_<layout>]/s<seed>_k<k>_n<npm>/<arm>.json`. Never write into `ckpt/`.

## Checkpoints (`ckpt/`)

| Path | What | Used for |
|---|---|---|
| `cond_base/s0..s9/policy.pt` (+ `report.json`) | conditional π_ref, `cond_base.py`, 40k steps | every conditional cell: dev seeds 0-4, confirmation seeds 5-9 |
| `base/s0/policy.pt` (+ `base_report.json`) | unconditional default-width base, `train_base.py` | the `base_k2` smoke test (PLAN §5.2, Phase 2) |

These are the only checkpoints the plan loads. A cell's logged negatives, pools, and arm runs are
rebuilt from π_ref and the seed, so no other files are needed. `ckpt/SHA256SUMS` holds the checksums.

**Provenance (2026-11-03).** The original cluster bases were pruned; only seed 0 of each toy survived.
All 11 checkpoints here were retrained on grogu with the unchanged code (Slurm jobs 4061455 and 4061463).
Retraining is deterministic: the new conditional s0 and unconditional s0 are **bitwise identical** to the
surviving originals, and the conditional `report.json` matches exactly. So these are the networks the
method map used, and B0/B1a should reproduce the method-map values.

**Not included: per-cell results.** The per-cell arm JSONs of the method map were pruned too. Only the
aggregates in `reference/` survive. PLAN Phase 0/1 compares against `reference/method_map.json` instead of
per-cell files (PLAN §5.1 "Otherwise" branch, except that π_ref does not need retraining).

## Not yet here (the plan expects them on the default branch before the first session)

- `.claude/agents/surgeon.md`, `surgeon-sonnet.md`, `triage.md` and `tools/surgeon_hook.py` (PLAN §7, §7.5)
- `SESSION_PROMPTS.md` (PLAN §5.1, §6.11)

Claude Code loads agent definitions at session start, so add these files before the session that uses them.
