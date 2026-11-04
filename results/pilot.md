# Phase 3 pilot (W2a)

Cells: cond `k4` and `alt`, npm 4 and 16, seeds 0-2 (12). Arms B4, R0, D1 (code) and A1, A2 (surgeon subagents, Opus).
Each run repairs one fixed target per cell (`scripts/pilot.py targets`: the open class-R cluster with the largest residual mass in the v0 atlas; all 12 targets are class R with purity 1.0) once: one session for A1/A2, up to 3 gate checks for every arm.
Gate thresholds after calibration (`results/pilot/calibration.md`): drift 0.15, other 0.02, between 0.02 (B4's first repair passed 0/12 at the development values, 1/12 loosened).
Final evaluation: `eval_final.evaluate` (toypref eval bank) with the target cluster; D = `tv_to_ideal`, Collateral = method-map `retain_tv`.
Decision rule (PLAN Phase 3): A1 or A2 removes ≥ 50% of the target modes' mass, with D below π_ref's D and Collateral ≤ 0.10, in ≥ 8 of 12 cells.

## Outcome

- **Round r1: FAIL** (A1 0/12, A2 2/12).
- **Round r2** (one improvement round, prompts only; Amendment 1): **FAIL** (A1 0/12, A2 4/12).
- Per Amendment 1 the work continues to Phases 4-6; this failure is an escalation trigger recorded here and in DONE_W2a.md.

## Table 2 (mean over the 12 cells)

### Round r1

| arm | D | Removed | Collateral | target removed | gate accepts | cells meeting rule |
|---|---|---|---|---|---|---|
| B4 | 0.494 | 0.045 | 0.007 | 0.080 | 1/12 | 1/12 |
| R0 | 0.515 | 0.000 | 0.000 | 0.000 | 0/12 | 0/12 |
| D1 | 0.478 | 0.082 | 0.017 | 0.124 | 2/12 | 1/12 |
| A1 | 0.515 | 0.000 | 0.000 | 0.000 | 0/12 | 0/12 |
| A2 | 0.482 | 0.070 | 0.036 | 0.284 | 4/12 | 2/12 |

π_ref (B0) mean D over these cells: 0.515. Per-cell rows: `results/pilot/r1/table2.md`.

### Round r2 (code arms identical to r1: deterministic, unaffected by prompt changes)

| arm | D | Removed | Collateral | target removed | gate accepts | cells meeting rule |
|---|---|---|---|---|---|---|
| B4 | 0.494 | 0.045 | 0.007 | 0.080 | 1/12 | 1/12 |
| R0 | 0.515 | 0.000 | 0.000 | 0.000 | 0/12 | 0/12 |
| D1 | 0.478 | 0.082 | 0.017 | 0.124 | 2/12 | 1/12 |
| A1 | 0.515 | 0.000 | 0.000 | 0.000 | 0/12 | 0/12 |
| A2 | 0.453 | 0.129 | 0.062 | 0.534 | 7/12 | 4/12 |

π_ref (B0) mean D over these cells: 0.515. Per-cell rows: `results/pilot/r2/table2.md`.

## Cost of one session and one run

| round | arm | sessions | $ / session (mean ± sd) | output tok | cache write | cache read | requests | peak context | CLI calls | decodes / run | gate checks / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | A1 | 12 | 0.539 ± 0.074 | 15209 | 30306 | 415141 | 20.8 | 32114 | 19.7 | 991268 | 1.5 |
| r1 | A2 | 12 | 0.642 ± 0.116 | 18882 | 35055 | 444788 | 19.8 | 37089 | 18.8 | 326706 | 2.1 |
| r2 | A1 | 12 | 0.245 ± 0.064 | 6392 | 14732 | 214826 | 18.0 | 17443 | 19.1 | 1351699 | 2.6 |
| r2 | A2 | 12 | 0.642 ± 0.153 | 18095 | 39251 | 416703 | 18.7 | 36487 | 17.5 | 893738 | 1.8 |

A pilot run of an LLM arm is one session, so $/run = $/session. All 48 pilot sessions: $24.81. Smoke sessions: A1 $0.84, A2 $0.93.
Wall time per session: 5-25 min (8 sessions sharing 4 CPU cores; the CLI tools are CPU-bound). Code arms (1 thread, per run): B4 203 s, R0 77 s, D1 278 s wall.
No session exceeded 150k context; all transcript audits clean (0 hits).

## What the sessions showed

- A2 (weight writes) gets accepts with gated grafts on `out`: a linear detector on activations plus a push or rotation toward neighbouring good behaviour, fitted in the sandbox. r2 accepts: 7/12.
- A1 (`targeted_train` only) never passed: every setting either barely cut the target or moved most actions (drift / between). In r1, 10/12 A1 surgeons designed a graft they could not apply.
- The main failure of A2 was test 1: the target counts the members' behaviour at every condition, and surgeons that repaired only the card's c range removed too little. r2 states this; A2 accepts rose 4 -> 7.
- Accepted r2 A2 grafts remove 0.82-0.99 of the target modes, with final Collateral (retain_tv) 0.08-0.15 (the gate's drift limit is 0.15 on the PLAN Collateral), so 3 of 7 accepted r2 cells miss the rule's Collateral ≤ 0.10.
- R0 removed nothing: a fixed action-space push does not transfer across c. B4/D1 retrain whole layers and fail test 5 (between) in most cells.
- Tool issues seen: Bash's 120 s default timeout cuts long sandbox fits; `--params` accepts groups only (`blocks.2.fc2` refused); quick_check drift (share moved) ≠ gate drift.
