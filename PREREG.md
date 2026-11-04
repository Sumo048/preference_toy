# PREREG: agentic policy repair on the toypref flow toy (Phase 5 freeze)

Written by W2b on 2026-11-04, after Phase 4 (seeds 0-4) and before any confirmatory run (seeds 5-9).
Plan: `docs/PLAN.md` with Amendment 1. From now on every change is a dated amendment at the end of this
file, with its reason (PLAN 4.5). No result on seeds 5-9 changes a method, threshold, prompt or this file's
analysis (PLAN 9.4).

Code state frozen at the commit that adds this file on branch `agentic-repair-toy`. W3 sessions branch from it.

## 1. Claims

The unit of analysis is one (setting, seed) pair, with D averaged over the npm values the arm ran on in that
setting (PLAN 9.5). D = toypref `compute` `tv_to_ideal` on the final-evaluation bank (`toyrepair.eval_final`).

| Claim | Comparison | Test | Direction |
|---|---|---|---|
| T1a | A3 vs B1a | paired t, one-sided | A3 lower |
| T1b | A3 vs B5 | paired t, one-sided | A3 lower |
| T2 | A3 vs D1 | paired t, one-sided | A3 lower |
| T3 | A2 vs A1 | paired t, two-sided | either |
| T4 | A2 vs R0 | paired t, two-sided | either |

T1 is supported only if both T1a and T1b are supported. Holm correction over the five tests together, at
family-wise α = 0.05. Pairs: every (setting, seed) where both arms have a valid final evaluation on every npm
the comparison uses (A1/A2 comparisons: items 1-3 of Section 4; A3 comparisons: every A3 cell).

Labels (PLAN 9.5):
- T1, T2. Supported: the corrected test rejects in the claimed direction, and the arm's mean Collateral is not
  above its comparator's by more than 0.02. Unsupported: the 95% interval of the mean paired difference
  excludes an improvement larger than 0.02. Inconclusive: neither.
- T3, T4. Supported: the corrected test rejects (report the direction). Unsupported: the 95% interval of the
  difference lies inside ±0.02. Inconclusive: neither.

Robustness checks (reported, not labelled): the seed-level test (D averaged over settings and npm, n = 5);
the A3 second replicate (Section 6) in place of the first; per-setting paired differences in the method-map
format (mean ± SE over seeds and npm, "lower in n of N").

Exploratory comparisons (estimates and 95% intervals, no labels): A3 vs A3-nobranch, A3 vs A3-sonnet,
A3 vs A3-nogate on `k4`, npm 16, seeds 5-9.

Secondary metrics (PLAN 9.5): Removed, Collateral (`retain_tv`), mass between modes, fix rate per first-atlas
cluster (residual basin mass halved), cost / tokens / decodes per accepted edit, edit size (fraction of weights
changed, units added), accepted edits by failure class and by edit call, gate gain minus final gain per
accepted edit (flag > 0.05 in D).

Minimum detectable differences at 80% power (one-sided α = 0.05 / Holm position 1, paired, from Phase 4
variance on seeds 0-4): see Section 8.

## 2. Gate (code; no LLM role can change it)

- Batch: N_G = 16,384 pairs (c, ξ) per check, fresh per check: gate pool 2, check index = the run's gate-check
  counter (`toyrepair.pools`, `SeedSequence([2, cell index, check index])`). Decoded by π_ref, current, new;
  labelled by the oracle (`CondMixture.assign`).
- Intervals: paired bootstrap, 2,000 resamples, as a multinomial draw over the 9 x 9 x 9 joint label
  categories of (ref, current, new) (equivalent to row resampling; `gate._joint_bootstrap`). Bounds one-sided
  95% except test 1's two-sided 95% interval.
- Tests (all must pass): 1 target: target-mode mass falls by ≥ 50% of its current value and the 95% lower
  bound of the paired fall > 0; 2 progress: upper bound of D(new) − D(current) < 0; 3 drift: upper bound of
  Collateral(new, π_ref) ≤ **0.15**; 4 other forgotten modes: upper bound of the rise ≤ **0.02** (cluster scope
  only); 5 between modes: upper bound of the rise ≤ **0.02**; 6 unit tests: shapes, no NaN, legal detectors,
  graft identity. Thresholds 3-5 are the Phase 3 calibration values (one loosening step from 0.10/0.01/0.01,
  `docs/DEV_LOG.md`); they are frozen here.
- Scopes: cluster scope (agents, B4, R0, D1): target = the majority oracle modes of the cluster's members;
  global scope (B1a, B1b, B5): target = union over first-atlas open clusters, test 4 not applied.
- Surgeon sees: pass/fail per test and test 1's fall with its 95% interval. Nothing else.
- Power: at N_G 16,384 test 2 accepts a true 0.01 improvement in 100% of 20 batches (W1-A2).
- A3-nogate: the gate accepts every edit; every test is still measured and logged.

## 3. Harness constants

- Miner (`toyrepair/miner.py`): affinity A_ij of PLAN 6.2; **agglomerative clustering, average linkage, cut
  at distance 0.7** (`CLUSTER_RULE = "agglomerative"`; HDBSCAN merged modes, W2a); residual basin mass from 256
  mining-pool noises per member; a cluster is open while its mean residual ≥ **0.02**; card cap 150 tokens,
  atlas cap 1,500 tokens.
- Branch tool (`toyrepair/flowstate.py`): steps {0, 2, 4, 6, 8, 10, 12, 14, 16}, M = **32** renoise branches
  per member and step, +64 when the 95% Wilson interval of p(k) contains 0.2; k_commit threshold **0.2**;
  primitives at k_commit and the two sampled steps before it: `push(u, α, k, h=3)` for 8 directions u and
  α ∈ {0.5, 1, 2} × basin radius, and `hop(τ, u)` with the toypref hop search; class R if a primitive rescues
  ≥ **50%** of members, else U; table cap 40 rows.
- quick_check: 4,096 mining-pool pairs for drift.
- Phase 4 tools (`toyrepair/act_tools.py`, `toyrepair/tracks.py`): `contrast` and `unit_card`, 4,096
  mining-pool pairs (check index 11), label-free behaviour tracks (101 conditions x 512 noises), output cap
  800 tokens. Given to A1, A2, A3 and the exploratory arms alike.
- Orchestrator (PLAN 6.10): triage selects an open cluster not on cool-down; fallback = largest residual
  mass; cool-down 2 iterations after a session without an accept; stop rules: no open cluster; 3 sessions in a
  row without an accept; gate checks reach 3 × (first-atlas open clusters + 2); run cost cap spent.
- The main agent passes the triage's JSON line and the surgeon's RESULT line to `orchestrate record
  --summary` (`docs/DEV_LOG.md`, W2b: the transcript's final text can be empty).

## 4. Arms and variants

| Arm | Definition (PLAN 9.2) | Frozen details |
|---|---|---|
| B0 | π_ref | — |
| B1a | `R_hopB_bsup_neg_AMdist` on all negatives | repo `train_arm`, as `scripts/cond_cell.py` |
| B1b | lowest-D method-map config on seeds 0-4 per setting | `reference/method_map.json` (optimistic; not a claim comparator) |
| B5 | gate-selected among 3 configs per setting | k1: `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AMdist`, `PSEUDO_vote_dist`; k2: `R_hopB_bsup_mine_neg_AMmap`, `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AL`; k4, pairs, b3p1: `R_hopB_bsup_mine_neg`, `R_hopB_bsup_neg_AMdist`, `R_hopB_bsup_mine_neg_AMmap`; contig: `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AMdist`, `PSEUDO_vote_dist`; alt: `R_hop_bsup_neg`, `KTO_b10_neg_AMmap`, `NPO_b10_neg_AL`. Global scope; keeps the accepted candidate with the lowest gate D. |
| B4 | iterative frozen recipe per cluster | `targeted_train(all, bc, 5000, anchor dist)` on hopB/bsup rows of the cluster; one candidate per cluster |
| R0 | iterative sampling-time controller | logistic detector on `blocks.2` at the best primitive's step; thresholds 0.5, 0.7, 0.3 |
| D1 | branch-and-distill | variants in order: `all_5000`, `all_1500`, `late_5000` (`docs/DEV_LOG.md`) |
| A1 | agent, `targeted_train` only | |
| A2 | agent, weight writes only (`graft_gated_path`, `rank1`, `ablate_direction`, `set_units`) | |
| A3 | agent, all edit calls | |
| A3-nobranch | A3 without the branch tool and branch table | exploratory |
| A3-sonnet | A3 with the `surgeon-sonnet` subagent | exploratory |
| A3-nogate | A3 with an accept-all gate | exploratory |

Code arms B4, R0, D1 run the full loop (no round cap), with the same gate-check cap per run as the agents.

## 5. Sessions, models, costs

- Surgeon: subagent `surgeon`, model Opus (`.claude/agents/surgeon.md`); A3-sonnet: `surgeon-sonnet`, model
  Sonnet; triage: `triage`, model Sonnet. Definitions on `main`, unchanged.
- Session limits: 20 CLI calls, 3 gate submissions, 150k context tokens (checked from the transcript; a session
  over the limit is marked). At most 8 surgeon subagents at a time per cloud session.
- Static prompt: `toyrepair/prompts/surgeon.md` + `docs/EDIT_REFERENCE.md` filtered per arm + the Phase 4
  command lines; ≤ 4,000 tokens per arm. Frozen.
- Cost cap per run: **$8** (PLAN 7.6, unchanged). No grid cost cap (Amendment 1).
- Isolation: the surgeon hook (`tools/surgeon_hook.py`) plus the transcript audit; a run with an audit hit is
  invalid.

## 6. Grid (seeds 5-9) and shares

Priority items (PLAN 9.1; Amendment 1: A3 on items 1-7, A1 and A2 on items 1-3; code arms on every cell with an
LLM arm; B0, B1a, B1b on all 140 cells: already done by W1-C, `results/baselines/`):

1. `k4`, npm 4 and 16
2. `alt`, npm 4 and 16
3. `k1`, npm 4 and 16
4. `contig`, npm 4 and 16
5. `pairs` and `b3p1`, npm 4 and 16
6. `k4` and `alt`, npm 1 and 64
7. `k2`, npm 4 and 16

Each item covers seeds 5, 6, 7, 8, 9. A cell is `<setting> s<seed> n<npm>`.

SHARES_PLACEHOLDER

Execution order inside a share: the rows of its table in order; inside a row, seeds 5, 6, 7, 8, 9, and npm in
the order listed. Code arms (B4, R0, D1, B5) run through `scripts/jobs.py` for every listed cell, in the same
order. Results: `results/grid/<setting>/s<seed>/n<npm>/<arm>/` (W3 section). A failed run is marked invalid with
its reason and rerun once. Cells a share does not reach by its deadline are reported as not run, never
replaced.

Replicates (Amendment 1.3): when its share is done, each W3 session runs a second A3 replicate
(`A3_rep1`) on its share's A3 cells, in the share's order, until 23:00 ET.

## 7. What W3 must not change

The gate, its thresholds and N_G; pool seeds; the miner rule and cut; branch thresholds and primitives;
session limits; subagent models; prompts (`toyrepair/prompts/`, `docs/EDIT_REFERENCE.md`, `cli.USAGE`); the
D1 and B5 lists; this grid. Bug fixes that change no method are allowed and go in an amendment.

## 8. Phase 4 numbers used here

PHASE4_PLACEHOLDER
