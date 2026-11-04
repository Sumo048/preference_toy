# A3LF_PREREG: A3 with a label-free gate (arm A3-LF)

Written by session A3LF on 2026-11-04 (ET), on branch `art/a3-lf` from `origin/agentic-repair-toy`
(PREREG.md frozen state). Committed and pushed **before any seed 5-9 cell was loaded** by this session. Before
this commit the session ran only unit tests and a check on development seed 0 (`k4 s0 n4`, `alt s0 n16`). It had read no
seed 5-9 result of any arm. Every later change goes in a dated amendment at the end of this file, with its reason.

## 1. Question

A3 is not negatives-only. Its gate scores each edit with oracle labels (forgotten-mode set, mode identity),
and the surgeon sees the pass/fail result of each of the six tests and test 1's number. A3-LF asks the following.
Does A3 still beat the frozen recipe (B1a) and the oracle-selected recipe (B1b) when the gate itself uses only the
logged negatives and π_ref?

## 2. The label-free gate `toyrepair/gate_lf.py`

Same structure as `toyrepair/gate.py` (PREREG Section 2): the same batch and the same six tests, thresholds,
bootstrap and surgeon view. Only the masses change. The masses come from the label-free behaviour tracks of
`toyrepair/tracks.py` (Phase 4, unchanged). These are built from 101 x 512 π_ref decodes (mining pool, stream 9),
clustered per condition and linked across conditions. They use no oracle, no labels, no ring geometry and no mode count.

- **Batch.** N_G = 16,384 pairs (c, ξ) from the gate pool, `SeedSequence([2, cell index, check index])`, with
  check index = the run's gate-check counter. This is the same batch the oracle gate would draw. Each pair is decoded
  by π_ref, the current version and the candidate.
- **Track of an action.** `Tracks.assign(a, c)`: the track of the nearest typical π_ref track decode at the nearest
  grid condition, if it lies within 2 x eps of it; else unassigned.
- **Buckets.** These are set per check from the logged negatives only (`Tracks.marked`, radius 3 x eps).
  - Bad tracks: the tracks that hold any logged negative's action a_i at c_i.
  - Target T: the tracks the target cluster's members mark.
  - Other forgotten O: the bad tracks that are not in T.
  - Good tracks G: the other tracks that hold at least 2% of π_ref's typical track-build decodes (summed over
    all grid conditions), one bucket each.
  - Between modes U: samples assigned to no track, or to an unmarked track below that 2% share (a fragment).
- **Why the 2% floor.** The task defined G as all tracks that no negative marks. On development seed 0, that set
  held tiny fragments: 5 of 9 unmarked tracks on `k4 s0 n4` held 5-20 of 16,384 samples. Uniform mass over 9
  tracks puts the ideal at 1/9. That is below the real behaviours' share of about 0.115, so moving target mass onto
  real behaviours left D_lf unchanged, and test 2 could almost never pass. The floor is the same 2% that
  `tracks.py` applies per condition. Dev check on the gate batch, labels from π_ref:
  - `k4 s0 n4`: G = 4 tracks, the 4 retained modes exactly. T = the cluster's mode (1,892 of 1,896 samples).
    O = the other 3 forgotten modes.
  - `alt s0 n16`: G = 5 tracks, because one retained mode splits into two tracks. T and O match the forgotten
    modes.
  - Script: `scripts/a3lf_devcheck.py`.
- **Masses.** These are shares of the whole batch, over all c.
- **Tests** (all must pass; thresholds equal to `gate.py`'s frozen values):
  1. Target: the mass on T falls by ≥ 50% of its current value, and the 2.5th percentile of the paired fall is > 0.
  2. Progress: the one-sided 95% upper bound of D_lf(new) − D_lf(current) is < 0. D_lf(m) = ½ Σ_b |m_b − ideal_b|
     over all buckets (T, O, each good track, U), where ideal is uniform over the good tracks.
  3. Drift: the one-sided 95% upper bound of Collateral_lf(new, π_ref) is ≤ 0.15. Collateral_lf is the TV between
     new and π_ref over the good tracks plus U, each renormalised over those buckets.
  4. Other forgotten: the one-sided 95% upper bound of the rise in mass on O is ≤ 0.02.
  5. Between modes: the one-sided 95% upper bound of the rise in mass on U is ≤ 0.02.
  6. Unit tests: `toyrepair.edits.unit_tests(new, current)`, the same hook as the oracle gate.
- **Bootstrap.** Paired, 2,000 resamples. It is a multinomial draw over the joint (ref, current, new) bucket
  categories, the oracle gate's method, seeded from the same gate-pool seed sequence. It switches to row resampling
  if the number of categories exceeds 300,000.
- **Surgeon view.** Unchanged: pass/fail per test plus test 1's fall and its 95% interval. The fall is now in
  track mass.
- **No oracle.** `gate_lf` imports `pools`, `tracks`, `policy` and `edits` only. For an A3-LF run, the CLI backend
  calls `gate_lf` instead of `gate`. The miner skips its oracle purity log inside the loop. A test
  (`tests/test_gate_lf.py`) runs the mine and gate path of an A3-LF run in a fresh interpreter and asserts that
  `toyrepair.oracle` is never imported.

## 3. Arm A3-LF and cells

- **Arm.** A3-LF is A3 with `gate_lf` in place of `gate`. The surgeon has the same commands (`cli.ARM_COMMANDS`).
  The static prompt is the same text, and the dossier names the arm "A3". The miner, branch tool, Phase 4 tools,
  session limits (20 calls, 3 submissions), gate-check cap, stop rules, $8 run cost cap, triage and models (surgeon
  Opus, triage Sonnet) are all the same.
- **Final evaluation.** `toyrepair.eval_final`, outside the loop, after the run stops (`scripts/a3lf.py eval`).
- **Shadow oracle gate.** After a run stops, every gate_lf check is re-run with the oracle gate
  (`gate.gate_check`). The re-run uses the same check index, so the same batch, and the same current and candidate
  versions. The result goes to the ledger as a `note` record with `event: gate_shadow` (the ledger's record kinds are fixed). Oracle purity of each atlas goes to
  `purity.csv`. The surgeon never sees either. They are written after the run ends, and the dossier's ledger summary
  reads only `gate` records.
- **Cells (seeds 5-9), in this order:**
  1. Main (20 cells): `k4` and `alt`; for each, seeds 5, 6, 7, 8, 9, npm 4 then 16.
  2. Extra, if time allows (15 cells): `k1`, `contig`, `b3p1` at npm 4, seeds 5-9.
  - Up to 8 surgeon subagents run at a time.
- **Bad runs.** A failed run or a run with a transcript audit hit is marked invalid with its reason and rerun once.
  Cells not reached by the deadline (Thursday 02:30 ET) are reported as not run.
- **Results.** `results/a3lf/<setting>/s<seed>/n<npm>/`: result.json, ledger, versions, patch stack, purity.csv.

## 4. Tests

D = `eval_final` D (toypref `tv_to_ideal` on the final-evaluation bank). Comparators come from the committed results
of other sessions, on the same cells:
- B1a and B1b: `origin/art/arms` `results/baselines/`, repo code. B1b is the lowest-D method-map config on seeds 0-4
  per setting (`toyrepair.arms.b1b.best_configs`).
- A3: `origin/art/grid-*` `results/grid/.../A3/`.
- B6: `origin/art/confirm-E4`, or `results/b6/` if E4 is absent.

| Test | Comparison | Pairs | Test | Correction |
|---|---|---|---|---|
| L1 | A3-LF vs B1a | main cells: one pair per (setting, seed, npm) | paired t, one-sided (A3-LF lower) | Holm over L1, L2 at α = 0.05 |
| L2 | A3-LF vs B1b | same | paired t, one-sided (A3-LF lower) | Holm over L1, L2 |
| L3 | A3-LF vs A3 | main cells where a valid A3 run exists | paired t, two-sided | none (α = 0.05) |

- Pairs pool seed and npm (and both settings) as the task specifies. An invalid run or a cell that was not run
  drops its pair.
- **Robustness (reported, not tested for claims):**
  - the same tests with D averaged over npm per (setting, seed) (the PREREG unit);
  - the same tests over all cells including the extra settings;
  - per setting: mean ± SE of the paired difference and "lower in n of N" against B1a, B1b, A3 and B6.
  - B6 is descriptive only: it is the best label-free code method, and A3-LF is not expected to beat it.
- **Collateral.** Each test also reports the mean Collateral difference. A one-sided win counts as "supported" only
  if A3-LF's mean Collateral is not above the comparator's by more than 0.02 (PREREG Section 1 rule).
- **Gate agreement.** For every gate_lf check: does the oracle gate's decision on the same batch agree?
  - We report the agreement rate, the 2 x 2 table (lf accept/reject x oracle accept/reject) and per-test
    agreement.
  - We report the share of lf-accepted edits that the oracle gate would reject, and the final-evaluation D gap
    for those runs.
- **Cost.** Per run: surgeon + triage subagent cost (`transcripts`, same price table) and sessions. The
  main-agent cost goes to `results/spend.jsonl`.

## 5. Deviations known before the first seed 5-9 cell

1. **Subagent type.** This Claude Code session started on a branch without `.claude/agents/`, so the `surgeon` and
   `triage` subagent types are not registered here. The Agent tool reports "Agent type 'surgeon' not found".
   - Surgeons therefore run as the built-in `claude` agent type with model Opus, and triage as `claude` with model
     Sonnet.
   - The text of `.claude/agents/surgeon.md` (or `triage.md`) is put verbatim at the head of each task message,
     followed by the W3 message ("Run <run>. Session token: <tok>. Start with `toyrepair-tool prompt --token <tok>`.").
   - `tools/role_hook_dispatch.py` adds `claude` to its role set. It is the only change to `tools/`, and
     `surgeon_hook.py` is unchanged. So the same command filter applies: only Bash, one plain `toyrepair-tool` call
     per command.
   - Checked before this commit: as agent type `claude`, `cat toyrepair/cell.py` and Read were blocked.
   - This session uses the `claude` type for nothing else. The transcript audit is unchanged.
   - Effect: the role text arrives as the first user message, not as the subagent system prompt, and the
     built-in agent's own system prompt stays in place. This differs from A3, and it is part of the A3-LF vs A3
     contrast (L3).
2. **Run ids.** Runs are named `a3lf_cond_<setting>_s<seed>_n<npm>`, not `grid_..._A3`. The id appears in the
   dossier header.
