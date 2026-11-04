# DEV_LOG

Dated development decisions (PLAN.md 4.5: after PREREG, each change is an amendment with its reason).

## 2026-11-04, W1-C: code arms

### D1 variant list (fixed; PLAN 9.2 "a fixed list of 3 variants (set in Phase 2)")

D1 takes the cluster's best primitive from the branch table, applies it to each member's logged
decode (xi_i, c_i) under the current version, keeps the members whose primitive decode leaves
B_i, and builds BC rows to those rescue targets with toypref basin supports
(`train.rescue_rows(..., support="bsup")`). It then submits, in this order, until the gate
accepts one (at most 3 gate checks per cluster, the same as an LLM session's submissions):

| # | Name | Parameters | Steps | Anchor | Support |
|---|---|---|---|---|---|
| 1 | `all_5000` | all | 5000 | `dist` (vote-masked density) | `bsup` |
| 2 | `all_1500` | all | 1500 | `dist` | `bsup` |
| 3 | `late_5000` | `blocks.2`, `out` | 5000 | `dist` | `bsup` |

Reasons:

- Variant 1 is the frozen recipe's training budget and anchor with the branch tool's targets in
  place of the hop/veto targets. It is the direct "distill the rescue" baseline for A3, so it goes
  first.
- The gate's likely failure modes for a first variant are drift (test 3), other forgotten modes
  (test 4) and between-mode mass (test 5): all signs of too much change. Variant 2 keeps
  everything and trains 3.3x shorter, the cheapest way to trade target removal for locality.
- Variant 3 gets locality from the parameter subset instead of the step count: only the last
  block and the head move, so early-step routing that serves retained modes is untouched. It
  mirrors the A1 surgeon's most natural restricted `targeted_train` call.
- All three keep the default locality penalty (`dist`) and `bsup` supports, because those are
  the frozen recipe's choices on seeds 0-4; changing them as well would confound the list.
- The list is fixed now (before any D1 run) and goes into PREREG unchanged unless Phase 3 records
  an amendment.

### R0 controller

- Detector: logistic regression on the current version's `blocks.2` activation (128-d) at the
  step where the best primitive acts (primitive.k: k_commit or one of the two sampled steps before
  it, 0 for a hop). PLAN 9.2 says "at k_commit"; the detector must fire where the primitive
  acts, so it uses primitive.k, which equals k_commit whenever the best primitive is at k_commit.
- Labels without the oracle: 64 mining-stream decodes at each member's c_i (label: action in
  B_i) plus 2,048 at uniform c (label 1 only if c is within 0.02 of a member j and the action is
  in B_j).
- Candidates: probability thresholds 0.5, 0.7, 0.3 (up to 3 gate checks per cluster).
- Implementation: `ControlledNet` wraps the current net and turns the primitive into an exact
  velocity change (push: -alpha*u/(h*dt) for steps k..k+h-1; hop: evaluate at xi+tau*u and
  subtract tau*u/dt at step 0), so the plain Euler decoder runs it. Where the detector is off the
  output is bit-identical. The fire decision is made at step k and held for steps k+1..k+h-1 of
  the same decode batch.
- Controller parameters are saved to `runs/repair/<run>/r0/<version>.npz`. They are code, not a
  weight patch, so `patches.rebuild` does not rebuild R0 versions.

### B1a, B1b, B5

- B1a and B1b run toypref's `train_arm` on all negatives with the same `Ctx` as
  `scripts/cond_cell.py`, so they reproduce it bit for bit. They are one-shot: the gate does not
  decide, and `--measure_gate` logs one global-scope check.
- B5 trains the setting's 3 configs (list in SESSION_PROMPTS W1-C, equal to the top 3 of
  `reference/method_map.json` on seeds 0-4; a test checks this), gates each in global scope
  against the first atlas's open clusters, and keeps the accepted candidate with the lowest gate
  `D_new` (fallback: largest test-1 fall). With no accept it keeps π_ref.
- The trained network is stored as one patch with call `toypref_train` whose arrays are the full
  changed tensors keyed by parameter name (`docs/INTERFACE_REQUESTS_W1-C.md`).

### B4

- Per cluster: `targeted_train(params=all, objective=bc, steps=5000, anchor=dist)` on
  `toypref_rows(cluster uids, rule=hopB, support=bsup)`, from the current version. One candidate
  per cluster: a second check of the same deterministic recipe would repeat the first.

### Loop schedule (B4, R0, D1)

As PLAN 6.10 with code in place of triage and surgeon: the open cluster with the largest residual
mass that is not on cool-down; R0 and D1 need class R, so the branch tool runs on open clusters
in that order until one is class R (a class-U cluster goes on cool-down); 2-iteration cool-down
after an attempt without an accept; stop on no open cluster, 3 attempts in a row without an
accept, gate checks at 3 x (first-atlas clusters + 2), no class-R cluster left (R0, D1), or
`max_rounds` (pilot: 1).

## W2a decisions (2026-11-04, seeds 0-4, before Phase 5)

### Miner clustering rule: agglomerative (`miner.CLUSTER_RULE = "agglomerative"`)

PLAN 6.2 clusters with HDBSCAN and falls back to average linkage cut at 0.7. W1-A1 found that
HDBSCAN merges different oracle modes on k4 npm 16: the largest cluster has purity 0.33-0.52 and
cohesion about 0.25, and no HDBSCAN setting fixed it. The fallback alone gives weighted purity
0.85 on k4 n16 and 0.89-0.99 on k4 n4 and alt; its largest k4 n16 cluster still mixes modes
(purity 0.56-0.62 on 4 of 5 seeds), but less. The pilot repairs the largest cluster, so W2a uses
the fallback rule alone for the pilot and later phases. Purity is still logged to `purity.csv`
only. Phase 5 freezes this choice in PREREG.md.

### Pilot target and mode

Every pilot arm repairs the same cluster per cell, fixed by `scripts/pilot.py targets`: the open
cluster with the largest residual mass in the v0 atlas whose branch class is R (else the next class-R
cluster). Code arms get it through `scripts/run_arm.py --cluster`; A1/A2 through
`orchestrate init --pilot_cluster <id> --max_sessions 1` (no triage, one session).

### Gate calibration (PLAN Phase 3, pre-pilot check 2)

B4's first repair (the pilot target cluster, frozen recipe on its negatives, locality penalty) on
the 12 pilot cells through the gate at the development thresholds: **0 of 12 accepted**. Test 5
(between modes) failed in all 12 (rise UB 0.011-0.067), drift in 8, target in 7 (B4 on `alt`
moves mass broadly: Collateral UB 0.16-0.38). Fewer than half passed, so tests 3-5 were loosened
one step: drift 0.10 -> 0.15, other forgotten 0.01 -> 0.02, between 0.01 -> 0.02
(`gate.DRIFT_MAX`, `OTHER_RISE_MAX`, `BETWEEN_RISE_MAX`). Checked again once on the same gate
batches (the numbers are deterministic per batch): **1 of 12** would pass (k4 s0 n4). The plan
allows one loosening step only, so the pilot runs at the loosened values and the calibration
failure is reported. Numbers: `results/pilot/calibration.md`.

### Pilot improvement round (Amendment 1): changes for round r2 (prompts and tools only)

Round r1 (24 sessions) showed four problems, read from the surgeons' one-line results and reports:
1. 10 of 12 A1 sessions built a gated graft in the sandbox and spent a call on `graft_gated_path`,
   which A1 does not have. Fix: the static prompt names the arm's edit calls, and EDIT_REFERENCE
   keeps only the paragraphs of those calls (`dossier.edit_reference_for`).
2. Many sessions treated the card's c range as the target and failed test 1 (fall 0.01-0.05),
   while the gate counts the members' bad behaviour at every condition (PLAN 10.1). Fix: one
   sentence in the Gate section says so.
3. Sessions compared quick_check's drift (share of actions moved) with test 3's threshold.
   Fix: one sentence states the two numbers differ.
4. The prompt said `quick_check` is free; it costs a call, and sessions ran out of calls. Fix:
   corrected, plus "keep 2 calls for each edit you still want to make and submit". The sandbox
   note says `basin(i)` takes the row of `negatives()`, not the uid.
No limit, threshold, gate test or decision rule changed. Round r1's static prompts are archived
in `results/pilot/r1/prompts/`. EDIT_REFERENCE's synthetic budget in the cap test is 1,200 tokens.

## W2b decisions (2026-11-04, seeds 0-4, before Phase 5)

### Phase 4 tools: `contrast` and `unit_card` (`toyrepair/act_tools.py`, `toyrepair/tracks.py`)

Sources. (a) The pilot's sandbox code (the committed smoke ledgers `results/pilot/smoke_llm/*_ledger.jsonl`;
the r1/r2 run directories were not committed): surgeons sampled pi_ref at many c, labelled the target behaviour
at every c by hand (fitting the ring phase), fitted logistic probes on `blocks.1.act` and reported recall and
false-fire rate, then compared a candidate graft's target share per c bin. (b) E1-E3 (seeds 0-4): the
destination is linearly readable from step 0 per failure cluster (AUC 0.95-0.99), not with one global
direction (E2 0.64-0.72); a detector pooled over steps up to k_commit works best (E1); label-free behaviour
tracks across c give near-perfect bad/good labels at all c (E1, E3). W2a's main A2 failure was test 1 at
conditions outside the card's c range.

Design. The "target" of a cluster = decodes whose action lies in a behaviour track its members mark.
Tracks: pi_ref decodes of 512 fixed noises at 101 conditions, typical decodes clustered by toypref's basin
rule, clusters linked by mutual nearest centroids across neighbouring conditions (E3's method). Label-free:
pi_ref decodes and the logged negatives only. Both tools use 4,096 mining-pool pairs (check index 11) over
all c, so calls on different versions see the same noises.
- `contrast --layer L --k K [--cluster ID] [--version V]`: target share per c decile (also for a candidate
  version: the "target at all c" view W2a asked for), probe held-out AUC by step (0..14), a probe pooled over
  steps 0..K saved as `contrast_c<ID>_<L>_k<K>.npy` in raw activation units (fires where acts.w + b > 0, so it
  is a legal graft detector as is), its recall and false-fire rates, top 10 units by effect size at K, and the
  velocity mean difference at K. PLAN 6.5 says "decodes at nearby conditions that end outside"; the
  comparison set is all other decodes at all c instead, because the gate tests every c and a detector must
  stay off everywhere else.
- `unit_card --layer L --idx I [--version V]`: by step, the unit's mean, sd, and correlations with x_k, c,
  the final target label and whether x_hat is in the target behaviour.
Static prompt: only the two USAGE lines (and a shorter sandbox line); no exploration finding enters it.
A3 + Phase 4 static part: 4,000 tokens (cap 4,000).

Oracle diagnostic (not shown to any LLM; `k4 s1 n4`, `k4 s2 n16`, `alt s0 n16`, all open clusters): when
the members mark one track, precision of the target label against the cluster's oracle target modes is
0.99-1.00 and recall 0.94-0.99; when they mark two tracks (a cluster that mixes modes, or a member near a
boundary), precision is about 0.5. `contrast` prints how many tracks the members mark.

### Code arms in Phase 4

B4, R0 and D1 already run the full loop (`arms/base.iterate`, `max_rounds=None`). Phase 4 runs them through
`scripts/phase4.py manifest` with `jobs.py --timeout 5400`: a full recursive run can exceed rule 9's
20-minute unit (one pilot round took up to 278 s, and a run can make up to 3 x (clusters + 2) gate checks).

### Triage hand-back (harness bug, fixed in the driver)

`orchestrate record` reads the triage choice from the subagent transcript's final text. In this session the
subagents return their reply through a hand-back call, so the final text is empty and `record` fell back to
the largest-residual cluster. That happened for the first triage of 6 Phase 4 A3 runs (s0 n4, s1 n4, s3 n4,
s4 n4, s1 n16, s4 n16; the fallback matched triage's choice in 2). From then on the main agent passes the
triage JSON (and the surgeon's RESULT line) to `record --summary`. The fallback is PLAN 7.1's own rule, so
those runs stay valid; the ledger marks them `fallback: true`. W3 must pass `--summary` the same way.

### Surgeon calls that go to the background (W2b, Phase 4)

A surgeon call that runs past Bash's 120 s default (`targeted_train`, long sandbox fits) is moved to the
background by Claude Code. The surgeon then sometimes ends its turn "waiting on background work" and does not
resume: run p4_cond_k4_s1_n16_A3 session s002 sat idle for 45 min after its `targeted_train` finished. The main
agent resumed it with one message ("your background command finished; continue in the foreground"), and the
session then finished normally (accepted). Several surgeons also reported a lost call when the output of a
backgrounded sandbox command could not be read. W3: if a surgeon is idle more than 10 minutes with calls left,
send the same resume message; record it in the run's notes. No limit or prompt changes.

### Observed in Phase 4, not changed (W2b)

- Cluster membership: the card names 3 exemplar uids (PLAN 6.2) and the sandbox export holds all logged negatives,
  so surgeons of clusters with more than 3 members guess the rest (several reports: "membership is approximate").
  `contrast` labels the cluster's behaviour from all members, which partly covers this. Left as specified by PLAN
  6.2 so that Phase 4 and the grid use the same dossier; listed for the report.
- Failure mode seen most often: a gated constant push sends the whole target mass into one neighbouring mode,
  which fails test 2 (progress) or 3 (drift); surgeons who split the mass between both neighbours passed.

## W2b, after the Phase 5 commit: code baseline B6 (user request; PREREG Amendment 1)

- B6 = E3's best label-free method (`a=track60k-w=1-st=20000`, track-masked density BC), ported to
  `toyrepair/arms/b6_track.py` with E3's tracker and loop unchanged (registered in `arms.base.CODE_ARMS`).
- Reproduction against E3's stored results, seeds 0-4: `k4 s0 n4` D 0.0208, `alt s1 n16` D 0.01575: identical
  in D, Removed, Collateral, mass_off and track_keep (`results/b6_repro/`). About 6 min per cell on one thread.
- Run: all 140 cond cells of seeds 5-9 (`results/b6/manifest.jsonl`, jobs.py, 4 workers), beside B0/B1a/B1b.
  The full seed 5-9 baselines were on `art/arms` only (W2a merged it before W1-C finished); W2b copied
  `results/baselines/` from `art/arms` (621 JSON files) onto `agentic-repair-toy`.
- T1 gains T1c (A3 vs B6); Holm over six tests. The tracker is not given to the surgeon arms (no data option,
  edit call or prompt change). Note: `contrast`/`unit_card` already use the separate label-free tracker
  `toyrepair/tracks.py` to label a cluster's behaviour (since Phase 4); unchanged.
- An independent session (E4, branch `art/confirm-E4`) is running the same E3 spec on seeds 5-9 with its own
  code; W4 can cross-check B6 against it.
- Done (W2b, 2026-11-04 12:20 ET): B6 on all 140 cells of seeds 5-9, 0 failures (`results/b6/`, summary in
  `results/b6/summary.md`). Mean D 0.024 (Removed 0.996, Collateral 0.018) against B0 0.427, B1a 0.246, B1b 0.210;
  B6 lower than B1a in 35 of 35 (setting, seed) pairs. B0/B1a match PLAN 5.2's reference values. On 4 spot-checked
  cells B6 equals E4's independent run of the same spec (`art/confirm-E4`) to the last digit.
- Phase 4 final patch stacks exported to `results/phase4/patches/<run_id>/` (rebuild-checked,
  `scripts/export_patches.py`). Pilot A1/A2 stacks are not available: their run dirs were never committed by W2a
  (`results/pilot/patches/README.md`).
