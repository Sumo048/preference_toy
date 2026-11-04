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
