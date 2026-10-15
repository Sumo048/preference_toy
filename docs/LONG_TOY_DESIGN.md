# Long-horizon robo toy: problems and test bed (design)

Written 2026-10-14 17:04 EDT. Branch `robo-long`. This defines the problems the next iteration of
algorithms must solve and the cheap test bed for them. **No new algorithm is designed here.** The
only arms run are the existing ones, to measure the baseline price of trajectory labels.

## Decisions from Soumojit (2026-10-14)

1. An episode continues after a collision.
2. Methods may use unlabelled rollouts. Risk to measure: methods that treat unlabelled non-negative
   rollouts as positives can reweight the policy's modes toward whatever rollouts were collected.
3. Window labels are the ORACLE (a toy experiment). Trajectory labels are the ACTUAL method.

## Problems

| id | problem | evidence so far | what the test bed must expose |
|---|---|---|---|
| P1 | **Temporal localisation.** A trajectory label covers the whole episode, but the bad part is a window. The cause can come before the symptom. | Pinned SDPO with oracle bad chunks: +8.0; with padded trajectories: −62 to −85. | Long episodes, symptom vs causal windows, lag from cause to symptom. |
| P2 | **Shared good behaviours.** A negative trajectory's good chunks are the same skills successful trajectories use. Pushing them down removes skills everywhere. Frequency in negatives is informative only relative to a base rate. | SDPO_fresh falls with dilution (pad18 −0.9; epoch-matched −2.7). | Episodes made of shared sub-behaviours (waypoint segments), per-segment success metric. |
| P2b | **Confounded behaviours.** A harmless behaviour that co-occurs with failures cannot be separated by more data. | — | A harmless marker (speed) whose rate is correlated with failures, by a knob. |
| P3 | **Tail vs mode.** Most failures are tails of a good mode, and some are separate modes. | Bad chunk forms its own basin only 15-28% of the time; hop fails on tails. | Tail failures (injected chunks) and mode failures (a grazing route), by a knob. |
| P4 | **Edit budget without robot evaluation.** The effect depends on steps per labelled row and β, and we cannot tune on robot success. | Fixed steps confounded the row count; β 5 −23; pinned SDPO better with fewer steps. | Every comparison uses one pre-declared budget rule. Two rules are reported (below). |
| P5 | **Closed-loop and cross-chunk effects.** Edits shift later states; some failures belong to chunk pairs; fixing one failure type can create another. | Hop: jerk down, collisions up; online loop fixed 5, made 9. | Per-failure-type rates, good episodes broken, events after the first. |
| P6 | **Label noise.** False negatives, missed failures, loose window boundaries. | SDPO tolerates 25% flips; others do not. | Flip, recall and window-offset knobs. |
| P7 | **Sparse per setup, states never repeat.** 1-3 negatives per setup. "More data" means more setups. | — | Data curve over the number of setups with negatives. |
| P8 | **Mode reweighting from unlabelled data.** Using unlabelled rollouts as positives drags mode shares toward the collected pool. | Decision 2. | Per-segment homotopy share drift vs base, and a skewed-pool knob (used once a method reads unlabelled rollouts). |

## Test bed

### Environment (`toypref.robol`, a copy of `toypref.robo` with these changes)

- **Course:** K waypoint segments (default K = 4). Each segment is a robo2d setup (1-3 disc
  obstacles); its start is the previous segment's goal, and its heading turns by U(−60°, 60°).
  A course is accepted when no obstacle comes near another segment's corridor.
- **Observation:** position, current waypoint − position, next waypoint − current waypoint, the 3
  nearest obstacles (of all segments) and the last step velocity. The waypoint advances within
  GOAL_TOL. Success means reaching the last waypoint.
- **Contact continues the episode:** a step that hits a disc stops at the contact point and slides
  along the tangent. Every contact event is recorded with its step.
- **Horizon:** 18 chunks of 8 steps.

### Failure sources (base demos, as in robo2d, rates calibrated so the per-episode bad rate is about 15-20%)

- F1 tail collision (injected chunk into the obstacle).
- F2 tail jerk (zig-zag chunk).
- F3 momentum collision (cause before symptom): at states where the demo turns near an obstacle, the
  chunk continues straight at the last velocity. The contact happens one chunk or more later.
- F4 (knob) route mode: a share of demos takes a grazing route past one obstacle, giving a separate,
  colliding mode.
- Marker (knob, P2b): the injection rate depends on demo speed with correlation ρ.

### Labels

- **Trajectory (actual method):** every chunk of each negative episode is a row.
- **Window oracles:**
  - **symptom:** chunks that contain an event onset;
  - **causal:** chunks c in [event − 3, event] whose noise, resampled 4 times with every other noise
    fixed, removes the event at least half the time;
  - **human-style:** the causal window with a random offset of −1 to +1 chunk and a width of 1-3
    chunks.
- **Unlabelled pool:** 8 base rollouts per training setup, stored for later methods.

### Data curve (P7)

512 training setups that can carry negatives, nested subsets N_s ∈ {32, 128, 512}, plus 128 training
setups with no negatives and 128 held-out setups. The base is trained once per seed, on 4000 base
courses plus every training course, so the base is the same at every N_s.

Eval panel (fixed across N_s and label type), 32 CRN rollouts per setup:
- `panel`: the 32 setups labelled at every N_s;
- `noneg`: 128 training setups without negatives;
- `held`: 128 held-out setups.

### Metrics

- Clean success (final waypoint reached, no contact, no jerk).
- Per-failure-type rates: F1/F3 contacts, jerk, F4 route share.
- **Segment success** (P2): the share of segments traversed with no event.
- Good episodes broken (paired with the base on CRN).
- **Mode drift** (P8): per-segment homotopy share TV vs base.
- **Speed shift** (P2b).
- Behaviour change on unlabelled states.
- Every arm is paired with the B0 rollout of its own task.

### Budget rules (P4)

- **R1:** steps = clip(6 × labelled rows, 500, 20000); 6 steps per row is the native robo2d ratio.
- **R2:** fixed 5000 steps.

Both are reported. A label-type comparison is valid only within one rule.

### The price of trajectory labels

For each arm, G_window(N_s) and G_traj(N_s) on `panel`, `noneg` and `held`.
- **Price** = N_traj / N_window needed to reach the same gain, interpolated in log N_s, at 50% of
  the arm's best window gain.
- **Floor** = the gap at N_s = 512.
- Collateral (segment success, good episodes broken, mode drift, speed shift) is read at equal gain.

## Validation before any baseline run (pass criteria)

1. Base clean success 0.75-0.85 at K = 4.
2. In negative trajectories the event chunks are at most 15% of the rows.
3. The causal oracle removes the event in at least half of the resamples where it names a chunk.
   Report the lag distribution.
4. At least half of the segments are multimodal (2 or more homotopy classes) under the base.
5. Replaying the logged negatives reproduces them.

## Baseline arms (existing, unchanged)

SDPO_fresh_b25, SDPO_pin_b25, NPO_b100_neg, KTO_b100_neg, plus B0. They run on trajectory labels and
on the three window oracles, at N_s ∈ {32, 128, 512}, under R1 and R2, on seeds 0-4.

## Decisions from Soumojit (2026-10-14 20:16 EDT)

1. The headline oracle is the WINDOW label: per event, the contiguous span from the earliest causal
   chunk through the symptom chunk. Symptom, causal and human-style are reported alongside.
2. Scaling laws under BOTH budget rules (R1, R2). The data curve is densified to
   N_s in {16, 32, 64, 128, 256, 512}.
3. Base calibration target confirmed: clean success 0.75-0.85, mostly contacts plus some jerk.
4. First round as proposed: SDPO_fresh_b25, SDPO_pin_b25, NPO_b100_neg, KTO_b100_neg, plus B0, on
   the default environment, seeds 0-4. The knobs (F4 route mode, P2b speed marker) come after.
