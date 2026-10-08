# Practical rerouting with one negative per condition: results

Pre-registration: `docs/COND_REROUTE_PREREG.md` (12:42; exploration outcome and frozen list
13:29). Exploration: seeds 0-4, job 3993459 (1,200 runs; `results/cond/exploration.md`).
Confirmation: seeds 5-9, job 3993642 (360 runs; `results/cond/confirm.md`).

## Answer

**Update (round 2): the basin-veto version `R_hopB_bsup_neg` supersedes it; see the round 2
section.** `R_hop_bsup_neg` is a version of the unconditional winner that needs no second negative
at the same condition and no labels. It is the best negatives-only method with half the behaviours
forgotten, confirmed on fresh seeds.

**What it does with one logged negative (c_i, xi_i, a_i):**
1. **Find the bad basin.** Decode 128 pi_ref samples at c_i (forward passes on the logged
   observation; no environment). Drop the atypical ones (kNN density). Cluster the rest by
   single linkage at a self-calibrated radius (3 x the median nearest-neighbour spacing). The
   cluster holding a_i is the bad basin.
2. **Hop.** Starting from the logged noise xi_i, march along 16 random noise directions until
   the decode leaves the bad basin, then step 0.2 further. The closest exit gives the target:
   the smallest change to the noise -> action map that changes behaviour.
3. **Basin support.** Draw 16 jitters around xi_i at sigma 0.6. Keep the ones (up to 8) that
   still decode into the bad basin, and train all of them, plus xi_i, to the target with ordinary
   flow matching at c_i. No anchor, no positives, no labels.

**Confirmation, n_forget = 4 (score = tv_to_ideal, lower is better; B0 0.515):**

| arm | score | forget ratio | retain TV | off-mode |
|---|---|---|---|---|
| **R_hop_bsup_neg** | **0.398** | 0.689 | 0.130 | 0.063 |
| R_own_nhood_neg (oracle mode width, reference) | 0.456 | 0.722 | 0.120 | 0.105 |
| NPO_b10_neg_A (with unlabelled pi_ref anchor) | 0.481 | 0.816 | 0.020 | 0.084 |
| NPO_b100_neg | 0.500 | 0.922 | 0.018 | 0.052 |
| R_ours_neg (repo's method) | 0.593 | 0.391 | 0.223 | 0.377 |

- Against NPO-b100 the paired difference is -0.102 +- 0.022, lower in 15/20 runs.
- Against anchored NPO-b10 it is -0.083 +- 0.016, lower in 15/20. Not dominated.
- It even beats the oracle-width rule, which knows how wide a mode is.

**It does not win when less is forgotten:**
- n_forget = 2: best on the mean (0.226 vs 0.235), but not significant.
- n_forget = 1: it loses (0.187 vs 0.142). Doing almost nothing is cheap when only 1/8 of the
  behaviour is bad, and rerouting's collateral (retain TV about 0.12) then dominates.

## What the variants taught us (exploration, n_forget = 4)

- **Hop beats basin targets** (0.355-0.369 vs 0.436-0.479). The nearest exit in NOISE space
  changes the map less than the nearest other cluster in action space.
- **Basin support helps slightly** over fixed 0.3 jitters (0.355 vs 0.369).
- **The PU classifier did not help.** Trained on a few negatives across conditions it is too
  weak: 53% of its targets were in a forgotten mode, and it scored 0.474.
- **Transplanting a bad noise to other conditions partly works.** It is still bad there 65% of
  the time at n_forget 4 (chance 50%) and 55% at n_forget 2 (chance 25%), so pi_ref's noise ->
  behaviour partition is partly stable across conditions. Transplant halves collateral (retain TV
  0.058 vs 0.124) but forgets less, and scored 0.406 vs 0.369.
- **Forgetting generalises across conditions, imperfectly.** At conditions far from every logged
  negative, 51% of the forgotten-behaviour mass survives, against 44% at nearby conditions.

## Round 2: basin veto (pre-registered 20:02; confirmed on seeds 5-9, job 3994217)

**How "bad behaviour" is detected.** The hop's only notion of bad is "the same behaviour as this
negative at this condition": the single-linkage cluster of pi_ref's own decodes at c_i that holds
a_i. A march step "leaves the bad behaviour" when its decode's nearest clustered decode lies within
eps and belongs to a different cluster. That cannot see that the other behaviour is forgotten too.

**Basin veto (`R_hopB_bsup_neg`).** It fills that gap with the other negatives, without needing
any at the same condition.
- Each candidate exit noise (the first decode of each new behaviour along each direction) is
  decoded at every OTHER negative's condition c_j.
- Count how often it lands in negative j's bad cluster there.
- Veto if that rate is at least half the median rate for the logged bad noises themselves.
- Take the nearest exit that survives.
- This needs no count of forgotten behaviours, no labels, and no shared condition. It relies on
  pi_ref's noise -> behaviour partition being partly stable across conditions (measured: 65%).

**Confirmation, seeds 5-9, n_forget = 4:**

| arm | score | forget ratio | retain TV | off | targets in a forgotten mode |
|---|---|---|---|---|---|
| **R_hopB_bsup_neg** | **0.301** | **0.391** | 0.157 | 0.070 | 0.30 |
| R_hop_bsup_neg | 0.398 | 0.689 | 0.130 | 0.063 | 0.57 |
| NPO_b10_neg_A | 0.481 | 0.816 | 0.020 | 0.084 | - |
| NPO_b100_neg | 0.500 | 0.922 | 0.018 | 0.052 | - |

- **V2 confirmed:** it beats plain hop by -0.097 +- 0.026 (lower in 17/20). It removes about 61%
  of the forgotten mass rather than 31%, at slightly more collateral (retain TV 0.157 vs 0.130).
- **It beats NPO-b100 by -0.199 +- 0.028 and anchored NPO-b10 by -0.180 +- 0.023.**
- **V3 not confirmed:** at n_forget = 2 it is best on the mean (0.224 vs NPO-b10_A 0.235) but
  -0.011 +- 0.018 is not significant.
- **At n_forget = 1 it loses** (0.195 vs 0.142).
- **V1 holds:** both vetoes lowered target-in-forget, the basin veto far more (exploration
  n_forget 4: 0.41 -> 0.22; the noise veto only to 0.37). The noise veto's self-calibrated radius
  (about 0.21) is too small to cover other bad basins.
- **Cross-condition reach:** with the veto, 30% of forgotten mass survives near the logged
  conditions and 51% far from them.

## The remaining failure, and the next idea (written before round 2; kept for the record)

57% of the winner's targets land in ANOTHER forgotten mode (confirmation, n_forget = 4). One
negative identifies the bad behaviour at its own condition, but not which neighbouring
behaviours are also bad, and that caps forgetting at about 31%.

The transplant measurement suggests a practical fix that needs no same-condition negatives: a
**noise veto**. Pool all logged bad noises xi_j (from any condition) and reject hop exits whose
noise lies close to any of them. Bad behaviours occupy partly condition-stable regions of noise
space (65% stability), so logged negatives from other conditions mark where the other bad basins
probably are. Not yet run.

## Practical requirements and caveats

- **Needs:** the logged observation, the logged noise, the ability to run pi_ref forward on that
  observation (about 1,100 decodes per negative), and nothing else. A pi0-style chunked policy
  logs its noise; the clustering runs on decoded action chunks (600-dim). Single linkage with a
  self-calibrated radius should transfer, but kNN typicality and clustering in high dimension
  are untested.
- **A weak field.** All negatives-only methods are weak here; the absolute gain is 0.10 on a
  scale where B0 is 0.515.
- **Fixed constants.** The step size, margin, jitter scale and the eps multiplier were fixed
  before outcomes and never tuned.
