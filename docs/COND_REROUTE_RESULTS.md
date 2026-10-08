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

## Round 3: working at 1-2 forgotten, and forget-set layouts (confirmed on seeds 5-9, job 3997602)

Pre-registration: `docs/COND_REROUTE_PREREG.md` (round 3 at 13:45, the user's local-anchor
amendment at 13:52, layouts at 13:56, frozen recipe and comparators at 14:59). Tables:
`results/cond/r3_exploration.md` and `results/cond/r3_confirm.md`.

**The recipe.** One fixed recipe, frozen by worst case across seven settings before
confirmation: `R_hopB_bsup_neg_AMdist`.
- **Targets and support:** hop + basin veto + basin support, as in round 2.
- **New, the vote-masked density anchor.** The logged negatives cast a "bad vote" on any noise:
  decode it at each negative's condition and check whether it lands in that negative's bad
  behaviour. Pinning flow matching on a 4,096-sample unlabelled pi_ref pool is restricted to
  samples the vote does NOT flag, with fresh noise (density-level).
- It uses no labels and no same-condition negatives.

**Confirmation (score = tv_to_ideal; comparator = that setting's best non-rerouting arm in exploration):**

| setting | recipe | comparator | B0 | diff | lower in | |
|---|---|---|---|---|---|---|
| random, 1 forgotten | 0.107 | 0.113 NPO_b10 + masked anchor | 0.149 | -0.006 +- 0.002 | 15/20 | **pass** |
| random, 2 forgotten | 0.166 | 0.208 NPO_b10 + local anchor | 0.270 | -0.041 +- 0.016 | 14/20 | **pass** |
| random, 4 forgotten | 0.265 | 0.283 pseudo-positive BC | 0.515 | -0.018 +- 0.005 | 17/20 | **pass** |
| pairs (o,o+1,o+4,o+5) | 0.251 | 0.288 pseudo-positive BC | 0.513 | -0.037 +- 0.007 | 18/20 | **pass** |
| block 3 + 1 | 0.256 | 0.285 pseudo-positive BC | 0.514 | -0.029 +- 0.006 | 18/20 | **pass** |
| contiguous (one side) | 0.228 | 0.230 NPO_b10 + masked anchor | 0.513 | -0.003 +- 0.003 | 13/20 | fail (tie) |
| alternating | 0.450 | 0.400 KTO_b10 + masked anchor | 0.516 | +0.050 +- 0.021 | 8/20 | **fail (loses)** |

**"Works at 1-2 forgotten" is CONFIRMED** (both pre-registered tests pass).

**Read with care:**
- **The masked anchor carries most of the gain.** NPO with the same anchor and the anchor alone
  (pseudo-positive BC) both reach 0.113 at 1 forgotten. Rerouting's margin over them is small
  there (0.006) and clearer at 2 forgotten (0.166 vs 0.177 for NPO + the same anchor).
- **Mechanistically**, the anchor removes the global drift that sank hopB at 1 forgotten: score
  0.195 -> 0.107, retain TV 0.13 -> 0.07, off-mode 0.09 -> 0.036.
- **Alternating layout: the vote breaks.** Every retained mode sits between two forgotten ones,
  and pi_ref's noise -> behaviour partition shifts by about a spacing across conditions, so a
  bad noise often decodes to a RETAINED behaviour at another condition (vote precision ~0.50).
  The vote-based anchor and veto then work against you.
- **Contiguous layout:** the recipe and NPO + masked anchor tie.

**The user's local anchor** (pi_ref triplets at each negative's condition outside its behaviour,
kept pinned; secondary arms, reported, not tested):
- **Most conservative variant:** retain TV 0.03-0.06, flips 0.09-0.17.
- **It forgets less than the masked anchor:** at 2 forgotten, forget ratio 0.53 vs 0.33 and
  score 0.195 vs 0.166.
- **Best arm of all on the alternating layout:** plain hop + local anchor `R_hop_bsup_neg_AL` scored
  0.352, against 0.400 and 0.450, precisely where the vote fails.
- **Weak on the contiguous layout** (0.38-0.48), where it only constrains the few logged conditions.

**Layout predictions:**
- **L1 (hop worst on contiguous, best on alternating): confirmed in exploration.** Plain hop
  scored 0.47 on contig vs 0.26 on alt.
- **L2 (the win holds on every layout): false.** It fails on alternating and ties on contiguous.

**Practical reading:** no single recipe wins every layout. The vote-masked recipe is the
default when forgotten behaviours cluster or are few. When the forgotten behaviours interleave
with retained ones, the local anchor (no cross-condition vote) is the safer choice. Choosing
between them needs a check of vote precision, which in practice means a small labelled audit.
