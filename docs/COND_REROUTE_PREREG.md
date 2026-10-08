# Practical rerouting without same-condition negatives (pre-registration)

Written 2026-10-07 12:42 EDT, before any run on a trained base. Branch `cond-reroute`
(worktree `~/repos/toypref-cond`, forked from `reroute-wins` at 2068a83). Outputs:
`/grogu/datasets/rollout/soumojib/toypref/runs/cond/cells/`.

## Problem

`R_near_nf_nhood_neg`, the only rerouting variant that won anything (unconditional toy, NEG,
4/8 forgotten), picks its target as the nearest pi_ref decode farther than d0 from EVERY logged
negative. That works only because all negatives share one condition. A real policy is
conditional, and no two rollouts share a condition. So the method must work with:
- one negative per condition: its condition c_i, logged noise xi_i, and action a_i;
- pi_ref queried freely, at c_i or at other logged conditions (a forward pass on a logged
  observation needs no environment);
- no mode labels.

## Conditional toy

8 ring modes whose ring rotates by 1.5 spacings and grows (r = 2.5 + c) over c in [0, 1). An
action location does not identify a mode without its condition. A behaviour is the mode
index, consistent across conditions. The base is a conditional rectified flow (Fourier
condition embedding), 40k steps. Every rollout is (c ~ U, xi ~ N), so all conditions are
unique (asserted).

## Methods (all negatives-only; the `_A` arms add an unlabelled pi_ref self-distillation anchor)

| arm | target | support | extra |
|---|---|---|---|
| R_ours_neg | 4 fresh pi_ref(c_i) decodes | logged | repo's Ours |
| R_own_nhood_neg | nearest typical decode > d0 from a_i (ORACLE mode width; reference, not practical) | nhood | |
| R_basin_nhood_neg | nearest typical decode outside a_i's basin (self-calibrated clustering of 128 pi_ref(c_i) decodes) | nhood | |
| R_basin_bsup_neg | same | jitters at 0.6 that stay in a_i's basin | |
| R_hop_nhood_neg | first decode outside a_i's basin when marching xi_i along random noise directions (+ margin) | nhood | |
| R_hop_bsup_neg | same | basin support | |
| R_pu_nhood_neg | nearest typical decode that an nnPU classifier (negatives vs unlabelled rollouts, across conditions) calls good | nhood | class prior assumed known |
| R_*_tx_neg | as above | as above | each negative's noise re-decoded at 8 other logged conditions and rerouted there too |
| NPO_b10/b100_neg, KTO_b10/b100_neg | baselines (negatives only) | | |

## Predictions

- **P1:** basin/hop/pu targets almost never fall in the negative's OWN behaviour. They can fall
  in ANOTHER forgotten behaviour (basin/hop cannot know it is forgotten), at a rate rising with
  n_forget; pu reduces that.
- **P2:** the transplant's bet (pi_ref's noise -> behaviour map is stable across conditions) is
  measured by `transplant_frac_bad`. The prediction is above the base rate n_forget/8, but
  well below 1, because the ring rotates by 1.5 spacings.
- **P3:** with one negative per condition, forgetting must generalise across conditions. Without
  transplant, rerouting forgets near the logged conditions (`neg_still_forget_near`) much
  better than far from them (`_far`). More negatives help.
- **P4, main:** in the NEG setting with 4/8 forgotten, the best PRACTICAL rerouting arm beats the
  best of NPO/KTO negatives-only. This is the conditional replication of the unconditional win;
  the prediction is weak.

## Decision rule (fixed now)

Score `tv_to_ideal`, pooled seed-mean. "Practical rerouting is best in setting S at n_forget = k"
requires all four:
1. a practical rerouting arm (not `R_own`) has the lowest pooled score among S's arms;
2. it beats the best non-rerouting arm by more than 2 sem, seed-paired over (seed, n_neg);
3. it is not dominated on (forget ratio, retain TV, off mass);
4. it replicates on fresh seeds 5-9.

Exploration: seeds 0-4, n_forget {1, 2, 4}, n_neg {1, 4, 16, 64}, all arms. Confirmation: seeds
5-9, the same grid, with the arm list frozen to the exploration winner, the best baseline, B0,
R_ours_neg and R_own_nhood_neg.

## Exploration outcome and frozen confirmation list (2026-10-07 13:29 EDT)

Exploration passed criteria 1-3 only at n_forget = 4, in both NEG and NEG+U, with the same arm:
`R_hop_bsup_neg`. It scored 0.355, against NPO_b100_neg at 0.516 (paired -0.161 +- 0.032, lower in
18/20) and NPO_b10_neg_A at 0.455 in NEG+U (paired -0.099 +- 0.025). It did not pass at n_forget 1
or 2.

Confirmation (seeds 5-9, n_forget {1, 2, 4} x n_neg {1, 4, 16, 64}) uses these frozen arms:
R_hop_bsup_neg, NPO_b100_neg (best NEG baseline), NPO_b10_neg_A (best NEG+U baseline), B0_frozen,
R_ours_neg and R_own_nhood_neg. It passes iff, at n_forget = 4, R_hop_bsup_neg beats each of
the two baselines by more than 2 sem (seed-paired over seed x n_neg) and is not dominated.

## Round 2: noise veto (registered before running)

**Problem (measured).** 57% of R_hop_bsup_neg's targets land in ANOTHER forgotten behaviour
(confirmation, n_forget = 4). The hop knows only "same behaviour as this negative".

**Vetoes**, using only OTHER negatives (other conditions), all self-calibrated:
- `hopV`, noise veto: reject a hop exit whose noise is within rho of any other logged bad
  noise; rho = 0.5 x the median plain-hop exit distance over the cell's negatives.
- `hopB`, basin veto: decode the exit noise at every other negative's condition c_j and take
  the fraction f of those decodes that land in negative j's bad basin. Reject if
  f >= 0.5 x median_j f_j, where f_j is the same fraction for logged bad noise xi_j. Disabled
  when there is no other negative.
- `hopVB`: both.

Candidates are the first decode of each new behaviour along each of the 16 directions (up to 3
per direction), plus the 0.2 margin, tried in order of noise distance. If all are vetoed, fall
back to the plain hop and count it. Support = bsup in every case, so only the target changes.

**Predictions.**
- V1: every veto lowers target-in-forget below plain hop (0.57 at n_forget 4). Basin veto
  lowers it most.
- V2: the best veto arm beats R_hop_bsup_neg on tv_to_ideal at n_forget = 4, seed-paired by
  more than 2 sem.
- V3: at n_forget = 2, the best veto arm beats the best negatives-only baseline by more than
  2 sem. Plain hop did not.
- V4: with n_neg = 1 and n_forget = 1 there is no other negative of another mode, so nothing
  changes; with one negative per mode, the other modes' negatives drive the veto.

**Decision.**
- Exploration: seeds 0-4, same grid; new arms R_hopV_bsup_neg, R_hopB_bsup_neg, R_hopVB_bsup_neg.
  Existing arms are reused (the code for them is unchanged).
- Confirmation: seeds 5-9. Frozen arms: the best veto arm, R_hop_bsup_neg, NPO_b100_neg,
  NPO_b10_neg_A and B0.
- V2 and V3 are tested on the confirmation seeds.

### Round 2 exploration outcome (2026-10-07 20:16 EDT)
Best veto arm: R_hopB_bsup_neg (basin veto). At n_forget 4 it scored 0.314, against plain hop's
0.355 and NPO_b100_neg's 0.516; target-in-forget fell from 0.41 to 0.22. At n_forget 2 and 1 it
did not improve on plain hop (0.260 vs 0.256; 0.193 vs 0.188). The noise veto barely moved
targets (0.345 at n_forget 4). Confirmation runs R_hopB_bsup_neg on seeds 5-9; the comparators
already exist there from unchanged code.

## Round 3: making it work at n_forget = 1 and 2 (registered before any code or run)

**Diagnosis (from rounds 1-2, measured).**
- At n_forget = 1, hopB removes about 75% of the forgotten mass, but pays retain TV 0.13,
  off-mode 0.09 and retain flips 0.33. Its score is 0.195, against NPO's 0.142 and B0's 0.149.
- The flips show global drift of the noise -> action map from training on a few pinned rows.
- The plain self-distillation anchor (`_A` arms) removes the drift but also fights the edit
  everywhere the forgotten behaviour lives; anchored arms forgot at most ~20%.

**Mechanism: a label-free bad vote on any noise.** v(xi) is the fraction of logged negatives j
for which pi_ref(c_j, xi) lands in negative j's bad basin; the threshold is the veto's,
theta = 0.5 x median_j v_{-j}(xi_j), or 0.5 when there is only one negative. Using it on a
4,096-sample unlabelled pi_ref pool (random conditions):
- **AM (masked anchor):** distil pi_ref only on pool samples with v < theta. `map` = pinned
  (xi, pi_ref(xi)); `dist` = fresh noise.
- **mine:** pool samples with v >= theta (up to 256) become extra rerouting rows, with a hopB
  target at their own condition and nhood support.

**Arms.**
- Rerouting: R_hopB_bsup_neg_AMmap, R_hopB_bsup_neg_AMdist, R_hopB_bsup_mine_neg_AMmap,
  R_hopB_bsup_mine_neg (no anchor).
- Baselines with the same masked anchor: NPO_b10/b100_neg_AMmap, KTO_b10/b100_neg_AMmap,
  NPO_b10_neg_AMdist.
- Label-free pseudo-positive baselines (the masked anchor alone): PSEUDO_vote_map,
  PSEUDO_vote_dist.

**Predictions.**
- R1: the masked anchor cuts hopB's retain flips at n_forget 1-2 by at least half, and keeps
  most of its forgetting (forget ratio within 0.15 of hopB's).
- R2: mining lowers forgotten mass far from the logged conditions (`neg_still_forget_far`).
- R3, the main test: at n_forget = 1 AND at n_forget = 2, the best rerouting arm beats the
  best non-rerouting arm by more than 2 sem, seed-paired over (seed, n_neg). The non-rerouting
  pool includes the new masked-anchor baselines, PSEUDO and every earlier negatives-only
  baseline. Prediction: uncertain, because PSEUDO_vote_dist may win outright, as
  PSEUDO_nf_dist nearly did in the unconditional toy.
- R4: no regression at n_forget = 4 against R_hopB_bsup_neg.

**Decision.**
- Exploration: seeds 0-4, n_forget {1, 2, 4}, n_neg {1, 4, 16, 64}.
- Confirmation: seeds 5-9, with arms frozen to the best rerouting arm, the best non-rerouting
  arm (and the best masked-anchor baseline if different), R_hopB_bsup_neg, NPO_b100_neg and B0.
- R3 needs the n_forget 1 and 2 tests both to pass on the confirmation seeds. If only one
  passes, that is reported as such.

### Round 3 amendment: the user's local anchor (added before any round-3 run)

At each logged negative's condition c_i, draw 64 fresh (c_i, noise, pi_ref(c_i, noise)) triplets
from the negative's own stream. Label each with the negative's behaviour clustering (the
basins). Keep the triplets assigned to a cluster other than the negative's (and not to a gap).
Anchor them pinned ("keep sampling that action at that noise"). This uses no other negative
and no labels: other forgotten behaviours are unknown, as in practice.

Arms:
- R_hop_bsup_neg_AL: plain hop + local anchor; uses nothing from other negatives.
- R_hopB_bsup_neg_AL: hop + basin veto + local anchor.
- R_hopB_bsup_neg_AML: local + vote-masked anchor pools combined.
- NPO_b10_neg_AL, NPO_b100_neg_AL, KTO_b10_neg_AL: the same local anchor on baselines.

These join the round-3 non-rerouting pool for R3. The predictions stand. One added prediction:
- R5: the local anchor cuts retain flips near the logged conditions, but less at other
  conditions, because it only constrains n_forget x n_neg conditions.

### Forget-set layouts (user request, added before any round-3 run)

Until now the forgotten set was the first n_forget modes of a per-seed random permutation. At
n_forget = 4 (half the modes) we now also evaluate four fixed layouts, each rotated by a per-seed
random offset o on the ring:
- `contig`: o, o+1, o+2, o+3 (one side). The two end modes have 1 retained neighbour; the inner
  ones have 0.
- `alt`: o, o+2, o+4, o+6 (alternating). Every forgotten mode has 2 retained neighbours.
- `pairs`: o, o+1, o+4, o+5 (mix). 1 retained neighbour each.
- `b3p1`: o, o+1, o+2, o+5 (mix: block of 3 + 1).

Arms: every baseline (B0, NPO/KTO b10/b100 neg-only, NPO_b10_neg_A), R_ours_neg, R_hop_bsup_neg,
R_hopB_bsup_neg, and every round-3 arm. Seeds 0-4, n_neg {1, 4, 16, 64}.

Predictions:
- L1: plain hop is worst on `contig`, whose inner modes' nearest exits are forgotten, and best on
  `alt`. The basin veto's gain over plain hop is largest on `contig`.
- L2: the confirmed n_forget = 4 win (hopB vs the best neg-only baseline) holds on every layout.
  Prediction weakest for `contig`.

The confirmation (seeds 5-9) covers every layout whose exploration passes criteria 1-3.

### Round 3 + layouts: exploration outcome and frozen confirmation (before any seed-5-9 run of these arms)

In exploration (`results/cond/r3_exploration.md`) a rerouting arm passed criteria 1-3 in all
seven settings: random layout at n_forget 1, 2 and 4, and the contig / alt / pairs / b3p1 layouts at
n_forget 4. The winning arm differed by setting, and per-setting picks over about 20 rerouting
arms are a selection effect.

**A SINGLE recipe is therefore frozen, chosen by worst case across the seven settings:**
`R_hopB_bsup_neg_AMdist` = hop + basin veto + basin support + vote-masked density anchor.
- In exploration it wins 6/7 settings, by -0.009 (random 1), -0.003 (random 2), -0.043
  (random 4), -0.007 (contig), -0.043 (pairs) and -0.028 (b3p1). It loses on alt (+0.054).
- Its mean gap to the best other arm is -0.011, and its worst-case gap +0.054 is the
  smallest of any rerouting arm.

**Confirmation:** seeds 5-9, n_neg {1, 4, 16, 64}, each setting. The frozen comparator is that
setting's best non-rerouting arm in exploration:
- random 1: NPO_b10_neg_AMdist;
- random 2: NPO_b10_neg_AL;
- random 4: PSEUDO_vote_dist;
- contig: NPO_b10_neg_AMdist;
- alt: KTO_b10_neg_AMmap;
- pairs: PSEUDO_vote_dist;
- b3p1: PSEUDO_vote_dist.

Pass per setting iff the recipe is lower by more than 2 sem (seed-paired, 20 runs) and is not
dominated. **"Works at 1-2 forgotten" requires random 1 AND random 2 to pass.**

Secondary arms (reported, not tested): R_hopB_bsup_mine_neg (runner-up), and the user's
local-anchor arms R_hopB_bsup_neg_AL and R_hop_bsup_neg_AL. NPO_b100_neg and B0 run as references.
Prediction: random 1, random 2 and contig are likely to FAIL the 2-sem test, since their
exploration margins were within about 3 sem.
