# Results: 51 arms x 24 cells x 5 seeds (6,120 runs)

Written 2026-10-06 21:07 EDT from sweep job 3989925 (+ RM_bon8 rerun 3990350, see
"Corrections"). Numbers are seed means; every arm-vs-arm comparison is seed-paired (same
base, same edit set, same eval noises). Full tables: `results/report_tables.md`; every
scalar: `results/all_runs.csv`; figures: `results/figures/`. Setup: `docs/DESIGN.md`.

"forget ratio" = forgotten-mode mass / pi_ref's (0 = fully forgotten). "retain TV" =
total-variation distance between the arm's and pi_ref's distributions over the RETAINED
modes (0 = untouched). B0 has off-mode mass 0.025.

## Bottom line

1. **With 512 positives, the positives do the forgetting.** BC on T_p alone (B1) reaches
   forget ratio 0.004 at retain TV 0.050, beside the privileged oracle's 0.004 / 0.012, in
   every cell and independent of `n_neg`. No anchored arm improves on it by a practical
   margin. DPO (beta 1) forgets more than B1 without extra collateral in 22/24 cells (seed-paired, 2
   sem), but the margin is 0.004 forget ratio and 0.008 TV.
2. **Every anchored rerouting arm is worse than B1 on both axes.** Retain TV is higher in
   100% of cells, and forget ratio is worse in 75-100% of cells. Collateral *grows* with
   `n_neg`; forgetting does not improve.
3. **Rerouting has two mechanisms of failure, both measured.** (a) Its targets are
   unlabelled pi_ref decodes, and 23% of them (29% for Ours) land in a forgotten mode,
   rising to 39-45% at 4 forgotten modes. So it re-teaches the modes it should remove,
   and its forget ratio grows with `n_forget`. (b) Farthest-of-8 sends the rerouted noise
   across the ring rather than to the neighbouring modes, so retained noises change mode
   too: retain flips 0.27-0.36 against B1's 0.17.
4. **Negatives-only, where `n_neg` actually matters, no arm gets both axes.**
   - KTO forgets fully from 4 negatives per mode (0.22 to 0.004) but pushes 26-50% of the
     mass off-mode.
   - NPO (beta 100) hardly forgets, and forgets *less* with more negatives (0.67 to 0.91).
   - Rerouting forgets partially (0.35-0.56) at retain TV 0.3-0.6.
   - KL-to-ref (lambda 1) is the only rerouting variant that improves steadily with
     `n_neg` (0.80 to 0.38) at bounded collateral (TV 0.07 to 0.20).
5. **None of the support or inverse variants changes the picture.** That covers nhood,
   spherical, select, inverted ball, S-reach, noised partial inversion, and the online
   loop: all within about 0.1 of R_spread_nhood on both axes, and none beats it on both.

## 1. Anchored arms (T_p available)

| arm | forget ratio | retain TV | retain flips | off mass |
|---|---|---|---|---|
| ORACLE_retain_bc (privileged) | 0.004 | 0.012 | 0.148 | 0.023 |
| B1_filtered | 0.004 | 0.050 | 0.165 | 0.021 |
| DPO_cpl_b1 | 0.000 | 0.042 | 0.156 | 0.021 |
| B3_npo_b1 / b10 / b100 | 0.000 / 0.010 / 0.054 | 0.067 / 0.053 / 0.055 | 0.21 / 0.14 / 0.13 | 0.02 |
| B3_npo_b0.1 | 0.000 | 0.294 | 0.506 | 0.028 |
| B5_kto_b0.1 / b1 / b10 / b100 | 0.000 / 0.000 / 0.043 / 0.364 | 0.43 / 0.20 / 0.09 / 0.03 | | 0.04-0.09 |
| DPO_cpl_b25 | 0.647 | 0.022 | 0.050 | 0.033 |
| B4_naive_ga (neg. control) | 0.000 | 0.607 | 0.765 | 0.277 |
| R_ours_both | 0.187 | 0.118 | 0.295 | 0.037 |
| R_spread_log_both | 0.140 | 0.115 | 0.271 | 0.076 |
| R_spread_nhood_both | 0.202 | 0.182 | 0.364 | 0.081 |
| R_guide_{log,nhood}_both | 0.028 / 0.038 | 0.103 / 0.132 | 0.46 / 0.58 | **0.312 / 0.388** |
| inverse / reach / noised-inversion `_both` | 0.20-0.27 | 0.16-0.22 | 0.33-0.45 | 0.07-0.09 |

- **Low beta over-trains.** At beta 0.1, NPO and KTO forget completely but damage the
  retained modes (TV 0.29 / 0.43, 1.5-2 retained modes below half their mass).
- **High beta brakes.** At beta 100 (KTO) and beta 25 (DPO), the brake stops forgetting.
- **Guide arms "forget" by creating garbage.** 92% of guide targets are off-mode: the
  repulsive sampler plus a farthest-of-8 pick selects the most-repelled decode.
- **Rerouting's damage is immediate.** Its collateral appears within the first 500 steps
  (`figures/trajectories_k*.png`), so it is not an artefact of over-training.

**Rerouting vs `n_neg`** (n_forget = 2, forget ratio / retain TV):

| arm | n=1 | n=4 | n=32 |
|---|---|---|---|
| B1_filtered | 0.00 / 0.05 | 0.00 / 0.05 | 0.00 / 0.05 |
| R_ours_both | 0.09 / 0.08 | 0.22 / 0.17 | 0.19 / 0.20 |
| R_spread_log_both | 0.04 / 0.06 | 0.07 / 0.11 | 0.11 / 0.21 |
| R_spread_nhood_both | 0.16 / 0.11 | 0.16 / 0.22 | 0.13 / 0.26 |

**Rerouting vs `n_forget`** (forget ratio):

| arm | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| R_ours_both | 0.087 | 0.165 | 0.221 | 0.276 |
| R_spread_nhood_both | 0.008 | 0.123 | 0.331 | 0.346 |

This tracks the share of targets that land in a forgotten mode: 0.8% / 12% / 32% / 39%
for spread.

## 2. Negatives-only arms (no T_p)

| arm | forget ratio | retain TV | off mass | modes < 50% |
|---|---|---|---|---|
| B3_npo_b100_negonly | 0.801 | 0.021 | 0.049 | 0.0 |
| B5_kto_b100_negonly | 0.071 | 0.184 | 0.358 | 1.2 |
| R_ours_neg | 0.375 | 0.317 | 0.293 | 2.4 |
| R_spread_log_neg | 0.420 | 0.418 | 0.382 | 3.2 |
| R_spread_nhood_neg | 0.457 | 0.493 | 0.263 | 3.1 |
| R_spread_nhood_neg_kl0.1 / 0.3 / 1 | 0.49 / 0.50 / 0.53 | 0.27 / 0.23 / 0.17 | 0.17 / 0.14 / 0.11 | 1.7 / 1.1 / 0.4 |
| R_guide_{log,nhood}_neg | 0.08 / 0.07 | 0.47 / 0.51 | **0.87 / 0.87** | 5.0 |
| support / inverse variants `_neg` | 0.40-0.56 | 0.49-0.59 | 0.17-0.31 | 3.1-3.5 |
| R_online2_spread_nhood_neg (2x labels) | 0.446 | 0.499 | 0.284 | 3.2 |

**By `n_neg`** (all n_forget):

| arm | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| KTO negonly: forget | 0.22 | 0.14 | 0.05 | 0.01 | 0.00 | 0.00 |
| KTO negonly: off mass | 0.26 | 0.28 | 0.34 | 0.38 | 0.39 | 0.50 |
| NPO negonly: forget | 0.67 | 0.71 | 0.79 | 0.84 | 0.88 | 0.91 |
| R_spread_nhood_neg: forget | 0.59 | 0.48 | 0.46 | 0.42 | 0.38 | 0.42 |
| R_spread_nhood_neg_kl1: forget | 0.80 | 0.62 | 0.52 | 0.44 | 0.38 | 0.38 |
| R_spread_nhood_neg_kl1: retain TV | 0.07 | 0.13 | 0.18 | 0.21 | 0.21 | 0.20 |

- **Without an anchor, rerouting overwrites the map.** A full network trained only on a
  few pinned (noise, target) pairs collapses the retained modes onto a few curves
  (`figures/samples_s0_k2_n8.png`). This is the 2-D counterpart of the preference repo's
  negatives-only collapse.
- **The online loop buys nothing.** Its second round, at twice the label budget, matches
  one-shot R_spread_nhood_neg.
- **The inverted ball does not fail here the way it did in 600-D.** In 2-D its noises stay
  on-prior (mean norm 1.5-2.7, max 7.7 at rho 3, where the 600-D case was about 10^4x off).
  It still does no better than plain nhood, so the inverse family's problem is not only
  off-prior noise.

## 3. Where the forgotten mass goes

From the paired 9x9 transition matrices:

- **B1, NPO and DPO hand a forgotten mode's noise to its ring neighbours.** 90-95% lands
  there, against 34% for an even spread: the flow closes the gap locally.
- **Anchored rerouting sends it farther.** Only 52-81% lands in a neighbour; negatives-only
  rerouting manages 23-57%, mostly near chance.
- **Best-of-8 sends it to the reward model's favourite modes** (19% neighbour).

## 4. Best-of-N reranking (RM_bon8)

- **The reward model separates its training pairs perfectly** (accuracy 1.0), and BoN-8
  removes the forgotten modes (forget ratio 0.002).
- **BoN also reweights the retained modes by reward:** retain TV 0.45, retain flips 0.69.
- Best-of-N is reward maximisation over 8 draws, not rejection of negatives. A
  reject-and-resample filter would behave differently, but it is not the repo's baseline.

## Caveats

- **The toy has no state.** Positives drawn from the retained modes define the target
  distribution exactly, which is why B1 is near-oracle. In the robot setting, T_p only
  covers the states visited. The `n_neg` axis is therefore informative only for the
  negatives-only arms; an `n_pos` sweep (e.g. 0 / 16 / 64 / 512) is the natural next axis.
- **One LR (1e-4), fixed for every arm** by a pilot before the sweep, and 5000 steps. The
  probe trajectories show where each arm sits along its path, but the forgetting dials
  (lambda_-, lambda_+, beta) were run only at the preference repo's values or sweeps.
- **The method has no access to labels for its candidates** (faithful to the repo). A
  label-filtered target is the obvious variant to test next, given mechanism 3(a).

## Corrections

- **RM_bon8 (first sweep) was a port bug, now fixed and rerun.** The first sweep carried
  the repo's LayerNorm over to the 2-D action input. That reduced each point to about one
  bit, and training-pair accuracy averaged 0.70 (min 0.51). The LayerNorm was removed and
  all 120 RM_bon8 cells were rerun (job 3990350); only the rerun numbers appear above.
