# Robo-toy stress round (pre-registration)

Written 2026-10-12 21:51 EDT, before any outcome of this round. The only outcomes seen are the native
results of seeds 0-4 (`results/robo/`, `toypref-sdpo/results/practical_best.md`). Branch `robo-stress`
(worktree `~/repos/toypref-robo`).

## Question

Do the robot-usable methods survive the ways real labels go wrong? And is SDPO's small native gain
(+2 points clean success, seeds 0,1,2,4) caused by the labels, or by any small edit?

## Seeds

Seeds 0-2 and 4 (existing bases) and 5-9 (new bases, same recipe, job 4027730).
**Precondition (fixed now):** a seed whose frozen base has clean success < 0.5 on the `neg` group is
dropped from every analysis. Seed 3 already fails it (its base diverged at step ~10k).

## Label conditions (`src/toypref/robo/labels.py`)

Evaluation never changes. Groups, CRN noises and the replay set come from the native log.

| condition | what the method reads | experiment |
|---|---|---|
| native | every chunk of each logged negative episode (about 30% of the rows are the bad chunk) | reference |
| oracle | only the chunk that crashed or jerked (PRIVILEGED; upper reference, not a method) | E1 |
| pad6, pad18 | native + 6 or 18 good chunks per negative episode, from clean base episodes of the same setup (bad-chunk share about 11% and 5%) | E1 credit dilution |
| flip0.1, flip0.25, flip0.5 | that share of negative episodes is replaced by a clean episode of the same setup (nested) | E2 false labels |
| placebo | flip1.0: every labelled episode is clean | E2 / E5 control |
| recall0.5 | half of the negative episodes are labelled | E2 missed failures |
| cover0.25 | the negatives of a quarter of the setups only | E4 coverage |

## Arms

- **Core (every condition):** SDPO_fresh_b25, SDPO_fresh_b10, SDPO_pin_b25, NPO_b100_neg,
  KTO_b100_neg, R_hop_bsup_neg_AL, FILTER_rej. B0 is label-independent and runs once per seed.
- **Native only:**
  - strength sweep: SDPO_fresh_b{5,50,100}, NPO_b{30,300}_neg, KTO_b{30,300}_neg;
  - RAND_SDPO_fresh_b25 and RAND_NPO_b100_neg (a random edit with the same per-tensor norm as the
    trained edit).

FILTER_rej is an inference-time control:
- it samples 4 candidates per decision;
- it rejects a candidate within 0.5 x the median base-decode spread of a logged negative chunk whose
  observation is within the median negative 1-NN observation distance;
- it uses the first unrejected candidate (candidate 0 is the base's own decode).

## Metrics

**Gain G** = arm minus B0 in clean success (success with no collision and no jerk). It is seed-paired.
Its uncertainty is the sem over valid seeds. "> 0" means more than 2 sem above 0; "< 0" means more
than 2 sem below.
- Primary: G on the `neg` group.
- Secondary: G on `held`, plus bad rate, `good_broken` and `hom_tv`.

## Decision rules

- **E1 credit dilution.** An arm is *dilution-robust* if G(pad6) is not < 0 and G(pad18) is not
  < 0. It *breaks under dilution* if G(pad6) < 0. Report the slope of G across oracle, native, pad6
  and pad18.
- **E2 false labels.** An arm is *noise-robust* if G(flip0.25) is not < 0 and
  G(flip0.1) >= 0.5 x G(native). It *breaks* if G(flip0.1) < 0.
- **E2/E5 label dependence.** An arm's native gain is *caused by the labels* only if
  G(native) − G(placebo) > 2 sem.
- **E5 controls.** An arm *beats the random edit* if G(arm) − G(RAND_arm) > 2 sem (for the two arms
  with a RAND twin). Every arm is also compared with FILTER_rej on G and on `hom_tv`.
- **E5 strength.** For each family, plot clean success against behaviour change (`neg_hom_tv`)
  across β. A family is *a real preference method* only if some β satisfies both: G > 0, and the
  label-dependence test passes at that β (tested at β 25 for SDPO and β 100 for NPO/KTO, since
  placebo runs only there).
- **E4 coverage and recall.** Report G on `neg`, `noneg` and `held` for cover0.25 and recall0.5
  against native. No pass/fail rule: this is the dose map.

## Predictions (fixed now)

1. The SDPO gate makes it dilution-robust at β 25; NPO_b100, KTO_b100 and R_hop_bsup_neg_AL break
   under dilution (G(pad6) < 0).
2. SDPO_fresh_b25 is noise-robust at flip0.1.
3. **The prediction that would most change the conclusion:** SDPO_fresh_b25's placebo gain is
   positive. Pushing away from ANY logged chunk toward fresh base samples may sharpen the policy
   regardless of labels. If G(native) − G(placebo) is within 2 sem, SDPO's native gain is not
   preference learning.
4. RAND_SDPO_fresh_b25 has G within 2 sem of 0.
5. FILTER_rej has G > 0 on `neg` and G ≈ 0 on `held` (it only knows logged observations).

## Not in this round

- E3 (tail vs separate mode) needs an env knob.
- E6 (repeated rounds).
- Stage 2 (7-DoF simulator).
