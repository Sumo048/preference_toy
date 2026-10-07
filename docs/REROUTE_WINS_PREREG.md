# Where does rerouting win? (pre-registration)

Written 2026-10-07 00:13 EDT, after the main sweep (docs/RESULTS.md) and BEFORE any run of
this round. Branch `reroute-wins` (worktree `~/repos/toypref-rwin`). Outputs:
`/grogu/datasets/rollout/soumojib/toypref/runs/rw/<regime>/cells/`.

## Why rerouting lost in the main sweep (measured)

1. Abundant, unbiased positives make BC on T_p near-oracle (B1 0.004 forget / 0.050 TV).
2. Unlabelled targets: 23-29% of reroute targets land in a forgotten mode.
3. Farthest-of-8 routes the forgotten noise ACROSS the ring (52-81% to a neighbour vs B1's 90-95%),
   moving retained noises.
4. Negatives-only rerouting trains a full network on a few pinned pairs and collapses the map.
5. The "negatives-only" KL arms actually anchored on T_p (corrected 2026-10-07).

Each regime below removes one of those advantages for the baselines or one of those failure
modes for rerouting, and asks whether rerouting then wins.

## Settings (what data an arm may use)

- **NEG**: logged negatives T_n only.
- **NEG+U**: T_n plus unlabelled pi_ref samples (always available: you can sample pi_ref).
  Anchors: `self_map` / `self_dist` (all pi_ref samples; pinned noise vs fresh noise) and
  `nf_map` / `nf_dist` (only samples farther than d0 = 4 mode-std from every logged negative,
  a label-free filter that uses only T_n). `PSEUDO_nf_*` = the anchor alone (BC on
  pseudo-positives), the label-free analogue of B1 and the baseline rerouting must beat here.
- **POS+NEG**: T_p and T_n (the `both`-mix arms, NPO/KTO/DPO with BC, RM best-of-8, KL-on-T_p arms).
- Arms flagged `extra_label_budget` (label-filtered targets, online loop) are reported but are
  not eligible to be "best" in a setting whose data they exceed.

## New rerouting variants

- target `near_nf`: nearest-to-a_neg candidate (of 32 prior decodes) farther than d0 from every
  logged negative (label-free); `spread_nf`, `resample4_nf`: farthest-of-8 / Ours with the same
  filter; `near_filt`: nearest retained-labelled candidate (extra labels).
- **Amendment 2026-10-07 00:27 EDT (before any outcome was run):** smoke diagnostics on the
  TARGETS showed `near_nf` picking pi_ref's low-density bridge samples (50% of targets off-mode;
  `spread_nf` 25%). Every `_nf` rule therefore also requires the candidate to be TYPICAL under
  pi_ref: its 10th-nearest-neighbour distance to 4,000 pi_ref samples must be within the 90th
  percentile of pi_ref's own (label-free). After the fix: 0% off-mode, 0% in forget modes, no
  fallbacks (smoke cell s0_k2_n16). No outcome metric was looked at.
- support `nhoodx_<sigma>_<K>`: wider / denser jitter (0.6 or 1.0, K = 16).
- every NEG method (NPO b10/b100, KTO b10/b100, Ours, spread log/nhood, and the nf targets) x
  every anchor.

## Regimes and predictions

| id | regime | change | prediction |
|---|---|---|---|
| H1 | scarce positives | n_pos in {4, 16, 64} (launched separately, jobs 3990444/5) | B1, DPO, NPO+BC degrade at n_pos <= 16 (BC on few samples narrows retained modes); both-mix rerouting degrades too. Best overall at n_pos = 4 comes from NEG+U, not POS+NEG. |
| H2 | biased positives | `cover=0.5`, `cover=0.25`: T_p only from that share of retained modes | every T_p-BC method loses the uncovered retained modes (retain TV >= 0.3 at cover 0.25). Both-mix rerouting also BCs T_p, so it suffers too; the winner comes from NEG+U. |
| H3 | contaminated positives | `contam=0.2`: 20% of T_p from forgotten modes | B1 relearns part of the forgotten modes; DPO/NPO counteract via T_n; rerouting-both does not. Rerouting does NOT win. |
| H4 | NEG+U, compact modes | base regime, new arms | PSEUDO_nf_dist is near-oracle (with compact, separated modes the proximity filter is a perfect labeller). Anchored rerouting with nf targets is competitive but does not beat it. |
| H5 | overlapping modes | `std=0.45`, `std=0.6` (8 modes, 2-D kept) | proximity labelling and action-space likelihood pushes both start hitting neighbouring retained modes; noise-space rerouting with local targets gains relative ground. Most likely place for a rerouting win, prediction weak. |
| H6 | target rules | near vs farthest, nf/filt vs unfiltered | near targets cut retained flips vs farthest; filtered/nf targets cut target-in-forget to ~0 and bring both-mix rerouting's forget ratio near B1's. |

## Decision rule (fixed now)

Score: `tv_to_ideal` (as in the main sweep), pooled seed-mean over the cells of the phase.
"Rerouting is the best method in regime R, setting S" iff ALL of:
1. a rerouting arm has the lowest pooled `tv_to_ideal` among the arms eligible in S;
2. it beats the best eligible NON-rerouting arm, seed-paired over (seed, cell), by more than 2 sem;
3. it is not dominated on (forget ratio, retain TV, off mass) by an eligible arm;
4. it REPLICATES: same comparison on fresh seeds 5-9 over the full 4 x 6 grid, same sign, > 2 sem.

Exploration phase: seeds 0-4, n_forget in {1, 2, 4}, n_neg in {1, 4, 16} (45 cells per regime).
Only regimes/settings that pass 1-3 in exploration go to confirmation, with the arm list frozen
(the winning rerouting arm and the best non-rerouting arm, plus B0 and the setting's references).
Everything is reported whichever way it goes, including the regimes where rerouting loses.

## Confirmation hypothesis C1 (registered 2026-10-07 01:48 EDT, from exploration data)

Exploration (seeds 0-4, base regime) failed the pooled decision rule everywhere it has been
evaluated so far, but slicing the NEG setting by `n_forget` showed a strong, consistent
pattern: with 4 of 8 modes forgotten, the four best negatives-only arms are all rerouting
arms with the label-free `_nf` targets, led by `R_near_nf_nhood_neg` (tv_to_ideal 0.180 vs
NPO-negonly b100 0.498 and KTO-negonly b100 0.512; paired vs NPO -0.318 +- 0.025), and it is
best at every n_neg (1, 4, 16). At n_forget 1-2 it loses (+0.169, +0.082). This slice was
found after looking, so it is a hypothesis, not a result.

**C1.** In the NEG setting (logged negatives only), when a large share of the behaviour is
forgotten (n_forget >= 3 of 8), `R_near_nf_nhood_neg` has lower tv_to_ideal than EVERY
non-rerouting negatives-only arm (B3_npo_b10/b100_negonly, B5_kto_b10/b100_negonly), seed-paired
over (seed, n_neg) with the difference below -2 sem, separately at n_forget = 3 and 4.
**Side prediction (registered so a loss counts too):** at n_forget = 1 it does NOT beat the
best non-rerouting arm.

Test: fresh seeds 5-9 (new base policies, job 3991963), n_forget in {1, 2, 3, 4},
n_neg in {1, 2, 4, 8, 16, 32}. Arms frozen now: R_near_nf_nhood_neg, R_near_nf_log_neg,
R_ours_nf_neg, R_ours_neg (the original method, reference), B3_npo_b10_negonly,
B3_npo_b100_negonly, B5_kto_b10_negonly, B5_kto_b100_negonly, B0_frozen. Report the
non-domination check on (forget ratio, retain TV, off mass) as well. n_forget = 3 never
appeared in exploration, so it is fully out of sample.
