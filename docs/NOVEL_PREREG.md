# Round 4: new negatives-only algorithms (pre-registration)

Written before any round-4 outcome on a full run. The only outcomes seen so far are 100-step smoke
diagnostics of the masks. Branch `novel-negonly` (worktree `~/repos/toypref-novel`, forked from
`cond-reroute`). Origin: the design workflow wf_672dc093-507 (6 designers, 18 proposals, adversarial
critics, judge); its ranking is in `docs/novel_ranking.json`.

## Data each method may use (unchanged)

- The logged negatives (c_i, xi_i, a_i): one per condition.
- Forward decodes of pi_ref at logged conditions: the 4096 unlabelled anchor-pool conditions and
  this cell's negative conditions.
- Unlabelled pool samples.
- Nothing else: no mode labels, mode width, number of forgotten modes or positives. The oracle
  labeller is used only for diagnostics (mask precision and recall).

## Protocol note

Some design agents ran pilots and probed seeds 5-7 with the oracle while they chose rules. Seeds
5-9 are therefore NOT used to confirm any round-4 method. Confirmation uses fresh base policies on
seeds 10-14 (job 4005965). Exploration uses seeds 0-4, as before.

## Arms (implementation: `src/toypref/cond/track.py`, `experiment.py`)

| arm | proposal | what it does |
|---|---|---|
| TRACK_bc | #1 | Tracks behaviours across a mutual-kNN + MST graph on the shared-noise decode distance delta. OR-seeds the negatives' components, then runs fresh-noise BC on pool samples outside them. M = 128. |
| TRACK_bc_sortc | ablation of #1 | The same, on a chain graph over sorted c. |
| TRACK_bc_M16 | budget control of #3 | TRACK with a 16-noise bank. |
| RTRACK_bc | #2 | Two banks. Enrichment seeding: X_g >= 1, X_g > N m_g, and (X_g >= 2 or N m_g < 0.5); a seed holding more than half of a negative node's mass is dropped. A sample is bad only if both banks flag it. |
| JOINT_bc | #3 | A pooled (lambda c, a) density-connectivity labeller with 16 decodes per node. lambda comes from the plateau rule over k_c in {4, ..., 128}. |
| ATYP_dist | control | BC on every typical, assigned pool sample. No negatives used. This prices the tail-trimming explanation. |
| PRA_trk | #4 | Pinned rows on the TRACK mask. Good noises keep their own decode. Bad noises are Hungarian-assigned in action space to fresh good decodes. Training nodes are the negatives plus every 8th pool node. |
| REACH_recipe | #5 | The frozen recipe with the vote replaced by a reach-weighted vote. The curve w(delta) is kernel-regressed from negative pairs. The anchor excludes queries with abstaining weight (sum of abs(w) below its 10th percentile). With one negative it reduces to the recipe. |

**Comparators:**
- The frozen recipe (R_hopB_bsup_neg_AMdist).
- R_hop_bsup_neg, the plain-hop bar on the alternating layout.
- PSEUDO_vote_dist.
- NPO_b10_neg_AMdist.
- KTO_b10_neg_AMmap.
- B0.

## Settings

The 7 conditional settings at n_neg {1, 4, 16, 64}:
- random layout at n_forget 1, 2 and 4;
- contig, alt, pairs and b3p1 at n_forget 4.

**Stress cells** (proposal #2), random layout at n_forget 2 and 4, n_neg {16, 64}:
- `flip=0.1`: 10% of negatives replaced by retained-mode rollouts;
- `nodes=256` and `nodes=64`: a sparse condition graph.

## Predictions (from the judge's test plans; fixed now)

- **P1 (TRACK_bc).** Mean score 0.05-0.09 in every setting. Recipe values on seeds 5-9:
  random-1 0.107, random-2 0.166, random-4 0.265, contig 0.228, alt 0.450, pairs 0.251,
  b3p1 0.256. Worst cell at most 0.15. Mask precision and recall at least 0.9.
- **P2.** TRACK_bc beats ATYP_dist by at least 0.03 in every setting. Otherwise the gain is mostly
  tail trimming.
- **P3 (RTRACK_bc).**
  - Clean cost at most +0.01 against TRACK_bc.
  - Under flip=0.1 it beats TRACK_bc by at least 0.03.
  - At nodes=256 TRACK stays within 0.02 of clean.
  - Report the smallest node count at which TRACK still beats the recipe in at least 16/20 cells.
- **P4 (JOINT_bc).** Non-inferior to TRACK_bc (difference at most +0.01) in at least 6/7 settings.
  The plateau picks k_c = 32 on at least 4/5 seeds.
- **P5 (PRA_trk).** Mean score within ±0.01 of TRACK_bc, with retain flips at most 0.6x TRACK_bc's.
- **P6 (REACH_recipe).** Alternating 0.28-0.36 against the recipe's 0.450. Other settings within
  ±0.01 of the recipe.

## Decision rules (confirmation, seeds 10-14)

The recipe and every comparator are rerun on seeds 10-14.

- **TRACK_bc.**
  - A setting is a WIN if the paired mean difference against the rerun recipe is at most -0.02
    AND TRACK_bc is lower in at least 16/20 cells.
  - ADOPT if at least 6/7 settings are wins, no setting is worse than the recipe by more than
    0.01, and TRACK_bc beats ATYP_dist by at least 0.03 in every setting.
  - ABANDON if 3/7 or fewer settings are wins.
- **RTRACK_bc.** Becomes the default if its clean cost is at most +0.01 against TRACK_bc AND it
  beats TRACK_bc by at least 0.03 in the flip=0.1 cells.
- **JOINT_bc.** Promoted if non-inferior to TRACK_bc (at most +0.01) in at least 6/7 settings.
- **PRA_trk.** Adopted as the delivery if its mean score is at most TRACK_bc + 0.005 in at least
  6/7 settings AND its flips fall by at least 40%.
- **REACH_recipe.** Kept if it beats the recipe on alt by at least 0.05 and no other setting is
  worse by more than 0.01.

Every arm also faces the earlier standard: beat the best non-new arm in each setting by more than
2 sem, seed-paired, without being dominated.

## Caveat fixed in advance

The toy favours tracking: a dense 1-D condition, modes that never split or merge, and clean
negatives. A win here is necessary, not sufficient, for the robot. The stress cells and a
decode-only robot pre-test come before any pi0 claim.

### Amendment (2026-10-09 22:02 EDT, from mask diagnostics only; no training outcome seen)

- **JOINT_bc fixes.** My first implementation computed eps from within-node spacing, and the
  plateau then collapsed the pooled cloud into 1 component, so every sample was flagged. It now
  follows the pilot's definition: per k_c, eps = 3 x the median 1-NN distance of the pooled cloud
  at the provisional scale lam0 = 1/(k_c gap), then lam = eps/(k_c gap).
- **JOINT_bc plateau fallback.** When no k_c gives the same count at k_c, 2k_c and 4k_c, the
  largest k_c is used. Observed counts were [0, 8, 19, 9, 8, 8], which picks k_c = 128 and gives 8
  components.
- **Mask diagnostics** (s0, three cells): TRACK and RTRACK precision 0.97-0.98, recall ~1.0, 9
  components, 0 refusals. JOINT precision 0.97, recall 1.0.
