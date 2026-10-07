# Where does rerouting win? Results

Pre-registration: `docs/REROUTE_WINS_PREREG.md` (2026-10-07 00:13, amendment 00:27, C1 01:48).
Exploration: seeds 0-4, n_forget {1,2,4} x n_neg {1,4,16}; jobs 3990657 / 3990658 (990 tasks).
Confirmation C1: fresh seeds 5-9, n_forget {1,2,3,4} x n_neg {1..32}; job 3991975 (1,080 runs).
Tables: `results/rw/exploration.md`, `results/rw/confirm_C1.md`. Score = tv_to_ideal (lower is
better; frozen base 0.33-0.51 depending on n_forget).

## Answer

**Rerouting is the best method in exactly one place we found, and it held on fresh seeds.**
That place has three conditions:

- **Negatives only.** No positive samples and no unlabelled anchor.
- **A large share of the behaviour is forgotten:** 4 of the 8 modes.
- **The modes are compact enough that "far from every logged negative" identifies the other
  modes.**

The winning variant is new in this round. It is `R_near_nf_nhood_neg`:

- **Target:** the nearest pi_ref decode that is farther than d0 from every logged negative and
  typical under pi_ref.
- **Support:** the logged noise plus 4 jitters.

On fresh seeds it scores 0.290, against NPO-b100 0.498, KTO-b100 0.565, NPO-b10 0.566 and KTO-b10
0.875. It beats each of them by 0.21-0.59, lower in 25-29 of 30 paired runs, and is not dominated.
It removes almost all forgotten mass (forget ratio 0.03), keeps off-mode mass low (0.056), and
pays retain TV 0.27. The original method `R_ours_neg` scores 0.471.

**Everywhere else, a baseline is at least as good:**

| regime | NEG | NEG + unlabelled anchor | POS + NEG |
|---|---|---|---|
| base, 1-2 of 8 forgotten | NPO-b100 (barely edits) | KTO-b10 + nf anchor / pseudo-positive BC | DPO b1 |
| base, 3 of 8 forgotten | rerouting ties KTO-b100 (+0.013 +- 0.047) | (not tested at 3) | (not tested at 3) |
| base, 4 of 8 forgotten | **rerouting (confirmed)** | KTO / pseudo-positive BC (pooled) | DPO b1 (pooled) |
| positives cover 1/4 or 1/2 of retained modes | as base | as base, and this beats every POS+NEG method | KTO-b100 (BC methods collapse uncovered modes) |
| 20% contaminated positives | as base | as base | NPO b1 |
| overlapping modes (std 0.45) | NPO-b100 ~ rerouting (+0.008 +- 0.022) | KTO + nf anchor | DPO b1 |
| overlapping modes (std 0.6) | KTO-b100 | KTO + nf anchor | DPO b1 |

## Pre-registered hypotheses

- **H2, biased positives: CONFIRMED.**
  - Every T_p-BC method loses the uncovered retained modes. At cover 0.25 the best POS+NEG arm
    is KTO-b100 at 0.290; B1 and DPO are worse.
  - The best method overall ignores the positives: KTO or pseudo-positive BC with the
    unlabelled nf anchor, about 0.04.
  - Rerouting with positives (near_nf, both) does better than the BC-heavy baselines (0.42 / 0.64
    vs 0.48 / 0.71 for B1, DPO and NPO-b1 at cover 0.5 / 0.25) but still trails KTO-b100
    (0.22 / 0.29), so it is not best.
- **H3, contaminated positives: CONFIRMED.** Rerouting does not win; NPO b1 leads at 0.124, and
  label-free near rerouting is 3rd at 0.159.
- **H4, unlabelled anchor with compact modes: CONFIRMED.**
  - Pseudo-positive BC (`PSEUDO_nf_dist`, 0.041) and KTO + nf anchor (0.040) are near-oracle.
  - Anchored rerouting is close but behind: 0.054, paired +0.014 +- 0.002.
- **H5, overlapping modes favour rerouting: FALSIFIED.** Rerouting falls further behind as
  modes overlap (std 0.6: 0.372 vs KTO 0.261). The d0 exclusion (4 std = 2.4) then exceeds
  the mode spacing (2.3), so the label-free filter can no longer separate a forgotten mode
  from its neighbours.
- **H6, target rules: CONFIRMED.** Nearest + label-free-filtered targets cut target-in-forget
  from 23% to about 1% and target-off-mode from 13% to 0. Both-mix rerouting goes from 0.18 to
  0.070, ahead of NPO and close to B1 (0.063) and DPO (0.056).
- **H1, scarce positives:** pending the n_pos sweep (jobs 3990444/5; aggregation 3990446).
- **C1, NEG with n_forget >= 3: NOT CONFIRMED as registered.**
  - **n_forget = 4: confirmed**, every comparison passing.
  - **n_forget = 3:** beats NPO-b10 (-0.100 +- 0.039), NPO-b100 (-0.052 +- 0.022) and KTO-b10
    (-0.168 +- 0.062), but ties KTO-b100 (+0.013 +- 0.047).
  - **Side prediction held:** at n_forget = 1 it loses to all four, and is dominated by
    NPO-b10 and KTO-b100.

## Why rerouting wins there and nowhere else

- **Negatives-only baselines fail in two opposite ways, and both get worse as more is forgotten.**
  - NPO-b100's brake stops it from editing: forget ratio 0.94 at n_forget 4.
  - KTO and NPO at lower beta do erase the forgotten modes but push mass off the data, up to
    86% off-mode for KTO-b10 at n_forget 4.
  - With 1 forgotten mode both failures are cheap. With half the modes forgotten, neither is.
- **Rerouting with the label-free near target has an explicit destination.** The destination
  is a nearby pi_ref decode that is far from every negative and typical under pi_ref. So it
  removes the forgotten mass and puts it back on the data manifold: off-mode stays at 0.06.
- **It still pays retain TV of about 0.27, from routing whole regions of noise into
  particular neighbours.** That cost is constant across n_forget, so it only wins once the
  baselines' failures cost more.
- **The win depends on the label-free filter working,** which needs separable modes. It
  shrinks at std 0.45 and reverses at std 0.6.
- **The original method (`R_ours_neg`, unlabelled random resample targets) never wins.**
  27% of its targets land in forgotten modes.

## Caveats

- **Exploratory origin.** The winning slice was found by looking. It is a confirmed result only
  for the registered test at n_forget = 4 on seeds 5-9.
- **Low ceiling.** All negatives-only methods are far from the POS+NEG or NEG+U level (0.04-0.06).
  "Best" here means best of a weak field; if unlabelled pi_ref samples can be used, an anchored
  baseline does better than rerouting.
- **Fixed hyperparameters.** d0 = 4 mode-std assumes the mode width is known, and the
  typicality threshold (k = 10, 90th percentile) was fixed before outcomes were seen.
