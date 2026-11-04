# TRKNG preregistration: no-grid mask builders vs B6 (track-masked BC)

Session TRKNG, 2026-11-04. Branch `art/trkng` (from `origin/art/explore-E3`). This file and
`explore/TRKNG/frozen.json` are committed and pushed **before any seed 5–9 cell is built or loaded**.
Nothing in the method, the configs or the tests changes after this commit. The runner refuses seeds ≥ 5
unless `--stage confirm` is given, and `--stage confirm` reads only `frozen.json`.

## Question

B6 (`a=track60k-w=1-st=20000`) finds bad samples by clustering π_ref decodes at 101 grid values of c and
linking the clusters across neighbouring c. Can a mask built **without a c-grid**, from π_ref's own
activations, match B6's mask precision and final D?

Rules the builders follow (`explore/TRKNG/builders.py`, `trkng.py`): they never use c as a coordinate, a
grid or a feature, and never use the ring geometry or CondMixture. c enters only as π_ref's own input
when decoding. The inputs are the logged negatives (c, ξ, a), each decoded from its own (ξ, c), and π_ref
decodes with their hidden activations (output of `blocks.2`) at chosen Euler steps. The oracle
(`CondMixture.assign`, via `pool.mode`) is used only to score masks and in `compute()`.

## Frozen method (from stage 1, seeds 0–2, k4/alt/contig/pairs, npm 4)

Common to all builders: decode budget **N = 8,000**. The fit set is the first N samples of B6's
60,000-sample training pool (`make_pool`, purpose 90,001). The mask is applied to all 60,000 pool samples,
and the kept set (assigned and not flagged) feeds the **frozen B6 training step**, unchanged: flow-matching
BC on the kept samples with fresh noise, all weights, 20,000 steps, AdamW 1e-4 cosine (warm-up 500, floor
1e-5), betas (0.9, 0.95), grad clip 1, stream `gen(seed, P_TRAIN)`. This is the code path of
`e3.run_method` for B6's spec. The same path reproduces E3's B6 JSON for k4 s0 n4 exactly (D 0.0208,
keep 24,782).

Feature `f=s19` = activations of `blocks.2` at the last Euler step (step 19), from a decode of (ξ, c).

| Builder | Frozen config (`frozen.json`) | Rule |
|---|---|---|
| **TA** | `TA-f=s19-N8000` | k-means on the N fit activations, k ∈ {8, 12, 16, 24, 32, 48, 64} by silhouette (4,000-point subsample). A negative flags its nearest centroid. |
| **TB** | `TB-f=s19-mcs_frac=0.01-min_samples=10-typ=0.9-N8000` | PCA to 32 dims (fit on the fit set), then HDBSCAN (min_cluster_size 1% of N, min_samples 10). A negative flags the cluster of its nearest clustered fit point. Pool samples take the label of their nearest fit point; HDBSCAN noise is unassigned. |
| **TC** | `TC-f=s19-knn=10-tau_mult=1.75-typ=0.9-N8000` | kNN graph (k 10) over the PCA-32 features of the fit set. Edges are kept if shorter than τ = 1.75 × the median distance from a negative to its 10th-nearest fit point. Bad = connected components with a fit point within τ of a negative. Pool samples farther than τ from every fit point are unassigned. |

`typ=0.9` adds a label-free typicality filter in the builder's own feature space. A pool sample is
unassigned (dropped) if its distance to its 5th-nearest fit decode exceeds the 0.9 quantile of the fit
decodes' own 5th-neighbour distances. This is B6's typicality rule (k 5, q 0.9) without the grid.

**Best no-grid builder (for tests a–c): TC.**

### Development record (what was tried, and why these configs)

1. **Spec features fail.** TA on step 0 and TB/TC on steps 0, 4, 8 have mask precision 0.49–0.63.
   TB finds 2 clusters with 60–70% noise, and TC finds 1 component. Step-0 activations are smooth in the
   noise, so the noise regions that lead to different modes border each other. The fraction of kNN edges
   joining different modes is 13% at step 0, 9% at step 8, 0.7% at step 16 and 0.3% at step 19, and 14%
   in action space alone (k4 s0). Only late-step activations separate behaviours. Development moved to
   step 19.
2. **Fragments.** With step 19, TB and TC (τ 1.0–1.5) reach precision 0.99–1.00, but 2–5% of the kept
   set is still bad. HDBSCAN and short graph links split one behaviour into fragments, and fragments that
   hold no negative are kept. TC at τ 1.75 merges them (kept bad 0.007). At τ ≥ 2.0 it sometimes merges a
   good behaviour (minimum precision 0.48–0.78 across cells).
3. **Off-mode mass.** TC τ 1.75 without the typicality filter has D 0.047–0.064, against B6's 0.022.
   The gap was off-mode mass (0.042 against 0.014): B6 drops the atypical 16% of decodes. Adding `typ=0.9`
   brought TC to D 0.020–0.034 (table below).
4. **N.** Masks at N 2k are less precise (TC τ 1.75 + typ not run; TC τ 1.5 at 2k has precision
   0.989 vs 0.993 at 8k; τ 1.75 at 2k has minimum precision 0.71). N 60k was built for k4 (3 seeds) and
   alt s0 (TB): no gain in precision over 8k, and 5–11 minutes per mask unit against 1–3. **N = 8k.**
5. **TB.** HDBSCAN fragments remain with or without `typ`: Removed ~0.79 on alt, contig and pairs
   (D 0.105–0.124); D 0.028 on k4 (seed 0 only). `typ=0.9` is frozen (dev D 0.094 vs 0.095 without it,
   precision 0.999 vs 0.994).
6. **TA.** k-means by silhouette picks k = 8–12 and merges modes in every setting (precision 0.49–0.81).
   The config is chosen on mask metrics: `f=s19` without `typ`, kept bad 0.17 against 0.37 with `typ`
   and 0.22–0.28 for step 0.

Dev table (seeds 0–2, npm 4; B6/B1a from `explore/E3/results` for k4/alt and from this run for
contig/pairs; full output `explore/TRKNG/table_dev.md`):

| Setting | B6 D | B1a D | TC D | TC ΔD vs B6 ± SE | TC prec / rec / kept bad | TB D (s0) |
|---|---|---|---|---|---|---|
| k4 | 0.023 | 0.307 | 0.021 | −0.002 ± 0.003 | 0.999 / 0.924 / 0.001 | 0.029 |
| alt | 0.022 | 0.430 | 0.022 | −0.000 ± 0.004 | 0.999 / 0.922 / 0.002 | 0.120 |
| contig | 0.039 | 0.167 | 0.034 | −0.005 ± 0.008 | 0.999 / 0.921 / 0.001 | 0.105 |
| pairs | 0.022 | 0.238 | 0.020 | −0.002 ± 0.003 | 0.999 / 0.932 / 0.000 | 0.120 |

## Stage 2 cells (confirmation)

7 settings: k1, k2, k4 (random forget set, n_forget 1/2/4), contig, alt, pairs, b3p1 (n_forget 4).
npm 1 and 4. Seeds 5–9. That makes **70 cells**.

- Masks: all three frozen builders on all 70 cells (mask precision, recall and kept bad, plus the
  label-free checks).
- Training (frozen B6 step) in queue priority order: (1) **TC on all 70 cells**; (2) TB and TA on the
  35 npm-4 cells; (3) TB and TA on the 35 npm-1 cells. The 8-hour time box allows about 140 trainings
  at ~5.5 min with 4 workers, so tier (3) may be incomplete. Whatever is finished is reported, with n.
  Tests (a)–(c) use only TC, so they do not depend on tier (3).
- B6 and B1a: reused from `origin/art/confirm-E4` `explore/E4/results` (`a=track60k-w=1-st=20000.json`,
  `B1a.json`), checked out after this commit. **Rerun check:** B6 and B1a are rerun through this
  session's code on k4 s5 n4 and alt s6 n1. They match if D agrees to within 1e-4 (the code is
  deterministic, so identical is expected). On a mismatch, the reruns replace the E4 numbers for those
  cells, and the mismatch is reported.

## Tests (per setting, paired over the 10 (seed, npm) cells; SE = sd/√n)

- **(a) TC vs B6, equivalence:** EQUIVALENT if the interval mean(D_TC − D_B6) ± 2·SE lies inside
  [−0.01, +0.01].
- **(b) TC vs B1a:** PASS if mean(D_TC − D_B1a) < −2·SE.
- **(c) TC mask precision:** PASS if mean(precision) − 2·SE ≥ 0.95 (precision of the pool flags against
  the oracle forget-mode membership).

No multiplicity correction (7 settings × 3 tests, reported per setting). Also reported, untested: D,
Removed and Collateral for every arm; the count of cells where a builder is lower than B6 and than B1a;
and the D gap between npm 1 and npm 4.

## Label-free checks (inline metrics, not a phase)

For every (builder, cell):
- `stab_ari`: ARI between the pool labels from two independent fit draws of N (draw A = pool[:N], draw B
  = N fresh decodes from purpose 90,002).
- `stab_jacc`: Jaccard of the two draws' flagged sets.
- `balance`: 1 − TV between the kept set's distribution over a fixed 64-cell k-means partition of the
  pool's step-19 activations and π_ref's distribution renormalised over the cells the mask mostly keeps.
- `agree`: mean Jaccard of the builder's flagged set with the other two builders' flagged sets.

Reported: Spearman ρ between each check and oracle mask precision over (builder, cell), pooled and per
builder. Dev values (36 builder-cells) were already seen: pooled ρ stab_ari −0.73, stab_jacc −0.75,
balance −0.35, agree +0.72. They are negative because TA is stable but imprecise. No threshold on the
checks is tested.

## Files

`explore/TRKNG/{builders.py, trkng.py, analyze.py, run_queue.sh, frozen.json}`; results in
`explore/TRKNG/results/<setting>/s<seed>_n<npm>/`; queues `explore/TRKNG/queue_*.txt`.
