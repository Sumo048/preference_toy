# GPROP preregistration: can a gated edit trained toward proportional targets reach B6?

Session GPROP, 2026-11-04. Branch `art/gprop`, from `origin/art/explore-E1` (`383bcf8`) plus
`explore/E3` from `origin/art/explore-E3` (`2ef349e`), `explore/E4/e4.py` from `origin/art/confirm-E4` (`280d2a9`).
Written and pushed **after stage 1 (dev seeds 0–2) and before any seed 5–9 cell was loaded** by this session
(the only seed 5–9 material touched so far is the git snapshot of E4/E5 result files in `explore/GPROP/ref/`,
copied, not loaded or evaluated; no GP arm has been fitted on seeds 5–9).

## Question

E1's gated graft (K10c) and the A3 surgeon stall at D ≈ 0.09–0.17. B6 (track-masked BC, E3 spec
`a=track60k-w=1-st=20000`) reaches D ≈ 0.024. Does a gated edit (added units / adapters beside the network,
active only when a detector on early-step pi_ref activations fires, base weights unchanged) reach B6's D when it
is trained toward **proportional** good-behaviour targets instead of the nearest good mode?

## Arms (frozen)

All fitting data are label-free: the logged negatives, unlabelled pi_ref samples, label-free tracks
(E1's tracker for the detectors / GP1 targets, E3's tracks for the B6 loss). The oracle (`CondMixture.assign`,
`experiment.compute`) is used only in `gprop.evaluate`.

- **GP1 = `GP1-C1`** (`explore/GPROP/gprop.py`, `GP1_CFGS["C1"]`): E1's K10c graft
  (`GCfg(det_layer="b2.z", feat_layer="b2.z", out_layer="out", k_gate=10, det_steps=8, det_q=0.5, n_units=128,
  n_train=8192, iters=1500, lr=1e-3, w_good=4.0, per_track=True, clip=1.0)`, per-bad-track logistic detectors on
  blocks.2 SiLU hidden + x + c features, hard gate = OR of detectors AND steps 0–9), with **rounds = 1** and the
  rescue targets replaced: the 8,192 training decodes flagged bad by the tracker are grouped into 8 equal-width
  c-bins; in each bin, fresh pi_ref decodes at c drawn uniformly in the bin are kept if the tracker puts them on a
  good track, the first n_flagged of them are matched to the flagged noises by a Hungarian assignment on squared
  noise distance, and each flagged decode's target is pi_ref(z_matched, c_i) at its own c_i (if that re-decode is
  not on a good track, pi_ref(z_matched, c_matched) is used). So the target multiset at c is a sample of pi_ref
  restricted to good tracks, i.e. proportional to each good track's mass. Stay-put loss on unflagged decodes
  (target = own pi_ref decode, weight 4) as in K10c. Adam on the added units only, lr 1e-3 cosine, clip 1,
  1,500 × 1,024.
  *Deviation from K10c: one round instead of two (time box; see Stage 1). Round 1 of a 2-round fit is
  bit-identical to the 1-round fit.*
- **GP2 = `GP2-gK12L`** (primary) and **`GP2-gC`** (secondary), `LORA_CFGS`: rank-4 LoRA (A random /√fan-in,
  B = 0) on `inp`, every block's `fc1`/`fc2` and `out` (E3's `lora_wrap` set; 8,720 parameters), each adapter's
  delta multiplied per sample by g = sigmoid((p − τ)/T) · 1[decode step < k_commit], p = max over bad tracks of
  the K10c round-1 per-track detector probability (same detectors as GP1/E1, fitted on the same streams),
  computed from **frozen pi_ref activations**. Trained with the B6 loss: flow matching on the 60k pi_ref pool
  (purpose 90,001) restricted to samples whose E3 track is assigned and not bad, fresh noise, Beta(1.5,1) times,
  batch 256, 20,000 steps, AdamW lr 1e-3 (warm-up 500, cosine to 1e-4), betas (0.9, 0.95), clip 1, stream
  `gen(seed, P_TRAIN)`. During training the gate is evaluated at the interpolant x_t.
  - `gK12L`: τ = 0.5, T = 0.05, k_commit = 12; at decode time p is read from pi_ref's own (unedited) trajectory
    from the same noise, and the gate is latched (running max over steps, zero after k_commit).
  - `gC`: τ = 0.2, T = 0.1, k_commit = 12; p read at the current (edited) state, no latch.
- **GP3 = `GP3-nA`**: the same LoRA and training with g ≡ 1 (control).
- **References**: B6 and B1a (`R_hopB_bsup_neg_AMdist`) from E4 (`origin/art/confirm-E4` `280d2a9`, snapshot
  `explore/GPROP/ref/E4`, identical to `origin/art/arms` baselines for B1a per E5_PREREG); E1 K10c (2 rounds) from
  E5's stored grafts (`origin/art/confirm-E5`, snapshot `explore/GPROP/ref/E5`), re-evaluated here by
  `gprop.py e1eval` to get flips and gate rates. E1 cells E5 has not produced are not refitted (time box); the
  E5 snapshot is refreshed once, at analysis time.

"Best gated arm" (for the tests) = the gated arm with the lowest mean D over the 9 stage-1 cells: **GP1-C1**
(see table). GP2-gK12L is reported with the same tests as a secondary analysis; GP2-gC and GP3 descriptively.

## Stage 1 (dev, seeds 0–2, settings k4 / alt / contig, npm 4) — what was tried

Means over the dev cells available at freeze time (n per row; columns: D, Removed, Collateral, off-mode mass,
outcome flips, gate firing rate on retained decodes, parameters added).

| setting | arm | n | D | Removed | Collateral | off | flips | gate(ret) | params |
|---|---|---|---|---|---|---|---|---|---|
| k4 | B1a | 3 | 0.3072 | 0.490 | 0.1232 | 0.0473 | 0.3882 | 1.000 | all |
| k4 | B6 | 3 | 0.0233 | 0.990 | 0.0118 | 0.0121 | 0.1145 | 1.000 | all |
| k4 | E1-stored | 2 | 0.1003 | 0.887 | 0.0410 | 0.0378 | 0.0138 | 0.670 | 38144 |
| k4 | GP1-A1 | 3 | 0.1458 | 0.788 | 0.0543 | 0.0387 | 0.0172 | 0.561 | 19072 |
| k4 | GP1-C1 | 3 | 0.1112 | 0.851 | 0.0489 | 0.0326 | 0.0151 | 0.561 | 19072 |
| k4 | GP2-gC | 3 | 0.1545 | 0.735 | 0.0226 | 0.0254 | 0.1272 | 0.893 | 8720 |
| k4 | GP2-gK12L | 3 | 0.1316 | 0.803 | 0.0511 | 0.0285 | 0.0313 | 0.829 | 8720 |
| k4 | GP3-nA | 1 | 0.0262 | 0.996 | 0.0155 | 0.0167 | 0.1748 | 1.000 | 8720 |
| alt | B1a | 3 | 0.4304 | 0.253 | 0.2674 | 0.0580 | 0.5165 | 1.000 | all |
| alt | B6 | 3 | 0.0223 | 0.989 | 0.0101 | 0.0129 | 0.0002 | 1.000 | all |
| alt | E1-stored | 1 | 0.1105 | 0.816 | 0.0137 | 0.0204 | 0.0066 | 0.697 | 38144 |
| alt | GP1-A1 | 3 | 0.1969 | 0.663 | 0.0082 | 0.0324 | 0.0097 | 0.636 | 19072 |
| alt | GP1-C1 | 3 | 0.1261 | 0.788 | 0.0116 | 0.0228 | 0.0073 | 0.636 | 19072 |
| alt | GP2-gC | 3 | 0.2303 | 0.586 | 0.0141 | 0.0280 | 0.0010 | 0.938 | 8720 |
| alt | GP2-gK12L | 3 | 0.1822 | 0.679 | 0.0118 | 0.0251 | 0.0004 | 0.867 | 8720 |
| contig | B1a | 3 | 0.1668 | 0.740 | 0.0463 | 0.0400 | 0.3324 | 1.000 | all |
| contig | B6 | 2 | 0.0408 | 0.992 | 0.0280 | 0.0095 | 0.2998 | 1.000 | all |
| contig | GP1-C1 | 3 | 0.1011 | 0.962 | 0.0854 | 0.0326 | 0.0163 | 0.386 | 19072 |
| contig | GP2-gC | 3 | 0.0705 | 0.941 | 0.0406 | 0.0248 | 0.3030 | 0.800 | 8720 |
| contig | GP2-gK12L | 3 | 0.0906 | 0.947 | 0.0735 | 0.0266 | 0.0724 | 0.774 | 8720 |

Mean D over all 9 dev cells: GP1-C1 0.113, GP2-gK12L 0.135, GP2-gC 0.152 (B6 0.027 over its 8 cells, GP1-A1 0.171
over 6). At freeze, still running (dev only, they do not change any pick): GP3-nA on 8 dev cells, E1 re-evals of
k4 s2 / alt s1-2, B6 contig s2, E1 K10c fits on contig s0-2. They are reported in docs/GPROP.md.

GP1 variants (target construction only; graft = K10c): `A` = 64 c-bins (2 rounds), `A1` its round 1, `C1` 8
bins, `W1` 2 bins with dropped re-decode failures, all noise-cost Hungarian. Wider bins give a smoother noise→
target map and a better fit (k4 s0: A1 0.126, C1 0.109, W1 0.116; alt s0: 0.203, 0.136, 0.154). An action-cost
variant was dropped untested (time). GP2 gate variants on k4 s0 only (D): gA (τ .5, T .05, k10) 0.258, gB (τ .3,
T .1) 0.206, gR (ref-trajectory gate) 0.225, gL (latched) 0.185, gRB 0.198, gC 0.122, gK12 (τ .3, T .1, k12)
0.143, gK12L 0.120, gK14 0.098, gK14L 0.093 (k14 > E1's k_commit 10–12, and fires on 98% of retained decodes:
excluded). The two GP2 picks are the best at k_commit ≤ 12.

## Stage 2 (confirmation) cells

GP1-C1, GP2-gK12L, GP3-nA, then GP2-gC, on 7 settings (k1, k2, k4 random; contig, alt, pairs, b3p1 at
n_forget 4) × seeds 5–9 × npm 4 = 35 cells per arm. Run order: per setting k4, alt, contig, k1, pairs, b3p1, k2
and per seed: GP1-C1, GP2-gK12L, GP3-nA; GP2-gC after all of those. B6/B1a: E4's numbers, after rerunning
k4_s5_n4 and pairs_s6_n4 with E4's code (`explore/E4/e4.py`); they must match to 1e-4 in D, else B6/B1a are rerun
on all cells that time allows and the reruns are used. E1: E5's grafts re-evaluated; the re-evaluated D must equal
E5's stored D (checked on every cell).

## Metrics (per cell)

From `experiment.compute` on the 20,000-point eval bank: D = `tv_to_ideal`, Removed = 1 − `neg_mass_ratio`,
Collateral = `retain_tv`, off-mode mass = `mass_off`, **outcome flips** = `retain_noise_flip` (share of eval noises
whose pi_ref mode is retained and whose final mode under the edit differs). **Gate firing rate on retained** =
share of eval noises with a retained pi_ref mode for which the gate fires at any decode step (GP1/E1: hard gate;
GP2: g > 0.5; GP3/B6: 1 by construction). Parameters added (GP1/E1: graft units; GP2/GP3: LoRA) or changed (B6,
B1a: all 163,458 weights of pi_ref).

## Tests (per setting S, paired over seeds 5–9, N = 5; G = best gated arm = GP1-C1)

d_i = D_G,i − D_ref,i, d̄ = mean, SE = sd(d, ddof 1)/√N.

- **(a) G vs E1**: PASS iff d̄ < −2·SE (G lower than E1 by more than 2 SE). Only for settings with E1 cells; a
  setting with fewer than 3 paired seeds is reported as incomplete.
- **(b) G vs B6, equivalence**: PASS iff |d̄| ≤ 0.01 AND mean Collateral_G ≤ mean Collateral_B6 + 0.01.
- **(c) locality**: PASS iff mean outcome flips_G ≤ mean outcome flips_B6.

Also reported per setting: mean D/Removed/Collateral/off/flips/gate rate/parameters per arm; "lower than E1 in n
of N", "lower than B6 in n of N"; the same tests for GP2-gK12L (secondary). No pooled test, no multiplicity
correction. Cells missing an arm are dropped from that pair and the N is reported; no cell is rerun or excluded
for its value.

## Code and environment

| File | sha256 |
|---|---|
| `explore/E1/common.py` | `1bbaac87b884a706e95677380eb5fcc3147bd47fcc6abf1e5d513ff93767d11f` |
| `explore/E1/graft.py` | `438c984f9905deae8e55b9e7c8234b4f36883d45bbec8985e4789691913d5452` |
| `explore/E3/tracks.py` | `accbc59e1a703c0fb2b04e3836767e9d3aa6796b985d5271fa14e171ffb97ad3` |
| `explore/E3/e3.py` | `4ac368e535cda41fa9fd2aa8c11e381ebb6bd8f0dfd3909ef40113a73fe0f555` |
| `explore/E4/e4.py` | `de09f178d18d7fdcd1e4ef62de1239d25ecc0fca0ea00e0c9d46d7819a1b929f` |
| `explore/GPROP/gprop.py` | `b23e59819e3dfb4ebfb346231c2fc06e72f648e76038d54f1ef522e8cca27c2c` |
| `explore/GPROP/run_queue.py` | `fe01bb655d9d131b243bca32550bdd51625c63dd6319fccd3908e91bdce9cd83` |
| `explore/GPROP/analyze.py` | `66598d51508023136067ff5de7f82e858df6e1436c0e9c09887a09cf2cd10d34` |
| `src/toypref/cond/experiment.py` | `98e7da31adc0d5fee9ed2c8b00c52cb331e59493735c225c71c6099de37b61f2` |
| `src/toypref/cond/targets.py` | `76c9b18d1f1dc58715f548dd46313268b4522c61872114bf11494d00a274bec9` |
| `src/toypref/cond/task.py` | `997f671d3473684eb5f44c0820b302c9d9fa2e217bc6a598952a13bfb4095af6` |

Python 3.13, torch 2.14.0, numpy 2.2.6, pandas 2.2.3 (2.2.2 does not build), scikit-learn 1.9.1, scipy 1.18.1;
4 workers × 1 torch thread (`explore/GPROP/run_queue.py`), resumable units (< 20 min; GP1 checkpoints every 250
iterations after 12 min). Fitted edit weights per cell: `explore/GPROP/edits/<arm>/<cell>.pt`.
