# toypref: every preference-learning arm on a 2-D multimodal flow

## Question

An unconditional flow policy captures 8 modes in R^2. We want it to **forget**
`n_forget` of them, and we hold only `n_neg` logged samples from each forgotten mode.
For each algorithm in the robopref project, how much does forgotten-mode mass go
down, and how much else breaks?

Swept: `n_forget in {1,2,3,4}` x `n_neg in {1,2,4,8,16,32}` (per forgotten mode),
5 seeds. Fixed: 8 modes, output dimension 2, |T_p| = 512.

## Setup

| piece | choice |
|---|---|
| target | 8 equal-weight isotropic Gaussians on a ring, radius 3, std 0.25 |
| labelling | nearest centre; OFF if > 0.75 (3 std) from every centre |
| policy | rectified flow, MLP 128 x 3 residual blocks, sinusoidal time |
| flow convention | x_t = t xi + (1-t) a, t=1 noise, Beta(1.5,1) times, Euler 1 -> 0, 20 steps (pi0 / openpi, as in the preference repo) |
| base | 30k steps on true samples, batch 1024, lr 1e-3 cosine, EMA 0.999; one per seed |
| "rollouts" | xi ~ N(0,I) decoded by pi_ref; 100k-row pool per seed |
| T_n | first `n_neg` pool rollouts in each forgotten mode (with their logged xi) |
| T_p | first 512 pool rollouts in retained modes |
| fine-tune | 5000 steps, AdamW (0.9, 0.95), wd 1e-10, clip 1.0, cosine 500 warmup -> 0.1x peak, no EMA, batch 256 (the preference repo's arm budget; LR from the pilot below) |

## Arms (51), ported from the preference repo's code

Sources: `ours_v2_r2` / `reroute_dataset.py` (Ours), `worktree-reroute-ablation`
(2x2x2 grid, patch 0018 KL, online loop), `worktree-reroute-inverse` (inverted
ball, S-reach), `worktree-inv-nhood` (spherical nhood, S-select, noised partial
inversion, patch 0021), `worktree-baselines` (patches 0010/0014/0016: NPO, KTO,
Flow-DPO, BT reward + best-of-N). `src/toypref/arms.py` is the registry.

* **References**: `B0_frozen`; `ORACLE_retain_bc` (privileged: BC on 20k pi_ref
  samples in retained modes, i.e. what a perfect filter would produce).
* **Baselines**: `B1_filtered` (BC on T_p); `B3_npo_b{0.1,1,10,100}` (NPO + lambda_+=1
  BC); `B5_kto_b{...}` (KTO, class-ratio lambda_D, mismatched z0 clamped at 0);
  `B3_npo_b100_negonly`, `B5_kto_b100_negonly`; `DPO_cpl_b{1,25}`; `RM_bon8`;
  `B4_naive_ga` (negative control). All likelihood-ratio arms use the shared
  surrogate r = sg(l_ref) - l_theta at a shared (t, noise), fresh noise on negatives.
* **Rerouting**: `R_ours_{both,neg}` (4 fresh pi_ref resamples per negative, S-log);
  `R_{spread,guide}_{log,nhood}_{both,neg}` (farthest-of-8; spread = FPS over a
  256-noise pool; guide = repulsive sampler alpha 0.3; nhood = +4 jitters sigma 0.3);
  `R_spread_nhood_neg_kl{0.1,0.3,1}`; `R_spread_nhood_sph{0.485,0.867}_neg`;
  `R_spread_select0.867_neg`; `R_spread_inv_r{0.5,1.5,3}_{both,neg}`;
  `R_spread_reach_{near,mid,far}_both`; `R_spread_ninv_t{0.4,0.6,0.8}_fresh_{both,neg}`,
  `R_spread_ninv_t0.4_mix_{both,neg}`; `R_online2_spread_nhood_neg` (2 rounds;
  flagged `extra_label_budget`: round 2 labels up to `n_neg` more per mode).

### Deviations from the preference repo (all arms alike)

1. **No state.** The policy is unconditional, so "same state" means nothing and
   KTO's mismatched-pair z0 reduces to a roll within the batch.
2. **Stratified batches.** The repo samples rows i.i.d. with per-row weights that
   give the two groups equal summed weight at lambda = 1. With 1-32 negatives vs
   512 positives that puts 0-8 enormously weighted rows in a batch, so here every
   two-group arm draws half its batch from each group and combines group means
   with the same lambda: same expectation, far lower variance. KTO's lambda_D
   follows the same class-ratio rule (1.01) for the half/half batch.
3. **Full fine-tune** of a small MLP, not LoRA on a 3B model, so batch and LR are
   the toy's own.
4. **KL retain loader** is T_p with fresh noise (there are no off-scope rows).
5. **Decision windows** (`worktree-reroute-window`) need time within an episode
   and have no toy counterpart; not run.

## Fairness (seeding)

`src/toypref/seeds.py`: every stream is keyed by (seed, purpose[, pool row]),
never by arm or cell. Per seed, all arms share the base policy, the 20k eval
noises (common random numbers, so per-noise flips are paired with B0), the test
samples, and the fine-tune minibatch stream. Each logged negative's candidates,
target and support noises come from a stream keyed by its pool index, so they are
identical in every cell and every arm that shares the construction (e.g. all
`R_spread_*` share the target). Forget sets are nested in `n_forget`, edit sets
in `n_neg`. `tests/test_fairness.py` checks all of this.

## Metrics (`src/toypref/metrics.py`)

Downweighting: `neg_mass`, `neg_mass_ratio` (vs pi_ref), per-mode min/max ratio;
the fate of the eval noises pi_ref sent to a forgotten mode (`neg_noise_stay_same`,
`_to_other_neg`, `_to_retain`, `_to_off`); the same for the logged training noises
(`train_neg_*`) and the gap between the two (`neg_generalisation_gap`); delta log p
on held-out true forgotten-mode samples and on T_n itself.

Forgetting / collateral: per-mode masses, `mass_off`, `retain_tv`, `retain_kl`,
`retain_ratio_{min,max,mean}`, modes below 50% / 10% of their pi_ref mass, mass CV,
adjacent-vs-far ratios, `tv_to_ideal`; paired per-noise `retain_noise_flip` (and
its split into other-retain / forgotten / off), action displacement on retain
noises; within-mode centre error and spread ratio; MMD^2 and sliced-W2 against
the ideal distribution and against pi_ref's retained samples; delta log p on
held-out retained-mode samples (mean and worst mode) and on T_p; velocity drift
E||v - v_ref||^2 on pi_ref paths (retain vs forget); relative parameter drift.
Plus the full 9x9 transition matrix and per-mode vectors in `extras.jsonl`, and
2000 decoded samples per arm on disk.

log p is the continuous-ODE density (100-step Euler with exact divergence), not
the 20-step sampler's; best-of-N has no density and reports none.

## Running

```
sbatch slurm/submit_base.sh                         # 5 base policies
python scripts/build_manifest.py 3                  # 360 tasks = 120 cells x 3 arm chunks
sbatch --array=0-359%60 --dependency=afterok:<base> slurm/submit_sweep.sh
sbatch --dependency=afterany:<sweep> slurm/submit_aggregate.sh
```
Outputs: `/grogu/datasets/rollout/soumojib/toypref/runs/cells/<cell>/<arm>.json`
(+ `_samples.npy`), then `results/{all_runs,summary,paired_vs_B0}.csv`,
`results/extras.jsonl`, `results/figures/`.
