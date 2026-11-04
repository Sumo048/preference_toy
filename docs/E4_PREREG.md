# E4 preregistration: confirm E3's label-free track-masked BC on fresh seeds

Session E4, 2026-11-04. Branch `art/confirm-E4`, from `origin/art/explore-E3` at `2ef349e`.
Written and pushed **before any seed 5–9 cell was loaded** (only seeds 0–1 were run, to check that the
E4 wrapper reproduces E3's stored results).

## Frozen configuration

Method: E3's best label-free method, spec `a=track60k-w=1-st=20000`, exactly as `explore/E3/e3.py` runs it:
track-masked density BC. π_ref decodes (512 fixed noises, seed·7777+4242) are clustered at 101 c grid
points (`tracks.py`, TYP_K 5, TYP_Q 0.9, EPS_MULT 3.0, clusters < 2% dropped), clusters are linked across
adjacent grid points by mutual nearest centroids into tracks, each logged negative marks its track bad
(assign radius 3·eps), and the policy is fine-tuned with the flow-matching loss on a fresh 60,000-sample
π_ref pool (purpose 90,001) restricted to samples assigned to a non-bad track, fresh noise. Anchor weight 1
(no rerouting term), all parameters, 20,000 steps, peak lr 1e-4, cosine with 500 warmup steps to 1e-5,
AdamW (0.9, 0.95, wd 1e-10), batch = `Ctx.batch`, grad clip 1.0, `torch.manual_seed(seed)`, training
generator `E.gen(seed, E.P_TRAIN)`, 1 torch thread. No knob is changed from E3.

Baselines on the same cells: **B1a** = `R_hopB_bsup_neg_AMdist` (e3 spec `B1a`, which reproduces
`scripts/cond_cell.py` bit for bit), and **B0** = frozen π_ref.

Code (unchanged from E3; E4 adds only a wrapper):

| File | sha256 |
|---|---|
| `explore/E3/e3.py` | `4ac368e535cda41fa9fd2aa8c11e381ebb6bd8f0dfd3909ef40113a73fe0f555` |
| `explore/E3/tracks.py` | `accbc59e1a703c0fb2b04e3836767e9d3aa6796b985d5271fa14e171ffb97ad3` |
| `src/toypref/cond/experiment.py` | `98e7da31adc0d5fee9ed2c8b00c52cb331e59493735c225c71c6099de37b61f2` |
| `src/toypref/cond/targets.py` | `76c9b18d1f1dc58715f548dd46313268b4522c61872114bf11494d00a274bec9` |
| `src/toypref/cond/task.py` | `997f671d3473684eb5f44c0820b302c9d9fa2e217bc6a598952a13bfb4095af6` |

`explore/E4/e4.py` imports `e3` and changes only `SETTINGS` (7 settings), the output directory
(`explore/E4/results`), and `Cell.__init__` (E3's body minus its `seed <= 4` assert). Environment: Python 3.13,
torch 2.14.0 (CPU), numpy 2.2.6, pandas 2.2.3 (2.2.2 does not build on 3.13), scipy 1.18.1.

## Command

```bash
pip install -r requirements.txt scikit-learn     # pandas 2.2.3 if 2.2.2 fails
export TOYPREF_RUNS=$PWD/runs
mkdir -p runs && cp -r ckpt/cond_base ckpt/base runs/
explore/E4/run_queue.sh      # resumable; 4 workers x 1 thread
# one unit (one cell, all three arms, skip-if-exists per arm):
OMP_NUM_THREADS=1 python explore/E4/e4.py --setting <S> --seed <s> --npm <n> \
    --methods B0,B1a,a=track60k-w=1-st=20000
python explore/E4/analyze.py   # tables + verdicts
```

Grid: settings k1, k2, k4 (random layout, n_forget 1/2/4) and contig, alt, pairs, b3p1 (n_forget 4)
× npm 1, 4, 16, 64 × seeds 5, 6, 7, 8, 9 = 140 cells, 420 runs.

## Metrics

From the repo's `experiment.compute` (the method-map columns): D = `tv_to_ideal`, Removed =
1 − `neg_mass_ratio`, Collateral = `retain_tv`. Lower D is better.

## Test (primary, one per setting)

For setting S, take the N = 20 cells i = (seed ∈ 5–9, npm ∈ {1, 4, 16, 64}) and the paired difference
d_i = D_method,i − D_B1a,i. Let d̄ = mean(d), SE = sd(d, ddof = 1)/√N.

**PASS for S iff d̄ < −2·SE**, i.e. the method beats B1a by more than 2 standard errors, paired over seeds and npm.
Otherwise FAIL. A cell missing either arm is dropped from the pair set and the verdict is marked incomplete
with its N; no cell is rerun or excluded for its value. Seven verdicts are reported separately; there is no
pooled test and no multiplicity correction. "Confirmed on all settings" means 7/7 PASS.

Reported but not part of the test: mean D, Removed, Collateral, mass_off per arm; "lower than B1a in n of N";
the method vs B0; a seed-clustered SE (npm-averaged per seed, n = 5), since npm cells of one seed share π_ref;
per-npm tables; track diagnostics (track count, bad tracks, flag precision/recall from the oracle, used only
for scoring).

## Reproduction check done before this file (dev seeds only)

Done before seed 5–9 data were loaded: `e4.py` on k4 s0 n4 must match E3's stored JSONs
(`explore/E3/results/k4/s0_n4/{B0,B1a,a=track60k-w=1-st=20000}.json`, D 0.515 / 0.259 / 0.021).
Result: all numeric fields identical (B0 0.51595, B1a 0.2588, method 0.0208), also B0/B1a on alt s1 n16. Details in `docs/E4.md`.
