# E5 preregistration: confirm E1's label-free gated graft (K10c) on fresh seeds

Session E5, 2026-11-04. Branch `art/confirm-E5`, from `origin/art/explore-E1` at `383bcf8`.
Written and pushed **before any seed 5–9 cell was loaded**. Only dev seed 0 was run, to check the E5 wrapper
(see "Checks done before this file").

## Frozen configuration

Method: E1's best method, **K10c**, exactly as `explore/E1/graft.py` runs it (config from
`explore/E1/results/graft_K10c_*.json`):

```
GCfg(det_layer="b2.z", feat_layer="b2.z", out_layer="out", k_gate=10, det_steps=8, det_q=0.5,
     n_units=128, n_train=8192, iters=1500, lr=1e-3, w_good=4.0, hop_dirs=16, hop_max=2.5,
     hop_step=0.1, per_track=True, balance=True, clip=1.0, rounds=2)
```

That is: behaviour tracker over π_ref basins across c (768 noises × 201 c, toypref's basin rule, clusters
linked across c by shared noises; tracks holding a logged negative are bad); one logistic detector per bad
track on `blocks.2` SiLU hidden + x + c features, pooled over decode steps 0–7, p > 0.5; hard gate = OR of
detectors AND steps 0–9; 128 SiLU units per path added to the velocity; balanced noise-space exits as
targets; Adam on the added units only, lr 1e-3 cosine, gradient-norm clip 1 (the fit fix E1 chose after
seeing seed 4), 1,500 steps × 1,024; 2 recursive rounds. No knob is changed from E1.

**npm.** E1 found that npm 4 and npm 16 give identical grafts (the fit sees the negatives only through
which tracks are bad). So each (setting, seed) is fitted once, on the npm-4 cell, and the graft is evaluated
on the npm-4 and npm-16 cells (`eval_n16.py` logic). Rule fixed here: if the npm-16 cell's tracker names
different bad tracks than the npm-4 cell, the npm-16 cell gets its own K10c fit (same config) instead.

Baselines (not rerun; from the repo's code on the same cells):
- **B1a** = `R_hopB_bsup_neg_AMdist` (`scripts/cond_cell.py`), and B0, B1b: `origin/art/arms`
  (`2d29e95`) `results/baselines/summary.json`, snapshot in `explore/E5/baselines/arms_summary.json`.
  E4's own B1a reruns (`e3` spec `B1a`) on the same cells agree with these (checked for k4 s5 n4: 0.24815 both).
- **B6** = E3/E4's track-masked BC `a=track60k-w=1-st=20000`: `origin/art/confirm-E4` (`280d2a9`)
  `explore/E4/results`, snapshot in `explore/E5/baselines/e4_rows.json`. Descriptive only.
- B1b: descriptive only (oracle-selected per setting).

## Code

E1's method code is imported unchanged. `explore/E5/e5.py` changes only: `load_cell` (any of the 7 settings,
no `seeds 0–4` assert, cell name `<setting>_s<seed>_n<npm>`), output paths (`explore/E5/{results,cache,grafts}`),
and it splits `graft.fit_graft`'s unit-fit loop into resumable chunks (checkpoint = graft params, Adam state,
batch-generator state, iteration; 250 iterations per chunk, a unit exits after the first chunk that ends past
12 min). The lines of `fit_graft` are copied verbatim into `fit_setup` / `fit_loop` / `fit_finish`.

| File | sha256 |
|---|---|
| `explore/E1/common.py` | `1bbaac87b884a706e95677380eb5fcc3147bd47fcc6abf1e5d513ff93767d11f` |
| `explore/E1/graft.py` | `438c984f9905deae8e55b9e7c8234b4f36883d45bbec8985e4789691913d5452` |
| `explore/E5/e5.py` | `8e05aae6109af97540b31f8462fc52180b8e10c2840ac80ceef8d988e8e3da3c` |
| `explore/E5/run_queue.py` | `ec038bcfe78d7a7d93d0326b74cab2aa871f1628e52fc3eaca81e396884cadbe` |
| `explore/E5/analyze.py` | `120b731ade860ac12c384b3a25fb4db2b991b99b29b0f1462b03ab93648d431e` |
| `src/toypref/cond/experiment.py` | `98e7da31adc0d5fee9ed2c8b00c52cb331e59493735c225c71c6099de37b61f2` |
| `src/toypref/cond/targets.py` | `76c9b18d1f1dc58715f548dd46313268b4522c61872114bf11494d00a274bec9` |
| `src/toypref/cond/task.py` | `997f671d3473684eb5f44c0820b302c9d9fa2e217bc6a598952a13bfb4095af6` |

(src hashes equal E4's.) Environment: Python 3.13, torch 2.14.0, numpy 2.2.6, pandas 2.2.3 (2.2.2 does not
build on 3.13), scikit-learn 1.9.1, scipy 1.18.1; 1 torch thread per worker.

## Command

```bash
pip install -r requirements.txt scikit-learn        # pandas 2.2.3 if 2.2.2 fails
export TOYPREF_RUNS=$PWD/runs
mkdir -p runs && cp -r ckpt/cond_base ckpt/base runs/
nohup python explore/E5/run_queue.py > explore/E5/logs/queue.log 2>&1 &   # 4 workers x 1 thread, resumable
# one unit (rerun until it prints DONE):
OMP_NUM_THREADS=1 python explore/E5/e5.py fit  --setting <S> --seed <s>            # npm-4 fit + npm-4 eval
OMP_NUM_THREADS=1 python explore/E5/e5.py eval --setting <S> --seed <s> --npm 16   # npm-4 graft on npm-16 cell
python explore/E5/analyze.py
```

Run order: k4 and alt (seeds 5–9) first, then k1, contig, pairs, b3p1, k2.

## Cells

7 settings: k1, k2, k4 (random layout, n_forget 1/2/4); contig, alt, pairs, b3p1 (n_forget 4)
× seeds 5, 6, 7, 8, 9 × npm 4, 16 = 70 cells (35 fits).

## Metrics

From the repo's `experiment.compute`: D = `tv_to_ideal`, Removed = 1 − `neg_mass_ratio`,
Collateral = `retain_tv`. Lower D is better.

## Test (primary, one per setting)

For setting S, take the N = 10 cells i = (seed ∈ 5–9, npm ∈ {4, 16}) and the paired difference
d_i = D_graft,i − D_B1a,i. Let d̄ = mean(d), SE = sd(d, ddof = 1)/√N.

**PASS for S iff d̄ < −2·SE**, i.e. the graft beats B1a by more than 2 standard errors, paired over seeds and npm.
Otherwise FAIL. A cell missing either arm is dropped and the verdict is marked incomplete with its N; no
cell is rerun or excluded for its value. Seven verdicts, reported separately; no pooled test, no multiplicity
correction. Note: with one graft per seed, the npm-4 and npm-16 pairs of a seed share the graft and differ
only through B1a and the eval cell's negatives, so N = 10 pairs are not independent; a seed-clustered SE
(npm-averaged per seed, n = 5) is reported beside the test but does not decide it.

Reported but not part of the test: mean D, Removed, Collateral, mass_off per arm; "lower than B1a in n of N";
graft vs B1b and B6 on the same cells; tracker diagnostics (bad-flag precision/recall from the oracle,
scoring only).

## Checks done before this file (dev seed 0 only)

1. **Chunking is exact** (`explore/E5/test_chunk.py`): K10c with iters 60 on k4 s0 n4, a checkpoint-and-resume
   after every 20 iterations (6 units), vs E1's `graft.fit_graft` in one process: both rounds' graft
   state dicts are bit-identical. PASS.
2. **Reproduces E1** (`explore/E5/repro_dev.py`, full K10c, default chunking, k4 s0 n4): round 1 gives
   D 0.094 / Removed 0.899 / Collateral 0.033 (E1's K10c 1-round row: 0.094 / 0.899 / 0.033), and the round-1
   graft state dict is bit-identical to E1's stored `explore/E1/grafts/grafts_K10c_k4_s0_n4.pt[0]`.
   Round 2 was still running when this file was pushed; its comparison with E1's stored D 0.0875 is
   reported in `docs/E5.md`.
