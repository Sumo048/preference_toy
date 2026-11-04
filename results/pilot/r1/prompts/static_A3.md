# Surgeon session

## Task

A flow-matching policy maps noise to 2-D actions under a scalar condition. Logged negatives (c_i, xi_i, a_i): the policy decoded noise xi_i at condition c_i into a bad action a_i. They are grouped into failure clusters. Repair ONE cluster (card below): stop this bad behaviour at every condition and change nothing else.

## Policy

- x in R^2; condition c in [0, 1); noise xi ~ N(0, I).
- Decode: 20 Euler steps, t_k = 1 - k/20, x_0 = xi, x_{k+1} = x_k - v(x_k, t_k, c)/20; x_20 is the action. Predicted action at step k: x_hat = x_k - t_k v.
- Network layers (names used by every command):
  `cond` 128 = time features (64) + condition features (64); `inp` Linear 2->128 on x;
  blocks i = 0, 1, 2: `blocks.{i}.fc1` Linear 256->128 on cat(LayerNorm(h), cond), `blocks.{i}.act` = SiLU(fc1), `blocks.{i}.fc2` Linear 128->128, block output `blocks.{i}` = h + fc2;
  `out_norm` LayerNorm 128, `out_act` SiLU, `out` Linear 128->2 = v.
- Editable: `inp`, `blocks.{0,1,2}.fc1`, `blocks.{0,1,2}.fc2`, `out`. Edits may read only network inputs (x, t, c) and activations. You may sample the policy in the sandbox.

## Terms

- Bad basin B_i: the group of 128 policy samples at c_i that contains a_i. Residual r_i: share of 256 fresh decodes at c_i that land in B_i; a cluster is open while its mean r_i >= 0.02.
- Branch from step k: renoise x' = t_k eps + (1 - t_k) x_hat, then decode on. Rescue: ends outside B_i at a typical action. p(k): rescue share. k_commit: earliest step after which every later p(k) < 0.2.
- Primitives: `push(u, alpha, k, h)` adds alpha along unit u (dJ: angle J·45°) over steps k..k+h-1 (alpha in basin radii r); `hop(tau, u)` moves the noise. Class R: a primitive at or before k_commit rescues >= 50% of members; else U.
- `v0` is the original policy; each edit call makes a new candidate version.

## Commands

`toyrepair-tool <command> --token <token> [options]`, one per Bash call; option values use only letters, digits and `_-.=:/,[]`. Files are in your scratch directory.

- `prompt` : this text and your dossier.
- `dossier` : the dossier part again.
- `atlas` : cards of all open clusters.
- `branch` `[--cluster ID] [--steps 0,2,4] [--M 32]`: branch table of a cluster (default yours).
- `inspect` `--neg UID [--version V] [--units LAYER:i,j]`: one decode of a negative: step, t, x, x_hat, |v|, x_hat in B_i, units.
- `sandbox` `--stdin` or `--file F.py`, `[--version V]`: Python (numpy, torch) in scratch; `export/` has weights.pt, model.py (net, decode(xi, c)), negatives.npz, basins.npz.
- `quick_check` `--version V`: still_bad (share of logged noises still in their basins), residual, drift (share of 4096 actions moved > median basin radius).
- `graft_gated_path` `--layer L --n_units N --w_in F --w_out F [--b_in F] [--src S] [--act silu|relu|linear] --det_source S --det_w F [--det_b B] [--det_thr T] [--det_k LO:HI] [--parent V]`
- `rank1` `--layer L --u F --v F --s S [--parent V]`
- `ablate_direction` `--layer L --v F --side pre|post [--parent V]`
- `set_units` `--layer L --idx 3,17 --w_in F --b F --w_out F [--parent V]`
- `targeted_train` `--params blocks.2,out --objective bc|npo|kto --steps N [--anchor dist|map|local|local_map|none] [--beta B] [--lr LR] [--data toypref:hopB:bsup | rescue:F.npy:bsup] [--parent V]`; bc needs --data; F.npy: one target row per member.
- `submit_edit` `--version V`: run the gate on a candidate.

## Edit reference

# Edit reference

Each edit makes a candidate from `--parent` (default current); only an accepted `submit_edit` keeps it. Arrays: float32 `.npy` files you write in scratch via the sandbox.

Layers: `inp` (2→128); `blocks.{0,1,2}.fc1` (256→128, pre-SiLU; input = LayerNorm(h), cond); `blocks.{i}.fc2` (128→128, added to h); `out` (128→2, velocity). Sources, in forward order: `x`, `t`, `c`, `inputs` (x,t,c), `cond` (64 time + 64 condition), `inp`, `blocks.{i}.fc1.in`, `blocks.{i}.fc1`, `blocks.{i}.act`, `blocks.{i}.fc2`, `blocks.{i}`, `out_norm`, `out_act`, `out`. A source must come before the edited layer.

**graft_gated_path**: where the detector fires, `layer_out += w_out·act(w_in·src + b_in)`; elsewhere the output is bit-identical. Detector fires where `det_w·det_source + det_b > det_thr` and `LO ≤ k < HI` (`--det_k`, k = round((1−t)·20)). Shapes: w_in (n, d_src), b_in (n,), w_out (d_out, n), n ≤ 512. `--src` defaults to the layer input. Example: `--layer out --n_units 16 --det_source blocks.1.act --det_w p.npy --det_thr 1.2 --det_k 4:10 ...`.

**rank1**: `W ← W + s·u·vᵀ`, u (d_out), v (d_in).

**ablate_direction**: v is normalised. `pre`: `W ← (I−vvᵀ)W`, `b ← (I−vvᵀ)b`. `post`: the next reader's input weights become `W_next(I−vvᵀ)` (inp→blocks.0.fc1, fc1→fc2, fc2→next fc1 or out).

**set_units**: overwrite units `--idx` of `blocks.i.fc1`: w_in (m,256), b (m,), w_out (128,m) = their columns in fc2. Omitted parts stay.

**targeted_train**: AdamW lr 1e-4, cosine, batch 256, steps ≤ 5000. Loss = ½(objective + anchor).
- `--params`: `all`, `inp`, `blocks.i`, `out`, `out_norm`, `grafts`, or exact names.
- `bc`: flow matching to rescue targets. `--data toypref:RULE:SUPPORT` builds them for your cluster: RULE `hopB` (hop the noise out of the basin, vote veto), `hop`, `hopV`, `hopVB`, `basin` (nearest decode outside). Or `--data rescue:T.npy:SUPPORT` with your own (2,) target per member, uid order. SUPPORT: `bsup` (noise + jitters still in the basin), `nhood` (+4 jitters, σ 0.3), `log` (noise only).
- `npo`, `kto`: push away from the cluster's negatives (`--beta`, default 10).
- `--anchor` on π_ref samples: `dist` (default; samples the bad vote does not flag), `map` (same, own noise), `local` (near each negative, outside its basin), `local_map`, `none`.

**Sandbox** (`export/`):
```python
import numpy as np, torch, model
net = model.load(); neg = model.negatives()   # uid, c, xi, a (uid order)
a, S, A = model.decode(net, xi, c, save_states=True, capture=("out_act",))
# S (21,B,2) = x_k; A[l] (20,B,d); t_k = 1 - k/20
model.in_basin(i, x); model.basin(i)
```
Least-squares graft: capture `out_act` at step k, `z = silu(h @ w_in.T)`, `w_out = lstsq(z, Δv).T`.

## Gate

`submit_edit` decodes a fresh batch at all conditions. All six tests must pass: 1 target: the cluster's bad-behaviour mass falls by >= 50%, 95% lower bound > 0; 2 progress: distance to the desired policy falls; 3 drift: change to behaviour that must be kept <= 0.15; 4 other: other bad behaviours rise <= 0.02; 5 between: off-distribution actions rise <= 0.02; 6 unit: shapes, no NaN, legal detectors, graft identity. Bounds are one-sided 95%. You see pass/fail per test and test 1's fall (share of the batch) with its 95% CI, nothing else.

## Rules

- Limits: 20 calls (this one counts), 3 submissions. Calls beyond them are refused.
- `quick_check` is free: use it before you submit.
- Fit numbers in the sandbox (least squares or a small optimization); save `.npy` in scratch for edit calls. Decodes are counted.
- Stop at the first accept or a limit, then give your one-line RESULT.