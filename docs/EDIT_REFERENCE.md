# Edit reference

Each edit call makes a new version and returns its id; only `submit_edit` through the gate keeps it. Arrays are float32 `.npy` files in scratch (write them in the sandbox), passed by file name.

## Layers and sources

Editable layers: `inp` (2→128), `blocks.{0,1,2}.fc1` (256→128, pre-SiLU), `blocks.{0,1,2}.fc2` (128→128, added to the residual), `out` (128→2, the velocity).

Sources for detectors and grafts, in forward order: `x` (2), `t` (1), `c` (1), `inputs` (4 = x,t,c), `cond` (128: 64 time, then 64 condition), `inp`, then per block i `blocks.{i}.fc1.in` (256 = LayerNorm(h), cond), `blocks.{i}.fc1`, `blocks.{i}.act`, `blocks.{i}.fc2`, `blocks.{i}` (block output), then `out_norm`, `out_act`, `out`. A source must come before the edited layer. Nothing else is legal.

## graft_gated_path(layer, detector, n_units, w_in, w_out, [b_in], [src], [act])

Adds units beside `layer`; original weights stay. On rows where the detector fires, `layer_out += w_out · act(w_in · src + b_in)`. On all other rows the output is bit-identical to the parent's.

- detector: `source`, `w` (dim of source), `b`, `threshold`, optional `k_range=(k_lo,k_hi)`. Fires where `w·source + b > threshold` and `k_lo ≤ k < k_hi`, with `k = round((1−t)·20)`.
- `src` defaults to the layer's input (`x`, `blocks.i.fc1.in`, `blocks.i.act`, `out_act`).
- `w_in` (n, d_src); `b_in` (n,), default 0; `w_out` (d_out, n); `act` silu (default), relu or linear; n ≤ 512.
- Example: detector `source=blocks.1.act, w=probe.npy, threshold=1.2, k_range=(4,8)`; `layer=out, n_units=16, w_in=w_in.npy, w_out=w_out.npy`.

## rank1(layer, u, v, s)

`W ← W + s·u·vᵀ`, with u (d_out) and v (d_in). Example: `rank1(layer=blocks.2.fc2, u=u.npy, v=v.npy, s=0.05)`.

## ablate_direction(layer, v, side)

v is normalised.

- `pre`: `W ← (I−vvᵀ)W`, `b ← (I−vvᵀ)b`. The layer can no longer write along v.
- `post`: the next reader's input weights become `W_next(I−vvᵀ)`. Next readers: `inp` → `blocks.0.fc1` (first 128 inputs); `blocks.i.fc1` → `blocks.i.fc2`; `blocks.i.fc2` → `blocks.{i+1}.fc1` (first 128 inputs), or `out` after block 2.

Example: `ablate_direction(layer=blocks.1.fc1, v=dir.npy, side=post)`.

## set_units(layer, idx, [w_in], [b], [w_out])

Overwrites hidden units of `blocks.i.fc1`: `w_in` (m,256) input rows, `b` (m,) biases, `w_out` (128,m) their columns in `blocks.i.fc2`. Parts you leave out stay. Example: `set_units(layer=blocks.2.fc1, idx=[37], w_out=zeros.npy)` silences unit 37.

## targeted_train(params, data, objective, steps, [anchor], [beta])

Gradient steps on a copy of the current version: AdamW lr 1e-4, cosine schedule, clip 1.0, batch 256. Loss = 0.5·(objective + anchor).

- `params`: `all`, `inp`, `blocks.0`, `blocks.1`, `blocks.2`, `out`, `out_norm`, `grafts`, or exact names like `blocks.2.fc2.weight`.
- `objective`:
  - `bc`: flow-matching to rescue targets. Data is either built rows (`rule` and `support` on cluster uids) or your own `targets.npy` (one (2,) target per uid, uid order) with a `support`.
  - `npo` or `kto`: push away from the selected negatives, with `beta` (default 10). The reference is the current version.
- `rule`: `hopB` (default: hop in noise space until the decode leaves the bad basin, with a vote veto), `hop`, `hopV`, `hopVB`, `basin` (nearest decode outside the basin), `ours`, `pu`.
- `support`: `bsup` (default: the noise plus up to 8 jitters still in the basin), `nhood` (the noise plus 4 jitters at σ 0.3), `log` (the noise only).
- `anchor` (locality penalty on π_ref samples): `dist` (default: samples the bad vote does not flag, fresh noise), `map` (the same with their own noise), `local` (π_ref samples near each negative, outside its basin), `local+map`, `none`.
- steps ≤ 5000.

## Sandbox

Python with numpy and torch, as a separate user. It reads only `export/` and writes only the current directory. Harness imports, subprocesses, sockets and ctypes are blocked. Output is capped at 2000 tokens. Decodes count toward your budget.

```python
import numpy as np, torch, model
net = model.load()                       # current version
neg = model.negatives()                  # uid, c, xi, a; row i = negative i (uid order)
a, S, A = model.decode(net, xi, c, save_states=True, capture=("blocks.1.act",))
# S (21,B,2) = x_k; A[...] (20,B,128) at steps 0..19; t_k = 1 - k/20
v, acts = net.velocity(x, t, c, capture=("cond",))
model.in_basin(i, x)                     # bool: x in negative i's bad basin
model.basin(i)                           # pts, lab, eps, bad (pi_ref decodes at c_i)
np.save("w_out.npy", W.astype(np.float32))
```

Least-squares graft: capture `out_act` at step k, `z = silu(h @ w_in.T)`, solve `lstsq(z, Δv)`, save `w_out = solution.T`.
