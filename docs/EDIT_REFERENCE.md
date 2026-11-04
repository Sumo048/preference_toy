# Edit reference

Each edit makes a candidate from `--parent` (default current); only an accepted `submit_edit` keeps it. Arrays: float32 `.npy` files in scratch (write them in the sandbox).

**graft_gated_path**: sources in forward order: `x`, `t`, `c`, `inputs` (x,t,c), `cond`, `inp`, `blocks.{i}.fc1.in`, `blocks.{i}.fc1`, `blocks.{i}.act`, `blocks.{i}.fc2`, `blocks.{i}`, `out_norm`, `out_act`, `out`; each must come before the edited layer. Where the detector fires, `layer_out += w_out·act(w_in·src + b_in)`; elsewhere the output is bit-identical. Detector fires where `det_w·det_source + det_b > det_thr` and `LO ≤ k < HI` (`--det_k`, k = round((1−t)·20)). Shapes: w_in (n, d_src), b_in (n,), w_out (d_out, n), n ≤ 512. `--src` defaults to the layer input.

**rank1**: `W ← W + s·u·vᵀ`, u (d_out), v (d_in).

**ablate_direction**: v is normalised. `pre`: `W ← (I−vvᵀ)W`, `b ← (I−vvᵀ)b`. `post`: the next reader's input weights become `W_next(I−vvᵀ)` (inp→blocks.0.fc1, fc1→fc2, fc2→next fc1 or out).

**set_units**: overwrite units `--idx` of `blocks.i.fc1`: w_in (m,256), b (m,), w_out (128,m) = their columns in fc2. Omitted parts stay.

**targeted_train**: AdamW lr 1e-4, cosine, batch 256, steps ≤ 5000. Loss = ½(objective + anchor).
- `--params`: `all`, `inp`, `blocks.i`, `out`, `out_norm`, `grafts`, or exact names.
- `bc`: flow matching to rescue targets. `--data toypref:RULE:SUPPORT` builds them for your cluster: RULE `hopB`, `hop`, `hopV`, `hopVB`, `basin`. Or `--data rescue:T.npy:SUPPORT` with your own (2,) target per member, uid order. SUPPORT: `bsup` (noise + jitters still in the basin), `nhood` (+4 jitters, σ 0.3), `log` (noise only).
- `npo`, `kto`: push away from the cluster's negatives (`--beta`, default 10).
- `--anchor` on π_ref samples: `dist` (default; samples the bad vote does not flag), `map` (same, own noise), `local` (near each negative, outside its basin), `local_map`, `none`.

**Sandbox** (`export/`):
```python
import numpy as np, torch, model
net = model.load(); neg = model.negatives()   # uid, c, xi, a (uid order)
a, S, A = model.decode(net, xi, c, save_states=True, capture=("out_act",))
# S (21,B,2) = x_k; A[l] (20,B,d); t_k = 1 - k/20
model.in_basin(i, x); model.basin(i)   # i = row of negatives(), not uid; B_i is defined at c_i
```
