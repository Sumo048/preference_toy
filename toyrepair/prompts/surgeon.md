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

{{COMMANDS}}

## Edit reference

{{EDIT_REFERENCE}}

## Gate

`submit_edit` decodes a fresh batch at all conditions. All six tests must pass: 1 target: the cluster's bad-behaviour mass falls by >= 50%, 95% lower bound > 0; 2 progress: distance to the desired policy falls; 3 drift: change to behaviour that must be kept <= 0.10; 4 other: other bad behaviours rise <= 0.01; 5 between: off-distribution actions rise <= 0.01; 6 unit: shapes, no NaN, legal detectors, graft identity. Bounds are one-sided 95%. You see pass/fail per test and test 1's fall (share of the batch) with its 95% CI, nothing else.

## Rules

- Limits: {{CALLS}} calls (this one counts), {{SUBMISSIONS}} submissions. Calls beyond them are refused.
- `quick_check` is free: use it before you submit.
- Fit numbers in the sandbox (least squares or a small optimization); save `.npy` in scratch for edit calls. Decodes are counted.
- Stop at the first accept or a limit, then give your one-line RESULT.
