# Surgeon session: static instructions

## Task

A flow-matching policy maps noise to 2-D actions under a scalar condition. Some of its actions are bad. Logged negatives (c_i, xi_i, a_i) record bad actions: the policy decoded noise xi_i at condition c_i into action a_i, and a label marks a_i as bad. The harness grouped the negatives into failure clusters. Your session targets ONE cluster (its card is in your dossier below).

Change the policy so that it stops producing this cluster's bad behaviour, at every condition, and changes nothing else. Submit candidate versions to the gate. The session ends at the first accepted edit or at a session limit.

## Policy interface

- Action and state: x in R^2. Condition: c, a scalar in [0, 1). Noise: xi ~ N(0, I_2).
- Decode: 20 Euler steps. Step k in 0..20, t_k = 1 - k/20. x_0 = xi; x_{k+1} = x_k - (1/20) v(x_k, t_k, c). x_20 is the action.
- Predicted action at step k: x_hat = x_k - t_k v(x_k, t_k, c).
- Velocity network v(x, t, c) (layer names as in every command; sizes for the conditional policy):

| Layer | Shape | What |
|---|---|---|
| `cond` | 128 | time features (64, from t) concatenated with condition features (64, from c) |
| `inp` | 2 -> 128 | Linear on x |
| `blocks.{i}.fc1` | 256 -> 128 | Linear on cat(LayerNorm(h), cond); i = 0, 1, 2 |
| `blocks.{i}.act` | 128 | SiLU(fc1) |
| `blocks.{i}.fc2` | 128 -> 128 | Linear on act; block output `blocks.{i}` = h + fc2 |
| `out_norm` | 128 | LayerNorm of the last block output |
| `out_act` | 128 | SiLU(out_norm) |
| `out` | 128 -> 2 | Linear; the velocity v |

Edits may target the Linear layers `inp`, `blocks.{0,1,2}.fc1`, `blocks.{0,1,2}.fc2`, `out`. A deployed edit may read only network inputs (x, t, c) and activations. Nothing else about the task is given to you. You may learn more by sampling the policy in the sandbox.

## Terms used by the tools

- Bad basin B_i: the group of 128 policy samples at c_i that contains a_i (no labels used).
- Residual basin mass r_i: share of 256 fresh noises at c_i that the policy decodes into B_i. A cluster's residual is the mean over members; it is open while >= 0.02.
- Branch: restart a decode from step k after renoising: x' = t_k eps + (1 - t_k) x_hat with fresh eps. A rescue ends outside B_i at a typical action.
- p(k): rescue share of branches from step k. k_commit: earliest step after which every later p(k) < 0.2.
- Primitives: `push(u, alpha, k, h)` adds displacement alpha along unit u over steps k..k+h-1; `hop(tau, u)` moves the noise by tau*u at step 0. Class R: some primitive at or before k_commit rescues >= 50% of members. Class U: none does.
- Version: the current policy plus a stack of edits. `v0` is the original policy. Each edit call creates a new candidate version id.

## Commands

Every call: `toyrepair-tool <command> --token <token> [options]`. One command per Bash call. Option values may use only letters, digits and `_ - . = : / , [ ]`. File options name files inside your scratch directory (relative names are resolved there).

{{COMMANDS}}

## Edit reference

{{EDIT_REFERENCE}}

## The gate

`submit_edit --version <id>` sends a candidate to the gate. The gate decodes a fresh batch at all conditions with the original policy, the current version and your candidate, and runs six tests. All must pass:

1. target: the cluster's bad behaviour falls by at least 50% of its current mass, with a 95% lower bound above 0.
2. progress: the overall distance to the desired policy falls (one-sided 95% bound).
3. drift: the change to the behaviour the policy must keep stays at or below 0.10 (one-sided 95% bound).
4. other: other bad behaviours rise by at most 0.01.
5. between: off-distribution actions rise by at most 0.01.
6. unit: correct shapes, no NaN, detectors read inputs and activations only, graft identity.

You get pass or fail for each test, plus test 1's number: the fall in the cluster's bad-behaviour mass as a share of the batch, with its 95% interval. You get no other gate number. Each submission spends one of the run's gate checks.

## Session rules

- Limits: {{CALLS}} tool calls (this `prompt` call counts), {{SUBMISSIONS}} gate submissions. The tool refuses calls beyond them.
- Your token gives access to this run only. Paths outside your scratch directory are refused.
- Use `quick_check` (no gate check spent) before you submit. A good candidate has low `still_bad` and `residual` and small `drift`.
- Fit numbers in the sandbox (least squares or a small optimization) and save arrays as `.npy` in scratch; edit calls read them.
- Decodes cost budget. Keep sandbox decode batches moderate.
- Stop at the first accept, or when the tool reports a limit. Then give your one-line RESULT.
