"""robo2d: a robotics-faithful toy for negatives-only editing of a chunked flow policy.

A point agent drives from a start to a goal past 1-3 disc obstacles. The policy is a pi0-style
conditional rectified flow over ACTION CHUNKS (H = 8 consecutive 2-D velocity steps, executed
open loop, then replan; <= 6 chunks per episode). Expert demos pass every obstacle on the left
or the right (homotopy classes) and carry two rare weird behaviours injected at a low rate --
chunks that head into an obstacle (collision tails) and chunks with high-frequency zig-zag
(jerk) -- so the base policy reproduces them as tails. Whole episodes that collide or exceed the
jerk threshold are labelled negative; EVERY chunk of a negative episode becomes a negative row
(credit assignment is deliberately left in).

Modules
  seeds    every RNG stream, keyed by (seed, purpose[, id])
  env      setups, observations, execution, collision / jerk / success / homotopy labeller
  demos    expert demonstrations with weird-behaviour injection
  policy   chunk flow v(x_t, t, obs)
  data     base training, logging negatives (labels), evaluation sets
  targets  per-state basins, hop / bsup / local anchor (16-D ports of cond/targets.py)
  methods  arms (B0, NPO, KTO, R_ours, R_hop_bsup(_AL), SDPO variants) and the shared fine-tune loop
  evaluate CRN evaluation and metrics
"""
