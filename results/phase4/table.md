# Phase 4 (W2b): iterative arms on cond k4, npm 4 and 16, seeds 0-4

D, Removed, Collateral = toypref `compute` (`tv_to_ideal`, 1 - `neg_mass_ratio`, `retain_tv`) on the final-evaluation bank. B0/B1a: Phase 1 reproduction (results/phase1). Mean ± SE over cells.

| arm | cells | D | Removed | Collateral | accepted edits / run | gate checks / run | D < B1a |
|---|---|---|---|---|---|---|---|
| B0 | 10 | 0.516 ± 0.001 | 0.000 | 0.000 | - | - | 0/10 |
| B1a | 10 | 0.281 ± 0.022 | 0.563 | 0.133 | - | - | - |
| B4 | 10 | 0.464 ± 0.034 | 0.110 | 0.013 | 0.20 | 3.4 | 0/10 |
| R0 | 10 | 0.516 ± 0.001 | 0.000 | 0.000 | 0.00 | 9.0 | 0/10 |
| D1 | 10 | 0.428 ± 0.040 | 0.197 | 0.032 | 0.50 | 11.1 | 0/10 |
| A3 | 9 | 0.173 ± 0.025 | 0.772 | 0.102 | 4.11 | 12.7 | 7/9 |

## Per cell (D)

| cell | B0 | B1a | B4 | R0 | D1 | A3 | A3 stop | A3 sessions | A3 $ |
|---|---|---|---|---|---|---|---|---|---|
| cond_k4_s0_n4 | 0.516 | 0.259 | 0.267 | 0.516 | 0.261 | 0.122 | no_accept_streak | 11 | 6.32 |
| cond_k4_s1_n4 | 0.514 | 0.357 | 0.514 | 0.514 | 0.514 | 0.164 | no_open_cluster | 4 | 1.82 |
| cond_k4_s2_n4 | 0.513 | 0.306 | 0.513 | 0.513 | 0.328 | 0.143 | no_open_cluster | 4 | 2.08 |
| cond_k4_s3_n4 | 0.513 | 0.371 | 0.513 | 0.513 | 0.513 | 0.139 | no_open_cluster | 5 | 2.65 |
| cond_k4_s4_n4 | 0.522 | 0.319 | 0.522 | 0.522 | 0.522 | 0.152 | no_accept_streak | 10 | 5.30 |
| cond_k4_s0_n16 | 0.516 | 0.139 | 0.253 | 0.516 | 0.189 | 0.167 | no_accept_streak | 7 | 3.59 |
| cond_k4_s1_n16 | 0.514 | 0.219 | 0.514 | 0.514 | 0.514 | 0.206 | no_accept_streak | 6 | 3.16 |
| cond_k4_s2_n16 | 0.513 | 0.284 | 0.513 | 0.513 | 0.399 | 0.105 | no_open_cluster | 8 | 3.47 |
| cond_k4_s3_n16 | 0.513 | 0.227 | 0.513 | 0.513 | 0.513 | - | - | - | 0.00 |
| cond_k4_s4_n16 | 0.522 | 0.326 | 0.522 | 0.522 | 0.522 | 0.358 | no_accept_streak | 4 | 2.72 |

A3 cost per run (surgeon + triage subagents): mean $3.46, sd 1.48, total $31.11 over 9 runs.
