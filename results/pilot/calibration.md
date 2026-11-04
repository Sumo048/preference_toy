# Gate calibration: B4 first repair on the 12 pilot cells (W2a)

Development thresholds: drift ≤ 0.10, other ≤ 0.01, between ≤ 0.01. Loosened: 0.15 / 0.02 / 0.02.

| cell | failed tests (dev) | target fall % | Collateral UB | other rise UB | between rise UB | passes loosened |
|---|---|---|---|---|---|---|
| cond_alt_s0_n16 | drift, between | 54 | 0.234 | -0.103 | 0.0674 | no |
| cond_alt_s0_n4 | target, drift, between | -39 | 0.378 | -0.171 | 0.0354 | no |
| cond_alt_s1_n16 | target, drift, between | 19 | 0.285 | -0.141 | 0.0449 | no |
| cond_alt_s1_n4 | target, drift, between | 5 | 0.182 | -0.107 | 0.0469 | no |
| cond_alt_s2_n16 | target, drift, between | 26 | 0.159 | -0.105 | 0.0564 | no |
| cond_alt_s2_n4 | drift, between | 75 | 0.252 | 0.010 | 0.0452 | no |
| cond_k4_s0_n16 | between | 58 | 0.038 | -0.262 | 0.0247 | no |
| cond_k4_s0_n4 | between | 97 | 0.096 | -0.147 | 0.0167 | yes |
| cond_k4_s1_n16 | drift, between | 57 | 0.138 | -0.191 | 0.0209 | no |
| cond_k4_s1_n4 | target, drift, between | 30 | 0.200 | -0.127 | 0.0106 | no |
| cond_k4_s2_n16 | target, between | 39 | 0.070 | -0.135 | 0.0142 | no |
| cond_k4_s2_n4 | target, drift, between | 43 | 0.110 | -0.140 | 0.0391 | no |

Accepted at development thresholds: 0 of 12. At the loosened thresholds: 1 of 12.
