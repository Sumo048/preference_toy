# Where our practical algorithms are best (seeds 0-4; robo seeds with a working base)

## Summary

| toy | setting | best practical arm | our arms tied-best |
|---|---|---|---|
| unconditional | forget 1/8 | B5_kto_b10_negonly_Aself_dist | — |
| unconditional | forget 2/8 | SDPO_fresh_b1 | SDPO_fresh_b1 |
| unconditional | forget 4/8 | SDPO_fresh_b1 | SDPO_fresh_b1, SDPO_pin_b1, R_ours_neg_Aself_dist, R_ours_neg_Aself_map |
| unconditional | overlap std0.45 | B3_npo_b10_negonly_Aself_map | — |
| unconditional | overlap std0.6 | B3_npo_b10_negonly_Aself_map | — |
| conditional | forget 1/8 | KTO_b10_neg_AL | SDPO_fresh_b10, SDPO_fresh_b25 |
| conditional | forget 2/8 | NPO_b10_neg_AL | R_hop_bsup_neg_AL |
| conditional | forget 4/8 | R_hop_bsup_neg | R_hop_bsup_neg, R_hop_nhood_neg |
| conditional | forget 4/8, contig | NPO_b10_neg | SDPO_pin_b25, SDPO_fresh_b1, SDPO_hop_pin_b25, SDPO_hop_pin_bH10 |
| conditional | forget 4/8, alt | R_hop_bsup_neg | R_hop_bsup_neg |
| conditional | forget 4/8, pairs | R_hop_bsup_neg | R_hop_bsup_neg |
| conditional | forget 4/8, b3p1 | R_hop_bsup_neg | R_hop_bsup_neg |
| robo | setups with negatives | SDPO_fresh_b25 | SDPO_fresh_b25, SDPO_pin_b25 |
| robo | setups without negatives | SDPO_fresh_b25 | SDPO_fresh_b25 |
| robo | held-out setups | SDPO_fresh_b25 | SDPO_fresh_b25, SDPO_pin_b25 |

### Unconditional toy, forget 1 of 8 (negatives only)

best: `B5_kto_b10_negonly_Aself_dist` (0.120); tied-best (within 2 sem): `B5_kto_b10_negonly_Aself_dist`, `B3_npo_b10_negonly_Aself_map`, `B3_npo_b100_negonly_Aself_map`
**ours among the best: NONE**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | B5_kto_b10_negonly_Aself_dist |  | 0.120 | 0.704 | 0.011 | 0.036 | yes |
| 2 | B3_npo_b10_negonly_Aself_map |  | 0.122 | 0.191 | 0.044 | 0.082 | yes |
| 3 | B3_npo_b100_negonly_Aself_map |  | 0.124 | 0.233 | 0.043 | 0.080 | yes |
| 4 | B3_npo_b100_negonly |  | 0.132 | 0.596 | 0.029 | 0.060 |  |
| 5 | B3_npo_b10_negonly_Aself_dist |  | 0.132 | 0.772 | 0.009 | 0.039 |  |
| 6 | SDPO_fresh_b25 | yes | 0.136 | 0.785 | 0.046 | 0.026 |  |
| 7 | B5_kto_b100_negonly_Aself_dist |  | 0.142 | 0.872 | 0.008 | 0.037 |  |
| 8 | B0_frozen |  | 0.145 | 1.000 | 0.000 | 0.025 |  |
| 9 | B3_npo_b100_negonly_Aself_dist |  | 0.150 | 0.943 | 0.006 | 0.037 |  |
| 10 | B5_kto_b100_negonly_Aself_map |  | 0.152 | 0.612 | 0.014 | 0.078 |  |

### Unconditional toy, forget 2 of 8 (negatives only)

best: `SDPO_fresh_b1` (0.138); tied-best (within 2 sem): `SDPO_fresh_b1`
**ours among the best: SDPO_fresh_b1**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | SDPO_fresh_b1 | yes | 0.138 | 0.077 | 0.079 | 0.105 | yes |
| 2 | B3_npo_b10_negonly_Aself_map |  | 0.179 | 0.301 | 0.059 | 0.096 |  |
| 3 | B3_npo_b100_negonly_Aself_map |  | 0.184 | 0.322 | 0.058 | 0.098 |  |
| 4 | B5_kto_b10_negonly_Aself_dist |  | 0.213 | 0.734 | 0.019 | 0.035 |  |
| 5 | SDPO_pin_b25 | yes | 0.229 | 0.521 | 0.126 | 0.064 |  |
| 6 | R_ours_neg_Aself_dist | yes | 0.235 | 0.592 | 0.124 | 0.048 |  |
| 7 | B3_npo_b10_negonly_Aself_dist |  | 0.237 | 0.818 | 0.013 | 0.039 |  |
| 8 | B3_npo_b100_negonly |  | 0.237 | 0.669 | 0.049 | 0.076 |  |
| 9 | R_ours_neg_Aself_map | yes | 0.240 | 0.601 | 0.075 | 0.081 |  |
| 10 | B5_kto_b100_negonly_Aself_dist |  | 0.246 | 0.862 | 0.010 | 0.038 |  |

### Unconditional toy, forget 4 of 8 (negatives only)

best: `SDPO_fresh_b1` (0.372); tied-best (within 2 sem): `SDPO_fresh_b1`, `SDPO_pin_b1`, `R_ours_neg_Aself_dist`, `R_ours_neg_Aself_map`
**ours among the best: SDPO_fresh_b1, SDPO_pin_b1, R_ours_neg_Aself_dist, R_ours_neg_Aself_map**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | SDPO_fresh_b1 | yes | 0.372 | 0.513 | 0.024 | 0.124 | yes |
| 2 | SDPO_pin_b1 | yes | 0.386 | 0.133 | 0.241 | 0.269 | yes |
| 3 | R_ours_neg_Aself_dist | yes | 0.399 | 0.724 | 0.062 | 0.049 | yes |
| 4 | B3_npo_b10_negonly_Aself_map |  | 0.400 | 0.659 | 0.026 | 0.080 |  |
| 5 | B3_npo_b100_negonly_Aself_map |  | 0.406 | 0.692 | 0.026 | 0.071 |  |
| 6 | R_ours_neg_Aself_map | yes | 0.412 | 0.724 | 0.050 | 0.061 | yes |
| 7 | SDPO_pin_b25 | yes | 0.416 | 0.672 | 0.100 | 0.088 |  |
| 8 | B5_kto_b10_negonly_Aself_dist |  | 0.445 | 0.836 | 0.025 | 0.041 |  |
| 9 | R_ours_neg | yes | 0.452 | 0.549 | 0.230 | 0.156 |  |
| 10 | R_spread_nhood_neg_Aself_dist | yes | 0.465 | 0.751 | 0.150 | 0.101 |  |

### Unconditional toy, overlapping modes std0.45

best: `B3_npo_b10_negonly_Aself_map` (0.190); tied-best (within 2 sem): `B3_npo_b10_negonly_Aself_map`
**ours among the best: NONE**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | B3_npo_b10_negonly_Aself_map |  | 0.190 | 0.491 | 0.032 | 0.020 | yes |
| 2 | B3_npo_b100_negonly_Aself_map |  | 0.197 | 0.524 | 0.031 | 0.018 |  |
| 3 | B5_kto_b100_negonly_Aself_map |  | 0.226 | 0.697 | 0.022 | 0.011 |  |
| 4 | B5_kto_b10_negonly_Aself_dist |  | 0.232 | 0.738 | 0.019 | 0.009 |  |
| 5 | B5_kto_b10_negonly_Aself_map |  | 0.234 | 0.733 | 0.020 | 0.010 |  |
| 6 | R_ours_neg_Aself_map | yes | 0.235 | 0.650 | 0.067 | 0.021 |  |
| 7 | SDPO_pin_b25 | yes | 0.248 | 0.584 | 0.098 | 0.018 |  |
| 8 | R_ours_neg_Aself_dist | yes | 0.249 | 0.616 | 0.101 | 0.015 |  |
| 9 | B3_npo_b10_negonly_Aself_dist |  | 0.262 | 0.847 | 0.010 | 0.008 |  |
| 10 | B3_npo_b100_negonly |  | 0.264 | 0.780 | 0.031 | 0.015 |  |

### Unconditional toy, overlapping modes std0.6

best: `B3_npo_b10_negonly_Aself_map` (0.191); tied-best (within 2 sem): `B3_npo_b10_negonly_Aself_map`
**ours among the best: NONE**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | B3_npo_b10_negonly_Aself_map |  | 0.191 | 0.533 | 0.030 | 0.012 | yes |
| 2 | B3_npo_b100_negonly_Aself_map |  | 0.198 | 0.566 | 0.030 | 0.011 |  |
| 3 | B5_kto_b100_negonly_Aself_map |  | 0.219 | 0.688 | 0.022 | 0.006 |  |
| 4 | B5_kto_b10_negonly_Aself_map |  | 0.225 | 0.708 | 0.022 | 0.005 |  |
| 5 | B5_kto_b10_negonly_Aself_dist |  | 0.225 | 0.728 | 0.020 | 0.005 |  |
| 6 | R_ours_neg_Aself_map | yes | 0.233 | 0.658 | 0.070 | 0.010 |  |
| 7 | R_ours_neg_Aself_dist | yes | 0.250 | 0.627 | 0.104 | 0.008 |  |
| 8 | SDPO_pin_b25 | yes | 0.257 | 0.573 | 0.115 | 0.010 |  |
| 9 | B3_npo_b10_negonly_Aself_dist |  | 0.258 | 0.846 | 0.012 | 0.004 |  |
| 10 | B5_kto_b100_negonly |  | 0.261 | 0.304 | 0.147 | 0.063 |  |

### Conditional toy, forget 1/8

best: `KTO_b10_neg_AL` (0.119); tied-best (within 2 sem): `KTO_b10_neg_AL`, `NPO_b10_neg_AL`, `NPO_b100_neg_AL`, `SDPO_fresh_b10`, `SDPO_fresh_b25`
**ours among the best: SDPO_fresh_b10, SDPO_fresh_b25**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | KTO_b10_neg_AL |  | 0.119 | 0.162 | 0.051 | 0.085 | yes |
| 2 | NPO_b10_neg_AL |  | 0.119 | 0.183 | 0.051 | 0.084 | yes |
| 3 | NPO_b100_neg_AL |  | 0.121 | 0.274 | 0.046 | 0.078 | yes |
| 4 | SDPO_fresh_b10 | yes | 0.129 | 0.722 | 0.047 | 0.031 | yes |
| 5 | SDPO_fresh_b25 | yes | 0.135 | 0.876 | 0.021 | 0.027 | yes |
| 6 | R_hop_bsup_neg_AL | yes | 0.140 | 0.332 | 0.065 | 0.068 |  |
| 7 | NPO_b10_neg_A |  | 0.142 | 0.548 | 0.023 | 0.073 |  |
| 8 | SDPO_fresh_b100 | yes | 0.143 | 0.953 | 0.008 | 0.027 |  |
| 9 | NPO_b100_neg |  | 0.143 | 0.737 | 0.031 | 0.053 |  |
| 10 | B0_frozen |  | 0.149 | 1.000 | 0.000 | 0.027 |  |

### Conditional toy, forget 2/8

best: `NPO_b10_neg_AL` (0.200); tied-best (within 2 sem): `NPO_b10_neg_AL`, `R_hop_bsup_neg_AL`
**ours among the best: R_hop_bsup_neg_AL**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | NPO_b10_neg_AL |  | 0.200 | 0.423 | 0.047 | 0.097 | yes |
| 2 | R_hop_bsup_neg_AL | yes | 0.201 | 0.516 | 0.070 | 0.055 | yes |
| 3 | NPO_b100_neg_AL |  | 0.209 | 0.495 | 0.044 | 0.088 |  |
| 4 | KTO_b10_neg_AL |  | 0.227 | 0.505 | 0.036 | 0.105 |  |
| 5 | NPO_b10_neg_A |  | 0.234 | 0.610 | 0.024 | 0.086 |  |
| 6 | R_hop_nhood_neg_A | yes | 0.239 | 0.740 | 0.041 | 0.052 |  |
| 7 | SDPO_hop_pin_b25 | yes | 0.248 | 0.579 | 0.118 | 0.082 |  |
| 8 | SDPO_fresh_b10 | yes | 0.252 | 0.884 | 0.028 | 0.038 |  |
| 9 | NPO_b100_neg |  | 0.255 | 0.829 | 0.035 | 0.054 |  |
| 10 | R_hop_bsup_neg | yes | 0.256 | 0.431 | 0.160 | 0.078 |  |

### Conditional toy, forget 4/8

best: `R_hop_bsup_neg` (0.355); tied-best (within 2 sem): `R_hop_bsup_neg`, `R_hop_nhood_neg`
**ours among the best: R_hop_bsup_neg, R_hop_nhood_neg**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | R_hop_bsup_neg | yes | 0.355 | 0.611 | 0.132 | 0.052 | yes |
| 2 | R_hop_nhood_neg | yes | 0.369 | 0.657 | 0.124 | 0.044 | yes |
| 3 | R_hop_bsup_neg_AL | yes | 0.401 | 0.727 | 0.046 | 0.046 |  |
| 4 | NPO_b10_neg_AL |  | 0.424 | 0.688 | 0.027 | 0.088 |  |
| 5 | NPO_b100_neg_AL |  | 0.431 | 0.720 | 0.022 | 0.080 |  |
| 6 | R_basin_bsup_neg | yes | 0.436 | 0.720 | 0.132 | 0.085 |  |
| 7 | R_hop_nhood_neg_A | yes | 0.451 | 0.828 | 0.026 | 0.046 |  |
| 8 | NPO_b10_neg_A |  | 0.455 | 0.765 | 0.016 | 0.081 |  |
| 9 | SDPO_hop_pin_b25 | yes | 0.460 | 0.597 | 0.114 | 0.169 |  |
| 10 | KTO_b10_neg_AL |  | 0.462 | 0.726 | 0.022 | 0.108 |  |

### Conditional toy, forget 4/8, contig

best: `NPO_b10_neg` (0.431); tied-best (within 2 sem): `NPO_b10_neg`, `SDPO_pin_b25`, `SDPO_fresh_b1`, `SDPO_hop_pin_b25`, `SDPO_hop_pin_bH10`
**ours among the best: SDPO_pin_b25, SDPO_fresh_b1, SDPO_hop_pin_b25, SDPO_hop_pin_bH10**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | NPO_b10_neg |  | 0.431 | 0.449 | 0.129 | 0.213 | yes |
| 2 | SDPO_pin_b25 | yes | 0.431 | 0.729 | 0.067 | 0.077 | yes |
| 3 | SDPO_fresh_b1 | yes | 0.434 | 0.452 | 0.070 | 0.215 | yes |
| 4 | SDPO_hop_pin_b25 | yes | 0.435 | 0.674 | 0.117 | 0.108 | yes |
| 5 | SDPO_hop_pin_bH10 | yes | 0.453 | 0.402 | 0.212 | 0.237 | yes |
| 6 | KTO_b100_neg |  | 0.471 | 0.521 | 0.085 | 0.218 |  |
| 7 | R_hop_bsup_neg | yes | 0.471 | 0.819 | 0.108 | 0.073 |  |
| 8 | NPO_b10_neg_AL |  | 0.477 | 0.773 | 0.025 | 0.102 |  |
| 9 | R_hop_bsup_neg_AL | yes | 0.479 | 0.874 | 0.042 | 0.054 |  |
| 10 | NPO_b100_neg_AL |  | 0.481 | 0.802 | 0.024 | 0.092 |  |

### Conditional toy, forget 4/8, alt

best: `R_hop_bsup_neg` (0.261); tied-best (within 2 sem): `R_hop_bsup_neg`
**ours among the best: R_hop_bsup_neg**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | R_hop_bsup_neg | yes | 0.261 | 0.422 | 0.118 | 0.049 | yes |
| 2 | R_hop_bsup_neg_AL | yes | 0.347 | 0.630 | 0.031 | 0.041 |  |
| 3 | NPO_b10_neg_AL |  | 0.430 | 0.669 | 0.025 | 0.104 |  |
| 4 | NPO_b100_neg_AL |  | 0.438 | 0.710 | 0.023 | 0.092 |  |
| 5 | NPO_b10_neg_A |  | 0.441 | 0.727 | 0.009 | 0.087 |  |
| 6 | KTO_b10_neg_AL |  | 0.477 | 0.731 | 0.024 | 0.121 |  |
| 7 | SDPO_hop_pin_b25 | yes | 0.480 | 0.633 | 0.120 | 0.172 |  |
| 8 | SDPO_fresh_b10 | yes | 0.511 | 0.966 | 0.006 | 0.041 |  |
| 9 | SDPO_fresh_b25 | yes | 0.512 | 0.987 | 0.003 | 0.031 |  |
| 10 | SDPO_fresh_b100 | yes | 0.513 | 0.996 | 0.001 | 0.028 |  |

### Conditional toy, forget 4/8, pairs

best: `R_hop_bsup_neg` (0.345); tied-best (within 2 sem): `R_hop_bsup_neg`
**ours among the best: R_hop_bsup_neg**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | R_hop_bsup_neg | yes | 0.345 | 0.580 | 0.108 | 0.061 | yes |
| 2 | NPO_b10_neg_AL |  | 0.417 | 0.673 | 0.018 | 0.088 |  |
| 3 | SDPO_hop_pin_b25 | yes | 0.425 | 0.558 | 0.098 | 0.151 |  |
| 4 | R_hop_bsup_neg_AL | yes | 0.425 | 0.764 | 0.029 | 0.052 |  |
| 5 | NPO_b100_neg_AL |  | 0.426 | 0.700 | 0.019 | 0.084 |  |
| 6 | SDPO_hop_pin_bH10 | yes | 0.440 | 0.302 | 0.165 | 0.285 |  |
| 7 | NPO_b10_neg_A |  | 0.453 | 0.767 | 0.008 | 0.078 |  |
| 8 | KTO_b10_neg_AL |  | 0.459 | 0.721 | 0.017 | 0.107 |  |
| 9 | SDPO_pin_b25 | yes | 0.486 | 0.805 | 0.057 | 0.092 |  |
| 10 | SDPO_hop_pin_bH3 | yes | 0.503 | 0.207 | 0.225 | 0.385 |  |

### Conditional toy, forget 4/8, b3p1

best: `R_hop_bsup_neg` (0.376); tied-best (within 2 sem): `R_hop_bsup_neg`
**ours among the best: R_hop_bsup_neg**

| rank | arm | ours? | tv | forget | collateral_retainTV | off | tied-best |
|---|---|---|---|---|---|---|---|
| 1 | R_hop_bsup_neg | yes | 0.376 | 0.643 | 0.120 | 0.061 | yes |
| 2 | R_hop_bsup_neg_AL | yes | 0.439 | 0.790 | 0.024 | 0.054 |  |
| 3 | SDPO_hop_pin_b25 | yes | 0.440 | 0.630 | 0.097 | 0.133 |  |
| 4 | NPO_b10_neg_AL |  | 0.446 | 0.723 | 0.028 | 0.094 |  |
| 5 | NPO_b100_neg_AL |  | 0.451 | 0.747 | 0.025 | 0.087 |  |
| 6 | SDPO_hop_pin_bH10 | yes | 0.464 | 0.403 | 0.170 | 0.262 |  |
| 7 | SDPO_pin_b25 | yes | 0.466 | 0.828 | 0.037 | 0.062 |  |
| 8 | NPO_b10_neg_A |  | 0.467 | 0.789 | 0.013 | 0.082 |  |
| 9 | KTO_b10_neg_AL |  | 0.474 | 0.745 | 0.025 | 0.111 |  |
| 10 | SDPO_fresh_b10 | yes | 0.504 | 0.958 | 0.009 | 0.037 |  |

_robo: dropped seed(s) [3]: frozen base failed (B0 clean success [0.0])_

### Robo toy, setups with negatives (seeds [0, 1, 2, 4])

best: `SDPO_fresh_b25` (0.875); tied-best (within 2 sem): `SDPO_fresh_b25`, `SDPO_pin_b25`
**ours among the best: SDPO_fresh_b25, SDPO_pin_b25**

| rank | arm | ours? | clean_success | bad | collision | jerk | behaviour_change | tied-best |
|---|---|---|---|---|---|---|---|---|
| 1 | SDPO_fresh_b25 | yes | 0.875 | 0.122 | 0.067 | 0.062 | 0.074 | yes |
| 2 | SDPO_pin_b25 | yes | 0.855 | 0.139 | 0.083 | 0.072 | 0.184 | yes |
| 3 | B0_frozen |  | 0.854 | 0.145 | 0.090 | 0.064 | 0.000 |  |
| 4 | R_hop_bsup_neg_AL | yes | 0.848 | 0.151 | 0.148 | 0.003 | 0.213 |  |
| 5 | KTO_b100_neg |  | 0.827 | 0.169 | 0.122 | 0.056 | 0.144 |  |
| 6 | NPO_b100_neg |  | 0.810 | 0.185 | 0.138 | 0.057 | 0.166 |  |
| 7 | SDPO_hop_pin_b25 | yes | 0.791 | 0.203 | 0.157 | 0.060 | 0.337 |  |
| 8 | R_hop_bsup_neg | yes | 0.771 | 0.228 | 0.227 | 0.002 | 0.350 |  |
| 9 | R_ours_neg | yes | 0.743 | 0.240 | 0.141 | 0.129 | 0.163 |  |
| 10 | NPO_b10_neg |  | 0.237 | 0.725 | 0.152 | 0.625 | 0.280 |  |

_robo: dropped seed(s) [3]: frozen base failed (B0 clean success [0.0])_

### Robo toy, setups without negatives (seeds [0, 1, 2, 4])

best: `SDPO_fresh_b25` (0.869); tied-best (within 2 sem): `SDPO_fresh_b25`
**ours among the best: SDPO_fresh_b25**

| rank | arm | ours? | clean_success | bad | collision | jerk | behaviour_change | tied-best |
|---|---|---|---|---|---|---|---|---|
| 1 | SDPO_fresh_b25 | yes | 0.869 | 0.129 | 0.076 | 0.060 | 0.075 | yes |
| 2 | SDPO_pin_b25 | yes | 0.855 | 0.141 | 0.089 | 0.074 | 0.184 |  |
| 3 | B0_frozen |  | 0.851 | 0.148 | 0.095 | 0.063 | 0.000 |  |
| 4 | R_hop_bsup_neg_AL | yes | 0.839 | 0.161 | 0.158 | 0.003 | 0.220 |  |
| 5 | KTO_b100_neg |  | 0.833 | 0.163 | 0.113 | 0.061 | 0.156 |  |
| 6 | NPO_b100_neg |  | 0.811 | 0.184 | 0.137 | 0.060 | 0.164 |  |
| 7 | SDPO_hop_pin_b25 | yes | 0.762 | 0.227 | 0.175 | 0.064 | 0.345 |  |
| 8 | R_hop_bsup_neg | yes | 0.737 | 0.263 | 0.261 | 0.003 | 0.365 |  |
| 9 | R_ours_neg | yes | 0.731 | 0.251 | 0.146 | 0.135 | 0.154 |  |
| 10 | NPO_b10_neg |  | 0.216 | 0.757 | 0.155 | 0.658 | 0.300 |  |

_robo: dropped seed(s) [3]: frozen base failed (B0 clean success [0.0])_

### Robo toy, held-out setups (seeds [0, 1, 2, 4])

best: `SDPO_fresh_b25` (0.873); tied-best (within 2 sem): `SDPO_fresh_b25`, `SDPO_pin_b25`
**ours among the best: SDPO_fresh_b25, SDPO_pin_b25**

| rank | arm | ours? | clean_success | bad | collision | jerk | behaviour_change | tied-best |
|---|---|---|---|---|---|---|---|---|
| 1 | SDPO_fresh_b25 | yes | 0.873 | 0.126 | 0.070 | 0.063 | 0.071 | yes |
| 2 | SDPO_pin_b25 | yes | 0.861 | 0.132 | 0.079 | 0.069 | 0.184 | yes |
| 3 | B0_frozen |  | 0.854 | 0.146 | 0.089 | 0.064 | 0.000 |  |
| 4 | R_hop_bsup_neg_AL | yes | 0.850 | 0.150 | 0.148 | 0.003 | 0.216 |  |
| 5 | KTO_b100_neg |  | 0.839 | 0.159 | 0.110 | 0.059 | 0.133 |  |
| 6 | NPO_b100_neg |  | 0.826 | 0.171 | 0.122 | 0.058 | 0.152 |  |
| 7 | SDPO_hop_pin_b25 | yes | 0.804 | 0.176 | 0.145 | 0.037 | 0.324 |  |
| 8 | R_hop_bsup_neg | yes | 0.791 | 0.209 | 0.208 | 0.002 | 0.351 |  |
| 9 | R_ours_neg | yes | 0.754 | 0.231 | 0.129 | 0.130 | 0.158 |  |
| 10 | NPO_b10_neg |  | 0.278 | 0.683 | 0.141 | 0.588 | 0.276 |  |

