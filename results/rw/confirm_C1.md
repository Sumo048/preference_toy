# C1 confirmation (fresh seeds 5-9, full n_neg grid)

runs: 1080 ({'B0_frozen': 120, 'B3_npo_b100_negonly': 120, 'B3_npo_b10_negonly': 120, 'B5_kto_b100_negonly': 120, 'B5_kto_b10_negonly': 120, 'R_near_nf_log_neg': 120, 'R_near_nf_nhood_neg': 120, 'R_ours_neg': 120, 'R_ours_nf_neg': 120})

| n_forget | R_near_nf_nhood_neg | B3_npo_b10_negonly | B3_npo_b100_negonly | B5_kto_b10_negonly | B5_kto_b100_negonly | R_ours_neg | B0 |
|---|---|---|---|---|---|---|---|
| 1 | 0.310 | 0.225 | 0.124 | 0.230 | 0.170 | 0.555 | 0.147 |
| 2 | 0.328 | 0.527 | 0.247 | 0.519 | 0.321 | 0.508 | 0.269 |
| 3 | 0.310 | 0.410 | 0.362 | 0.478 | 0.297 | 0.469 | 0.390 |
| 4 | 0.290 | 0.566 | 0.498 | 0.875 | 0.565 | 0.471 | 0.514 |

## C1 test: R minus each non-rerouting arm, paired over (seed, n_neg), n = 30

| n_forget | vs arm | diff | sem | R lower in | passes (< -2 sem) |
|---|---|---|---|---|---|
| 1 | B3_npo_b10_negonly | +0.085 | 0.023 | 5/30 | N |
| 1 | B3_npo_b100_negonly | +0.186 | 0.018 | 0/30 | N |
| 1 | B5_kto_b10_negonly | +0.079 | 0.019 | 9/30 | N |
| 1 | B5_kto_b100_negonly | +0.140 | 0.022 | 3/30 | N |
| 2 | B3_npo_b10_negonly | -0.199 | 0.067 | 19/30 | Y |
| 2 | B3_npo_b100_negonly | +0.081 | 0.024 | 10/30 | N |
| 2 | B5_kto_b10_negonly | -0.191 | 0.072 | 18/30 | Y |
| 2 | B5_kto_b100_negonly | +0.007 | 0.050 | 13/30 | N |
| 3 | B3_npo_b10_negonly | -0.100 | 0.039 | 20/30 | Y |
| 3 | B3_npo_b100_negonly | -0.052 | 0.022 | 20/30 | Y |
| 3 | B5_kto_b10_negonly | -0.168 | 0.062 | 17/30 | Y |
| 3 | B5_kto_b100_negonly | +0.013 | 0.047 | 10/30 | N |
| 4 | B3_npo_b10_negonly | -0.276 | 0.033 | 28/30 | Y |
| 4 | B3_npo_b100_negonly | -0.208 | 0.018 | 29/30 | Y |
| 4 | B5_kto_b10_negonly | -0.586 | 0.051 | 28/30 | Y |
| 4 | B5_kto_b100_negonly | -0.276 | 0.046 | 25/30 | Y |

- n_forget=1: beats every non-rerouting neg-only arm: **NO**; dominated by: ['B3_npo_b10_negonly', 'B5_kto_b100_negonly']
- n_forget=2: beats every non-rerouting neg-only arm: **NO**; dominated by: -
- n_forget=3: beats every non-rerouting neg-only arm: **NO**; dominated by: -
- n_forget=4: beats every non-rerouting neg-only arm: **YES**; dominated by: -

**C1 (n_forget 3 and 4): NOT CONFIRMED**; side prediction (no win at n_forget=1): held

neg_mass_ratio by n_forget

n_forget                 1      2      3      4
arm                                            
B0_frozen            1.000  1.000  1.000  1.000
B3_npo_b100_negonly  0.586  0.734  0.890  0.935
B3_npo_b10_negonly   0.019  0.001  0.069  0.286
B5_kto_b100_negonly  0.023  0.020  0.040  0.062
B5_kto_b10_negonly   0.000  0.000  0.000  0.000
R_near_nf_log_neg    0.171  0.098  0.042  0.062
R_near_nf_nhood_neg  0.099  0.053  0.013  0.030
R_ours_neg           0.116  0.207  0.352  0.521
R_ours_nf_neg        0.054  0.035  0.042  0.076

retain_tv by n_forget

n_forget                 1      2      3      4
arm                                            
B0_frozen            0.000  0.000  0.000  0.000
B3_npo_b100_negonly  0.034  0.048  0.018  0.010
B3_npo_b10_negonly   0.202  0.360  0.304  0.214
B5_kto_b100_negonly  0.138  0.217  0.197  0.348
B5_kto_b10_negonly   0.202  0.370  0.339  0.617
R_near_nf_log_neg    0.195  0.186  0.206  0.228
R_near_nf_nhood_neg  0.258  0.289  0.293  0.268
R_ours_neg           0.352  0.386  0.357  0.325
R_ours_nf_neg        0.355  0.370  0.305  0.244

mass_off by n_forget

n_forget                 1      2      3      4
arm                                            
B0_frozen            0.025  0.025  0.025  0.025
B3_npo_b100_negonly  0.048  0.067  0.037  0.041
B3_npo_b10_negonly   0.106  0.458  0.294  0.415
B5_kto_b100_negonly  0.106  0.259  0.252  0.483
B5_kto_b10_negonly   0.155  0.466  0.411  0.857
R_near_nf_log_neg    0.199  0.259  0.199  0.151
R_near_nf_nhood_neg  0.127  0.121  0.070  0.056
R_ours_neg           0.474  0.302  0.214  0.106
R_ours_nf_neg        0.487  0.282  0.190  0.092

tv_to_ideal by n_neg at n_forget >= 3

n_neg                   1      2      4      8      16     32
arm                                                          
B0_frozen            0.452  0.452  0.452  0.452  0.452  0.452
B3_npo_b100_negonly  0.425  0.430  0.432  0.432  0.430  0.431
B3_npo_b10_negonly   0.507  0.534  0.464  0.481  0.487  0.455
B5_kto_b100_negonly  0.389  0.392  0.384  0.434  0.490  0.498
B5_kto_b10_negonly   0.689  0.704  0.591  0.675  0.729  0.675
R_near_nf_log_neg    0.329  0.309  0.276  0.306  0.311  0.270
R_near_nf_nhood_neg  0.343  0.289  0.291  0.289  0.301  0.286
R_ours_neg           0.506  0.462  0.432  0.462  0.477  0.480
R_ours_nf_neg        0.434  0.328  0.336  0.309  0.334  0.327
