# Conditional confirmation, seeds 5-9 (360 runs)

## n_forget=1

                 tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  neg_still_forget_near  neg_still_forget_far  rows_target_frac_in_forget
arm                                                                                                                                       
NPO_b10_neg_A          0.142           0.559      0.021     0.072                  0.124                 0.882                         NaN
NPO_b100_neg           0.143           0.728      0.035     0.055                  0.593                 0.693                         NaN
B0_frozen              0.149           1.000      0.000     0.028                  1.000                 1.000                         NaN
R_hop_bsup_neg         0.187           0.342      0.121     0.086                  0.107                 0.274                       0.106
R_own_nhood_neg        0.249           0.422      0.162     0.114                  0.084                 0.229                       0.059
R_ours_neg             0.524           0.113      0.233     0.496                  0.087                 0.107                       0.110

- R_hop_bsup_neg - NPO_b100_neg: +0.0444 +- 0.0118 (lower in 5/20) fail
- R_hop_bsup_neg - NPO_b10_neg_A: +0.0452 +- 0.0111 (lower in 2/20) fail
- dominated by: -

## n_forget=2

                 tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  neg_still_forget_near  neg_still_forget_far  rows_target_frac_in_forget
arm                                                                                                                                       
R_hop_bsup_neg         0.226           0.424      0.125     0.070                  0.239                 0.342                       0.209
NPO_b10_neg_A          0.235           0.633      0.024     0.081                  0.467                 0.894                         NaN
NPO_b100_neg           0.256           0.832      0.031     0.054                  0.770                 0.802                         NaN
B0_frozen              0.270           1.000      0.000     0.028                  1.000                 1.000                         NaN
R_own_nhood_neg        0.293           0.499      0.158     0.124                  0.285                 0.360                       0.218
R_ours_neg             0.578           0.207      0.232     0.500                  0.140                 0.166                       0.229

- R_hop_bsup_neg - NPO_b100_neg: -0.0299 +- 0.0172 (lower in 14/20) fail
- R_hop_bsup_neg - NPO_b10_neg_A: -0.0085 +- 0.0144 (lower in 13/20) fail
- dominated by: -

## n_forget=4

                 tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  neg_still_forget_near  neg_still_forget_far  rows_target_frac_in_forget
arm                                                                                                                                       
R_hop_bsup_neg         0.398           0.689      0.130     0.063                  0.583                 0.647                       0.569
R_own_nhood_neg        0.456           0.722      0.120     0.105                  0.584                 0.650                       0.556
NPO_b10_neg_A          0.481           0.816      0.020     0.084                  0.772                 0.898                         NaN
NPO_b100_neg           0.500           0.922      0.018     0.052                  0.905                 0.866                         NaN
B0_frozen              0.515           1.000      0.000     0.028                  1.000                 1.000                         NaN
R_ours_neg             0.593           0.391      0.223     0.377                  0.317                 0.239                       0.490

- R_hop_bsup_neg - NPO_b100_neg: -0.1022 +- 0.0222 (lower in 15/20) PASS
- R_hop_bsup_neg - NPO_b10_neg_A: -0.0828 +- 0.0164 (lower in 15/20) PASS
- dominated by: -

**CONFIRMATION at n_forget=4: PASSED**

