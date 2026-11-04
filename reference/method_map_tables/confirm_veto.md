# Round 2 (noise veto) confirmation, seeds 5-9 (300 runs)

## n_forget=1

                 tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  neg_still_forget_near  neg_still_forget_far  rows_target_frac_in_forget
arm                                                                                                                                                          
R_hopB_bsup_neg        0.195           0.254      0.132     0.091              0.329                  0.064                 0.163                       0.061
R_hop_bsup_neg         0.187           0.342      0.121     0.086              0.300                  0.107                 0.274                       0.106
NPO_b100_neg           0.143           0.728      0.035     0.055              0.112                  0.593                 0.693                         NaN
NPO_b10_neg_A          0.142           0.559      0.021     0.072              0.065                  0.124                 0.882                         NaN
B0_frozen              0.149           1.000      0.000     0.028              0.000                  1.000                 1.000                         NaN

- R_hopB_bsup_neg - R_hop_bsup_neg: +0.0082 +- 0.0065 (lower in 4/20) fail
- R_hopB_bsup_neg - NPO_b100_neg: +0.0527 +- 0.0103 (lower in 3/20) fail
- R_hopB_bsup_neg - NPO_b10_neg_A: +0.0535 +- 0.0113 (lower in 3/20) fail

## n_forget=2

                 tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  neg_still_forget_near  neg_still_forget_far  rows_target_frac_in_forget
arm                                                                                                                                                          
R_hopB_bsup_neg        0.224           0.297      0.158     0.066              0.397                  0.113                 0.252                       0.082
R_hop_bsup_neg         0.226           0.424      0.125     0.070              0.321                  0.239                 0.342                       0.209
NPO_b100_neg           0.256           0.832      0.031     0.054              0.095                  0.770                 0.802                         NaN
NPO_b10_neg_A          0.235           0.633      0.024     0.081              0.078                  0.467                 0.894                         NaN
B0_frozen              0.270           1.000      0.000     0.028              0.000                  1.000                 1.000                         NaN

- R_hopB_bsup_neg - R_hop_bsup_neg: -0.0025 +- 0.0167 (lower in 7/20) fail
- R_hopB_bsup_neg - NPO_b100_neg: -0.0323 +- 0.0180 (lower in 14/20) fail
- R_hopB_bsup_neg - NPO_b10_neg_A: -0.0109 +- 0.0179 (lower in 12/20) fail

## n_forget=4

                 tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  neg_still_forget_near  neg_still_forget_far  rows_target_frac_in_forget
arm                                                                                                                                                          
R_hopB_bsup_neg        0.301           0.391      0.157     0.070              0.443                  0.304                 0.511                       0.300
R_hop_bsup_neg         0.398           0.689      0.130     0.063              0.343                  0.583                 0.647                       0.569
NPO_b100_neg           0.500           0.922      0.018     0.052              0.059                  0.905                 0.866                         NaN
NPO_b10_neg_A          0.481           0.816      0.020     0.084              0.081                  0.772                 0.898                         NaN
B0_frozen              0.515           1.000      0.000     0.028              0.000                  1.000                 1.000                         NaN

- R_hopB_bsup_neg - R_hop_bsup_neg: -0.0969 +- 0.0261 (lower in 17/20) PASS
- R_hopB_bsup_neg - NPO_b100_neg: -0.1991 +- 0.0281 (lower in 17/20) PASS
- R_hopB_bsup_neg - NPO_b10_neg_A: -0.1797 +- 0.0230 (lower in 18/20) PASS

**V2 (beats plain hop at n_forget=4): CONFIRMED**
**V3 (beats best neg-only baseline at n_forget=2): NOT confirmed**
n_forget=4 vs both baselines: PASS
