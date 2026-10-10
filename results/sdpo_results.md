# Self-paired DPO: results in every previous toy setting

## Unconditional toy

### main sweep, n_forget=1 (seeds 0-4, n_neg 1-32; RW arms only on n_neg 1/4/16)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.501           0.005      0.414     0.284              0.734                    0.152
SDPO_pin_b1                        0.600           0.127      0.453     0.518              0.776                    0.152
SDPO_fresh_b25                     0.133           0.808      0.035     0.025              0.101                    0.152
SDPO_pin_b25                       0.159           0.245      0.114     0.073              0.308                    0.152
SDPO_nearnf_pin_b1                 0.476           0.000      0.401     0.342              0.575                    0.000
B3_npo_b100_negonly                0.132           0.626      0.029     0.056              0.112                      NaN
B3_npo_b10_negonly                 0.222           0.009      0.185     0.136              0.440                      NaN
B5_kto_b100_negonly                0.180           0.001      0.116     0.157              0.272                      NaN
B5_kto_b10_negonly                 0.257           0.000      0.204     0.206              0.491                      NaN
R_ours_neg                         0.611           0.145      0.340     0.553              0.846                      NaN
R_near_nf_nhood_neg                0.301           0.168      0.222     0.133              0.335                      NaN
R_near_nf_log_neg                  0.250           0.243      0.128     0.172              0.294                      NaN
B5_kto_b10_negonly_Anf_dist        0.039           0.002      0.012     0.037              0.188                      NaN
PSEUDO_nf_dist                     0.039           0.012      0.011     0.037              0.181                      NaN
B0_frozen                          0.145           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_fresh_b25` (0.133)
- vs `B3_npo_b100_negonly` (0.132): +0.0005 +- 0.0041, lower in 15/30 -> **tie**
- vs `R_ours_neg` (0.611): -0.4781 +- 0.0286, lower in 30/30 -> **better**
- vs `R_near_nf_nhood_neg` (0.301): -0.1648 +- 0.0294, lower in 15/15 -> **better**
- vs `B5_kto_b10_negonly_Anf_dist` (0.039): +0.0974 +- 0.0056, lower in 0/15 -> **worse**

### main sweep, n_forget=2 (seeds 0-4, n_neg 1-32; RW arms only on n_neg 1/4/16)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.126           0.075      0.068     0.098              0.215                    0.234
SDPO_pin_b1                        0.533           0.187      0.393     0.396              0.660                    0.234
SDPO_fresh_b25                     0.258           0.959      0.003     0.027              0.008                    0.234
SDPO_pin_b25                       0.213           0.481      0.117     0.064              0.271                    0.234
SDPO_nearnf_pin_b1                 0.655           0.000      0.567     0.540              0.732                    0.000
B3_npo_b100_negonly                0.236           0.707      0.043     0.065              0.127                      NaN
B3_npo_b10_negonly                 0.510           0.002      0.384     0.385              0.709                      NaN
B5_kto_b100_negonly                0.383           0.029      0.223     0.346              0.515                      NaN
B5_kto_b10_negonly                 0.512           0.000      0.421     0.395              0.711                      NaN
R_ours_neg                         0.557           0.319      0.383     0.350              0.805                      NaN
R_near_nf_nhood_neg                0.320           0.111      0.276     0.095              0.336                      NaN
R_near_nf_log_neg                  0.289           0.194      0.175     0.183              0.312                      NaN
B5_kto_b10_negonly_Anf_dist        0.039           0.004      0.020     0.035              0.201                      NaN
PSEUDO_nf_dist                     0.041           0.009      0.016     0.037              0.197                      NaN
B0_frozen                          0.266           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_fresh_b1` (0.126)
- vs `B3_npo_b100_negonly` (0.236): -0.1094 +- 0.0092, lower in 28/30 -> **better**
- vs `R_ours_neg` (0.557): -0.4305 +- 0.0167, lower in 30/30 -> **better**
- vs `R_near_nf_nhood_neg` (0.320): -0.1811 +- 0.0233, lower in 14/15 -> **better**
- vs `B5_kto_b10_negonly_Anf_dist` (0.039): +0.0995 +- 0.0139, lower in 0/15 -> **worse**

### main sweep, n_forget=3 (seeds 0-4, n_neg 1-32; RW arms only on n_neg 1/4/16)

                     tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                              
SDPO_fresh_b1              0.234           0.262      0.027     0.139              0.060                    0.362
SDPO_pin_b1                0.461           0.149      0.330     0.325              0.553                    0.362
SDPO_fresh_b25             0.383           0.981      0.002     0.026              0.003                    0.362
SDPO_pin_b25               0.312           0.638      0.076     0.072              0.133                    0.362
SDPO_nearnf_pin_b1         0.620           0.015      0.484     0.495              0.697                    0.011
B3_npo_b100_negonly        0.374           0.914      0.007     0.041              0.028                      NaN
B5_kto_b100_negonly        0.503           0.099      0.240     0.461              0.571                      NaN
R_ours_neg                 0.447           0.467      0.309     0.155              0.571                      NaN
B0_frozen                  0.388           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_fresh_b1` (0.234)
- vs `B3_npo_b100_negonly` (0.374): -0.1398 +- 0.0057, lower in 30/30 -> **better**
- vs `R_ours_neg` (0.447): -0.2129 +- 0.0234, lower in 30/30 -> **better**

### main sweep, n_forget=4 (seeds 0-4, n_neg 1-32; RW arms only on n_neg 1/4/16)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.374           0.538      0.019     0.114              0.018                    0.484
SDPO_pin_b1                        0.375           0.144      0.217     0.266              0.411                    0.484
SDPO_fresh_b25                     0.504           0.989      0.001     0.026              0.001                    0.484
SDPO_pin_b25                       0.428           0.701      0.077     0.087              0.118                    0.484
SDPO_nearnf_pin_b1                 0.549           0.010      0.459     0.415              0.627                    0.008
B3_npo_b100_negonly                0.498           0.958      0.005     0.035              0.010                      NaN
B3_npo_b10_negonly                 0.516           0.360      0.069     0.341              0.253                      NaN
B5_kto_b100_negonly                0.542           0.155      0.156     0.467              0.412                      NaN
B5_kto_b10_negonly                 0.996           0.000      0.642     0.996              1.000                      NaN
R_ours_neg                         0.425           0.570      0.235     0.114              0.489                      NaN
R_near_nf_nhood_neg                0.180           0.080      0.159     0.040              0.193                      NaN
R_near_nf_log_neg                  0.273           0.130      0.167     0.173              0.293                      NaN
B5_kto_b10_negonly_Anf_dist        0.042           0.008      0.019     0.035              0.098                      NaN
PSEUDO_nf_dist                     0.043           0.011      0.014     0.036              0.101                      NaN
B0_frozen                          0.509           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_fresh_b1` (0.374)
- vs `B3_npo_b100_negonly` (0.498): -0.1242 +- 0.0039, lower in 30/30 -> **better**
- vs `R_ours_neg` (0.425): -0.0513 +- 0.0221, lower in 18/30 -> **better**
- vs `R_near_nf_nhood_neg` (0.180): +0.1924 +- 0.0282, lower in 1/15 -> **worse**
- vs `B5_kto_b10_negonly_Anf_dist` (0.042): +0.3298 +- 0.0086, lower in 0/15 -> **worse**

### seeds 5-9, n_forget=1 (n_neg 1/4/16)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.484           0.009      0.414     0.231              0.715                    0.122
SDPO_pin_b1                        0.549           0.012      0.467     0.465              0.787                    0.122
SDPO_fresh_b25                     0.127           0.732      0.030     0.028              0.097                    0.122
SDPO_pin_b25                       0.179           0.185      0.151     0.065              0.348                    0.122
SDPO_nearnf_pin_b1                 0.621           0.000      0.517     0.399              0.700                    0.000
B5_kto_b10_negonly_Anf_dist        0.040           0.001      0.012     0.037              0.188                      NaN
PSEUDO_nf_dist                     0.041           0.006      0.011     0.039              0.181                      NaN
B0_frozen                          0.147           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_fresh_b25` (0.127)
- vs `B5_kto_b10_negonly_Anf_dist` (0.040): +0.0864 +- 0.0048, lower in 0/15 -> **worse**

### seeds 5-9, n_forget=2 (n_neg 1/4/16)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.247           0.082      0.170     0.134              0.364                    0.237
SDPO_pin_b1                        0.579           0.045      0.483     0.382              0.758                    0.237
SDPO_fresh_b25                     0.251           0.926      0.008     0.026              0.028                    0.237
SDPO_pin_b25                       0.200           0.325      0.123     0.079              0.343                    0.237
SDPO_nearnf_pin_b1                 0.640           0.000      0.543     0.445              0.750                    0.000
B5_kto_b10_negonly_Anf_dist        0.041           0.001      0.018     0.035              0.202                      NaN
PSEUDO_nf_dist                     0.040           0.005      0.014     0.037              0.200                      NaN
B0_frozen                          0.269           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_pin_b25` (0.200)
- vs `B5_kto_b10_negonly_Anf_dist` (0.041): +0.1590 +- 0.0150, lower in 0/15 -> **worse**

### seeds 5-9, n_forget=4 (n_neg 1/4/16)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.373           0.577      0.025     0.091              0.093                    0.511
SDPO_pin_b1                        0.528           0.116      0.335     0.399              0.707                    0.511
SDPO_fresh_b25                     0.508           0.987      0.002     0.026              0.004                    0.511
SDPO_pin_b25                       0.398           0.644      0.091     0.083              0.226                    0.511
SDPO_nearnf_pin_b1                 0.512           0.011      0.465     0.254              0.609                    0.000
B5_kto_b10_negonly_Anf_dist        0.042           0.002      0.032     0.031              0.255                      NaN
PSEUDO_nf_dist                     0.041           0.005      0.022     0.034              0.243                      NaN
B0_frozen                          0.514           1.000      0.000     0.025              0.000                      NaN

best SDPO: `SDPO_fresh_b1` (0.373)
- vs `B5_kto_b10_negonly_Anf_dist` (0.042): +0.3311 +- 0.0119, lower in 0/15 -> **worse**

### overlapping modes std0.45 (seeds 0-9, n_forget 1/2/4 pooled)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.295           0.302      0.151     0.082              0.314                    0.299
SDPO_pin_b1                        0.504           0.139      0.338     0.393              0.697                    0.299
SDPO_fresh_b25                     0.285           0.926      0.010     0.007              0.028                    0.299
SDPO_pin_b25                       0.244           0.526      0.108     0.017              0.243                    0.299
SDPO_nearnf_pin_b1                 0.626           0.052      0.525     0.430              0.780                    0.022
B3_npo_b100_negonly                0.264           0.780      0.031     0.015              0.078                      NaN
B3_npo_b10_negonly                 0.357           0.257      0.200     0.128              0.385                      NaN
B5_kto_b100_negonly                0.274           0.253      0.138     0.126              0.311                      NaN
B5_kto_b10_negonly                 0.597           0.003      0.422     0.489              0.768                      NaN
R_ours_neg                         0.495           0.403      0.311     0.278              0.670                      NaN
R_near_nf_nhood_neg                0.279           0.256      0.253     0.006              0.386                      NaN
R_near_nf_log_neg                  0.272           0.230      0.228     0.044              0.393                      NaN
B5_kto_b10_negonly_Anf_dist        0.200           0.008      0.198     0.008              0.379                      NaN
PSEUDO_nf_dist                     0.200           0.013      0.199     0.008              0.379                      NaN
B0_frozen                          0.296           1.000      0.000     0.007              0.000                      NaN

best SDPO: `SDPO_pin_b25` (0.244)
- vs `B3_npo_b100_negonly` (0.264): -0.0161 +- 0.0088, lower in 28/45 -> **tie**
- vs `R_ours_neg` (0.495): -0.2470 +- 0.0349, lower in 38/45 -> **better**
- vs `R_near_nf_nhood_neg` (0.279): -0.0304 +- 0.0246, lower in 24/45 -> **tie**
- vs `B5_kto_b10_negonly_Anf_dist` (0.200): +0.0439 +- 0.0134, lower in 30/90 -> **worse**

### overlapping modes std0.6 (seeds 0-9, n_forget 1/2/4 pooled)

                             tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                      
SDPO_fresh_b1                      0.280           0.323      0.145     0.064              0.295                    0.302
SDPO_pin_b1                        0.546           0.159      0.344     0.438              0.749                    0.302
SDPO_fresh_b25                     0.283           0.930      0.010     0.004              0.025                    0.302
SDPO_pin_b25                       0.244           0.518      0.116     0.011              0.254                    0.302
SDPO_nearnf_pin_b1                 0.605           0.043      0.557     0.240              0.794                    0.016
B3_npo_b100_negonly                0.264           0.789      0.036     0.008              0.084                      NaN
B3_npo_b10_negonly                 0.338           0.307      0.194     0.085              0.369                      NaN
B5_kto_b100_negonly                0.261           0.304      0.147     0.063              0.278                      NaN
B5_kto_b10_negonly                 0.611           0.013      0.438     0.489              0.795                      NaN
R_ours_neg                         0.462           0.396      0.311     0.232              0.654                      NaN
R_near_nf_nhood_neg                0.434           0.139      0.422     0.006              0.565                      NaN
R_near_nf_log_neg                  0.372           0.247      0.338     0.033              0.529                      NaN
B5_kto_b10_negonly_Anf_dist        0.348           0.004      0.349     0.006              0.519                      NaN
PSEUDO_nf_dist                     0.348           0.006      0.349     0.006              0.518                      NaN
B0_frozen                          0.293           1.000      0.000     0.004              0.000                      NaN

best SDPO: `SDPO_pin_b25` (0.244)
- vs `B3_npo_b100_negonly` (0.264): -0.0063 +- 0.0089, lower in 24/45 -> **tie**
- vs `R_ours_neg` (0.462): -0.2052 +- 0.0325, lower in 37/45 -> **better**
- vs `R_near_nf_nhood_neg` (0.434): -0.1771 +- 0.0196, lower in 42/45 -> **better**
- vs `B5_kto_b10_negonly_Anf_dist` (0.348): -0.1042 +- 0.0143, lower in 75/90 -> **better**

## Conditional toy

### seeds 0-4: random, n_forget=1

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.430           0.024      0.358     0.219              0.715                    0.130
SDPO_fresh_b10                0.129           0.722      0.047     0.031              0.144                    0.130
SDPO_fresh_b25                0.135           0.876      0.021     0.027              0.065                    0.130
SDPO_fresh_b100               0.143           0.953      0.008     0.027              0.026                    0.130
SDPO_pin_b1                   0.486           0.133      0.265     0.433              0.764                    0.130
SDPO_pin_b25                  0.185           0.407      0.109     0.116              0.337                    0.130
SDPO_hop_pin_b1               0.529           0.182      0.362     0.458              0.648                    0.081
SDPO_hop_pin_b25              0.164           0.483      0.091     0.067              0.217                    0.081
SDPO_hop_pin_bH1              0.533           0.187      0.373     0.462              0.655                    0.081
SDPO_hop_pin_bH3              0.402           0.213      0.263     0.313              0.544                    0.081
SDPO_hop_pin_bH10             0.226           0.322      0.144     0.123              0.339                    0.081
CTRL_hop_GA                   0.627           0.131      0.317     0.602              0.766                    0.081
CTRL_synB1_hop                0.709           1.238      0.692     0.099              0.859                    0.081
R_hopB_bsup_neg_AMdist        0.106           0.248      0.077     0.034              0.285                      NaN
R_hop_bsup_neg                0.188           0.440      0.107     0.084              0.297                      NaN
R_hop_bsup_neg_AL             0.140           0.332      0.065     0.068              0.108                      NaN
R_ours_neg                    0.511           0.123      0.230     0.472              0.790                      NaN
NPO_b100_neg                  0.143           0.737      0.031     0.053              0.112                      NaN
NPO_b10_neg                   0.202           0.147      0.155     0.137              0.463                      NaN
KTO_b10_neg                   0.270           0.042      0.220     0.199              0.493                      NaN
NPO_b10_neg_AMdist            0.115           0.282      0.085     0.042              0.280                      NaN
PSEUDO_vote_dist              0.116           0.315      0.086     0.038              0.279                      NaN
B0_frozen                     0.149           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_fresh_b10` (0.129)
- vs `NPO_b100_neg` (0.143): -0.0138 +- 0.0057, lower in 17/20 -> **better**
- vs `R_ours_neg` (0.511): -0.3823 +- 0.0419, lower in 20/20 -> **better**
- vs `R_hop_bsup_neg` (0.188): -0.0587 +- 0.0160, lower in 14/20 -> **better**
- vs `R_hopB_bsup_neg_AMdist` (0.106): +0.0227 +- 0.0067, lower in 6/20 -> **worse**

### seeds 0-4: random, n_forget=2

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.344           0.202      0.173     0.285              0.523                    0.256
SDPO_fresh_b10                0.252           0.884      0.028     0.038              0.095                    0.256
SDPO_fresh_b25                0.260           0.953      0.011     0.029              0.039                    0.256
SDPO_fresh_b100               0.266           0.985      0.004     0.027              0.014                    0.256
SDPO_pin_b1                   0.532           0.153      0.274     0.430              0.744                    0.256
SDPO_pin_b25                  0.257           0.597      0.118     0.097              0.293                    0.256
SDPO_hop_pin_b1               0.653           0.239      0.351     0.561              0.769                    0.099
SDPO_hop_pin_b25              0.248           0.579      0.118     0.082              0.264                    0.099
SDPO_hop_pin_bH1              0.690           0.245      0.369     0.598              0.790                    0.099
SDPO_hop_pin_bH3              0.539           0.309      0.303     0.410              0.676                    0.099
SDPO_hop_pin_bH10             0.343           0.423      0.199     0.183              0.438                    0.099
CTRL_hop_GA                   0.663           0.122      0.356     0.628              0.776                    0.099
CTRL_synB1_hop                0.560           0.552      0.536     0.121              0.789                    0.099
R_hopB_bsup_neg_AMdist        0.197           0.341      0.137     0.037              0.360                      NaN
R_hop_bsup_neg                0.256           0.431      0.160     0.078              0.361                      NaN
R_hop_bsup_neg_AL             0.201           0.516      0.070     0.055              0.096                      NaN
R_ours_neg                    0.567           0.297      0.272     0.454              0.824                      NaN
NPO_b100_neg                  0.255           0.829      0.035     0.054              0.097                      NaN
NPO_b10_neg                   0.387           0.210      0.197     0.307              0.621                      NaN
KTO_b10_neg                   0.519           0.083      0.365     0.420              0.721                      NaN
NPO_b10_neg_AMdist            0.217           0.356      0.151     0.044              0.347                      NaN
PSEUDO_vote_dist              0.219           0.378      0.154     0.040              0.356                      NaN
B0_frozen                     0.270           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.248)
- vs `NPO_b100_neg` (0.255): -0.0065 +- 0.0153, lower in 14/20 -> **tie**
- vs `R_ours_neg` (0.567): -0.3190 +- 0.0265, lower in 20/20 -> **better**
- vs `R_hop_bsup_neg` (0.256): -0.0074 +- 0.0248, lower in 11/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.197): +0.0511 +- 0.0126, lower in 3/20 -> **worse**

### seeds 0-4: random, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.544           0.568      0.075     0.267              0.403                    0.493
SDPO_fresh_b10                0.506           0.964      0.009     0.036              0.052                    0.493
SDPO_fresh_b25                0.511           0.987      0.004     0.029              0.020                    0.493
SDPO_fresh_b100               0.514           0.996      0.002     0.028              0.007                    0.493
SDPO_pin_b1                   0.622           0.289      0.229     0.481              0.628                    0.493
SDPO_pin_b25                  0.530           0.851      0.073     0.114              0.205                    0.493
SDPO_hop_pin_b1               0.674           0.262      0.322     0.537              0.751                    0.409
SDPO_hop_pin_b25              0.460           0.597      0.114     0.169              0.311                    0.409
SDPO_hop_pin_bH1              0.704           0.251      0.337     0.572              0.774                    0.409
SDPO_hop_pin_bH3              0.595           0.287      0.279     0.437              0.678                    0.409
SDPO_hop_pin_bH10             0.493           0.374      0.204     0.307              0.513                    0.409
CTRL_hop_GA                   0.716           0.198      0.237     0.619              0.729                    0.409
CTRL_synB1_hop                0.479           0.700      0.259     0.118              0.628                    0.409
R_hopB_bsup_neg_AMdist        0.318           0.514      0.112     0.042              0.298                      NaN
R_hop_bsup_neg                0.355           0.611      0.132     0.052              0.346                      NaN
R_hop_bsup_neg_AL             0.401           0.727      0.046     0.046              0.080                      NaN
R_ours_neg                    0.611           0.554      0.173     0.341              0.670                      NaN
NPO_b100_neg                  0.516           0.939      0.011     0.058              0.054                      NaN
NPO_b10_neg                   0.608           0.554      0.072     0.338              0.449                      NaN
KTO_b10_neg                   0.915           0.153      0.393     0.840              0.926                      NaN
NPO_b10_neg_AMdist            0.368           0.598      0.118     0.053              0.214                      NaN
PSEUDO_vote_dist              0.360           0.587      0.136     0.048              0.289                      NaN
B0_frozen                     0.516           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.460)
- vs `NPO_b100_neg` (0.516): -0.0561 +- 0.0115, lower in 17/20 -> **better**
- vs `R_ours_neg` (0.611): -0.1508 +- 0.0338, lower in 16/20 -> **better**
- vs `R_hop_bsup_neg` (0.355): +0.1049 +- 0.0289, lower in 3/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.318): +0.1426 +- 0.0251, lower in 1/20 -> **worse**

### seeds 0-4: contig, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.434           0.452      0.070     0.215              0.357                    0.498
SDPO_fresh_b10                0.493           0.947      0.008     0.033              0.045                    0.498
SDPO_fresh_b25                0.505           0.979      0.004     0.029              0.019                    0.498
SDPO_fresh_b100               0.510           0.994      0.002     0.028              0.006                    0.498
SDPO_pin_b1                   0.567           0.185      0.295     0.429              0.773                    0.498
SDPO_pin_b25                  0.431           0.729      0.067     0.077              0.224                    0.498
SDPO_hop_pin_b1               0.579           0.283      0.364     0.416              0.721                    0.763
SDPO_hop_pin_b25              0.435           0.674      0.117     0.108              0.275                    0.763
SDPO_hop_pin_bH1              0.599           0.294      0.363     0.430              0.723                    0.763
SDPO_hop_pin_bH3              0.523           0.270      0.307     0.347              0.663                    0.763
SDPO_hop_pin_bH10             0.453           0.402      0.212     0.237              0.467                    0.763
CTRL_hop_GA                   0.756           0.241      0.263     0.639              0.778                    0.763
CTRL_synB1_hop                0.810           1.427      0.522     0.116              0.904                    0.763
R_hopB_bsup_neg_AMdist        0.203           0.330      0.055     0.036              0.309                      NaN
R_hop_bsup_neg                0.471           0.819      0.108     0.073              0.329                      NaN
R_hop_bsup_neg_AL             0.479           0.874      0.042     0.054              0.079                      NaN
R_ours_neg                    0.590           0.356      0.233     0.411              0.842                      NaN
NPO_b100_neg                  0.488           0.903      0.026     0.049              0.080                      NaN
NPO_b10_neg                   0.431           0.449      0.129     0.213              0.505                      NaN
KTO_b10_neg                   0.594           0.050      0.342     0.523              0.776                      NaN
NPO_b10_neg_AMdist            0.210           0.314      0.065     0.046              0.295                      NaN
PSEUDO_vote_dist              0.215           0.338      0.067     0.040              0.297                      NaN
B0_frozen                     0.513           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_pin_b25` (0.431)
- vs `NPO_b100_neg` (0.488): -0.0566 +- 0.0082, lower in 20/20 -> **better**
- vs `R_ours_neg` (0.590): -0.1589 +- 0.0373, lower in 16/20 -> **better**
- vs `R_hop_bsup_neg` (0.471): -0.0400 +- 0.0244, lower in 10/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.203): +0.2278 +- 0.0290, lower in 2/20 -> **worse**

### seeds 0-4: alt, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.550           0.591      0.076     0.262              0.375                     0.49
SDPO_fresh_b10                0.511           0.966      0.006     0.041              0.060                     0.49
SDPO_fresh_b25                0.512           0.987      0.003     0.031              0.023                     0.49
SDPO_fresh_b100               0.513           0.996      0.001     0.028              0.008                     0.49
SDPO_pin_b1                   0.705           0.296      0.204     0.560              0.649                     0.49
SDPO_pin_b25                  0.534           0.909      0.064     0.092              0.186                     0.49
SDPO_hop_pin_b1               0.715           0.251      0.439     0.579              0.801                     0.11
SDPO_hop_pin_b25              0.480           0.633      0.120     0.172              0.314                     0.11
SDPO_hop_pin_bH1              0.749           0.247      0.459     0.610              0.831                     0.11
SDPO_hop_pin_bH3              0.625           0.298      0.364     0.463              0.698                     0.11
SDPO_hop_pin_bH10             0.538           0.427      0.214     0.325              0.519                     0.11
CTRL_hop_GA                   0.641           0.087      0.248     0.598              0.649                     0.11
CTRL_synB1_hop                0.317           0.286      0.182     0.148              0.524                     0.11
R_hopB_bsup_neg_AMdist        0.457           0.732      0.218     0.066              0.386                      NaN
R_hop_bsup_neg                0.261           0.422      0.118     0.049              0.335                      NaN
R_hop_bsup_neg_AL             0.347           0.630      0.031     0.041              0.086                      NaN
R_ours_neg                    0.641           0.599      0.145     0.349              0.654                      NaN
NPO_b100_neg                  0.519           0.944      0.010     0.059              0.055                      NaN
NPO_b10_neg                   0.598           0.582      0.077     0.315              0.404                      NaN
KTO_b10_neg                   0.898           0.201      0.426     0.800              0.919                      NaN
NPO_b10_neg_AMdist            0.461           0.760      0.134     0.088              0.230                      NaN
PSEUDO_vote_dist              0.494           0.788      0.238     0.075              0.387                      NaN
B0_frozen                     0.514           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.480)
- vs `NPO_b100_neg` (0.519): -0.0380 +- 0.0153, lower in 16/20 -> **better**
- vs `R_ours_neg` (0.641): -0.1607 +- 0.0234, lower in 18/20 -> **better**
- vs `R_hop_bsup_neg` (0.261): +0.2195 +- 0.0360, lower in 2/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.457): +0.0234 +- 0.0206, lower in 7/20 -> **tie**

### seeds 0-4: pairs, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.528           0.578      0.077     0.246              0.392                    0.447
SDPO_fresh_b10                0.503           0.957      0.010     0.036              0.064                    0.447
SDPO_fresh_b25                0.510           0.984      0.004     0.030              0.025                    0.447
SDPO_fresh_b100               0.514           0.995      0.002     0.028              0.008                    0.447
SDPO_pin_b1                   0.571           0.221      0.180     0.463              0.576                    0.447
SDPO_pin_b25                  0.486           0.805      0.057     0.092              0.144                    0.447
SDPO_hop_pin_b1               0.597           0.150      0.252     0.518              0.721                    0.469
SDPO_hop_pin_b25              0.425           0.558      0.098     0.151              0.283                    0.469
SDPO_hop_pin_bH1              0.639           0.140      0.270     0.567              0.751                    0.469
SDPO_hop_pin_bH3              0.503           0.207      0.225     0.385              0.598                    0.469
SDPO_hop_pin_bH10             0.440           0.302      0.165     0.285              0.464                    0.469
CTRL_hop_GA                   0.766           0.181      0.217     0.678              0.766                    0.469
CTRL_synB1_hop                0.573           0.893      0.189     0.134              0.638                    0.469
R_hopB_bsup_neg_AMdist        0.246           0.422      0.040     0.039              0.260                      NaN
R_hop_bsup_neg                0.345           0.580      0.108     0.061              0.311                      NaN
R_hop_bsup_neg_AL             0.425           0.764      0.029     0.052              0.087                      NaN
R_ours_neg                    0.525           0.400      0.141     0.330              0.637                      NaN
NPO_b100_neg                  0.509           0.926      0.011     0.056              0.045                      NaN
NPO_b10_neg                   0.590           0.556      0.090     0.319              0.434                      NaN
KTO_b10_neg                   0.893           0.102      0.421     0.843              0.911                      NaN
NPO_b10_neg_AMdist            0.317           0.548      0.036     0.049              0.146                      NaN
PSEUDO_vote_dist              0.288           0.501      0.046     0.043              0.211                      NaN
B0_frozen                     0.516           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.425)
- vs `NPO_b100_neg` (0.509): -0.0841 +- 0.0135, lower in 19/20 -> **better**
- vs `R_ours_neg` (0.525): -0.1005 +- 0.0444, lower in 11/20 -> **better**
- vs `R_hop_bsup_neg` (0.345): +0.0797 +- 0.0228, lower in 3/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.246): +0.1790 +- 0.0339, lower in 4/20 -> **worse**

### seeds 0-4: b3p1, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.505           0.531      0.089     0.246              0.366                    0.501
SDPO_fresh_b10                0.504           0.958      0.009     0.037              0.062                    0.501
SDPO_fresh_b25                0.510           0.985      0.004     0.029              0.024                    0.501
SDPO_fresh_b100               0.513           0.996      0.002     0.028              0.008                    0.501
SDPO_pin_b1                   0.581           0.215      0.175     0.476              0.631                    0.501
SDPO_pin_b25                  0.466           0.828      0.037     0.062              0.129                    0.501
SDPO_hop_pin_b1               0.645           0.261      0.294     0.499              0.737                    0.517
SDPO_hop_pin_b25              0.440           0.630      0.097     0.133              0.247                    0.517
SDPO_hop_pin_bH1              0.668           0.235      0.318     0.549              0.768                    0.517
SDPO_hop_pin_bH3              0.578           0.308      0.271     0.411              0.643                    0.517
SDPO_hop_pin_bH10             0.464           0.403      0.170     0.262              0.435                    0.517
CTRL_hop_GA                   0.769           0.179      0.287     0.682              0.800                    0.517
CTRL_synB1_hop                0.621           0.955      0.253     0.139              0.716                    0.517
R_hopB_bsup_neg_AMdist        0.251           0.421      0.055     0.037              0.260                      NaN
R_hop_bsup_neg                0.376           0.643      0.120     0.061              0.364                      NaN
R_hop_bsup_neg_AL             0.439           0.790      0.024     0.054              0.093                      NaN
R_ours_neg                    0.573           0.441      0.156     0.358              0.681                      NaN
NPO_b100_neg                  0.512           0.930      0.011     0.059              0.054                      NaN
NPO_b10_neg                   0.596           0.543      0.072     0.331              0.442                      NaN
KTO_b10_neg                   0.900           0.086      0.471     0.858              0.920                      NaN
NPO_b10_neg_AMdist            0.300           0.492      0.063     0.058              0.171                      NaN
PSEUDO_vote_dist              0.280           0.472      0.052     0.044              0.235                      NaN
B0_frozen                     0.515           1.000      0.000     0.027              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.440)
- vs `NPO_b100_neg` (0.512): -0.0724 +- 0.0107, lower in 18/20 -> **better**
- vs `R_ours_neg` (0.573): -0.1332 +- 0.0440, lower in 13/20 -> **better**
- vs `R_hop_bsup_neg` (0.376): +0.0634 +- 0.0296, lower in 4/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.251): +0.1883 +- 0.0293, lower in 2/20 -> **worse**

### seeds 5-9: random, n_forget=1

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.477           0.023      0.368     0.310              0.716                    0.102
SDPO_fresh_b10                0.141           0.624      0.064     0.039              0.181                    0.102
SDPO_fresh_b25                0.133           0.832      0.026     0.029              0.080                    0.102
SDPO_fresh_b100               0.140           0.920      0.011     0.029              0.037                    0.102
SDPO_pin_b1                   0.479           0.174      0.278     0.421              0.748                    0.102
SDPO_pin_b25                  0.157           0.462      0.094     0.070              0.270                    0.102
SDPO_hop_pin_b1               0.604           0.068      0.370     0.545              0.736                    0.116
SDPO_hop_pin_b25              0.157           0.403      0.103     0.052              0.233                    0.116
SDPO_hop_pin_bH1              0.643           0.065      0.383     0.601              0.767                    0.116
SDPO_hop_pin_bH3              0.456           0.124      0.317     0.358              0.618                    0.116
SDPO_hop_pin_bH10             0.263           0.226      0.184     0.150              0.395                    0.116
CTRL_hop_GA                   0.640           0.130      0.322     0.624              0.752                    0.116
CTRL_synB1_hop                0.712           1.016      0.702     0.085              0.853                    0.116
R_hopB_bsup_neg_AMdist        0.107           0.292      0.073     0.036              0.265                      NaN
R_hop_bsup_neg                0.187           0.342      0.121     0.086              0.300                      NaN
R_hop_bsup_neg_AL             0.142           0.337      0.063     0.071              0.109                      NaN
R_ours_neg                    0.524           0.113      0.233     0.496              0.792                      NaN
NPO_b100_neg                  0.143           0.728      0.035     0.055              0.112                      NaN
NPO_b10_neg_AMdist            0.113           0.313      0.080     0.042              0.259                      NaN
PSEUDO_vote_dist              0.113           0.344      0.080     0.038              0.258                      NaN
B0_frozen                     0.149           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_fresh_b25` (0.133)
- vs `NPO_b100_neg` (0.143): -0.0099 +- 0.0026, lower in 14/20 -> **better**
- vs `R_ours_neg` (0.524): -0.3912 +- 0.0283, lower in 20/20 -> **better**
- vs `R_hop_bsup_neg` (0.187): -0.0543 +- 0.0125, lower in 17/20 -> **better**
- vs `R_hopB_bsup_neg_AMdist` (0.107): +0.0256 +- 0.0059, lower in 3/20 -> **worse**

### seeds 5-9: random, n_forget=2

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.357           0.218      0.158     0.271              0.539                    0.268
SDPO_fresh_b10                0.256           0.861      0.033     0.047              0.116                    0.268
SDPO_fresh_b25                0.260           0.949      0.011     0.030              0.043                    0.268
SDPO_fresh_b100               0.267           0.986      0.004     0.028              0.015                    0.268
SDPO_pin_b1                   0.524           0.230      0.270     0.419              0.724                    0.268
SDPO_pin_b25                  0.236           0.588      0.089     0.086              0.243                    0.268
SDPO_hop_pin_b1               0.644           0.121      0.369     0.596              0.798                    0.218
SDPO_hop_pin_b25              0.231           0.510      0.104     0.081              0.259                    0.218
SDPO_hop_pin_bH1              0.672           0.117      0.387     0.625              0.823                    0.218
SDPO_hop_pin_bH3              0.511           0.156      0.303     0.434              0.679                    0.218
SDPO_hop_pin_bH10             0.340           0.302      0.175     0.240              0.479                    0.218
CTRL_hop_GA                   0.645           0.145      0.325     0.603              0.751                    0.218
CTRL_synB1_hop                0.558           0.914      0.533     0.127              0.809                    0.218
R_hopB_bsup_neg_AMdist        0.166           0.325      0.109     0.036              0.316                      NaN
R_hop_bsup_neg                0.226           0.424      0.125     0.070              0.321                      NaN
R_hop_bsup_neg_AL             0.211           0.598      0.062     0.060              0.100                      NaN
R_ours_neg                    0.578           0.207      0.232     0.500              0.850                      NaN
NPO_b100_neg                  0.256           0.832      0.031     0.054              0.095                      NaN
NPO_b10_neg_AMdist            0.177           0.357      0.113     0.044              0.296                      NaN
PSEUDO_vote_dist              0.178           0.365      0.116     0.041              0.305                      NaN
B0_frozen                     0.270           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.231)
- vs `NPO_b100_neg` (0.256): -0.0251 +- 0.0091, lower in 14/20 -> **better**
- vs `R_ours_neg` (0.578): -0.3471 +- 0.0297, lower in 20/20 -> **better**
- vs `R_hop_bsup_neg` (0.226): +0.0047 +- 0.0219, lower in 9/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.166): +0.0644 +- 0.0127, lower in 1/20 -> **worse**

### seeds 5-9: random, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.470           0.501      0.065     0.226              0.380                    0.495
SDPO_fresh_b10                0.501           0.951      0.011     0.038              0.055                    0.495
SDPO_fresh_b25                0.508           0.983      0.004     0.030              0.020                    0.495
SDPO_fresh_b100               0.513           0.995      0.002     0.028              0.007                    0.495
SDPO_pin_b1                   0.541           0.254      0.240     0.387              0.647                    0.495
SDPO_pin_b25                  0.451           0.799      0.053     0.062              0.156                    0.495
SDPO_hop_pin_b1               0.626           0.171      0.291     0.536              0.765                    0.570
SDPO_hop_pin_b25              0.412           0.653      0.100     0.094              0.245                    0.570
SDPO_hop_pin_bH1              0.660           0.172      0.289     0.572              0.798                    0.570
SDPO_hop_pin_bH3              0.554           0.228      0.276     0.430              0.701                    0.570
SDPO_hop_pin_bH10             0.447           0.364      0.176     0.259              0.470                    0.570
CTRL_hop_GA                   0.751           0.248      0.270     0.630              0.755                    0.570
CTRL_synB1_hop                0.649           1.112      0.405     0.101              0.792                    0.570
R_hopB_bsup_neg_AMdist        0.265           0.417      0.093     0.037              0.319                      NaN
R_hop_bsup_neg                0.398           0.689      0.130     0.063              0.343                      NaN
R_hop_bsup_neg_AL             0.446           0.825      0.048     0.045              0.084                      NaN
R_ours_neg                    0.593           0.391      0.223     0.377              0.756                      NaN
NPO_b100_neg                  0.500           0.922      0.018     0.052              0.059                      NaN
NPO_b10_neg_AMdist            0.284           0.431      0.104     0.048              0.294                      NaN
PSEUDO_vote_dist              0.283           0.450      0.100     0.042              0.307                      NaN
B0_frozen                     0.515           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.412)
- vs `NPO_b100_neg` (0.500): -0.0887 +- 0.0125, lower in 18/20 -> **better**
- vs `R_ours_neg` (0.593): -0.1810 +- 0.0327, lower in 18/20 -> **better**
- vs `R_hop_bsup_neg` (0.398): +0.0136 +- 0.0183, lower in 8/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.265): +0.1469 +- 0.0224, lower in 2/20 -> **worse**

### seeds 5-9: contig, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.430           0.412      0.072     0.230              0.409                    0.516
SDPO_fresh_b10                0.492           0.938      0.009     0.036              0.052                    0.516
SDPO_fresh_b25                0.504           0.979      0.003     0.029              0.019                    0.516
SDPO_fresh_b100               0.510           0.994      0.002     0.027              0.006                    0.516
SDPO_pin_b1                   0.594           0.245      0.274     0.445              0.791                    0.516
SDPO_pin_b25                  0.454           0.771      0.066     0.080              0.188                    0.516
SDPO_hop_pin_b1               0.662           0.277      0.240     0.522              0.786                    0.747
SDPO_hop_pin_b25              0.455           0.618      0.100     0.155              0.290                    0.747
SDPO_hop_pin_bH1              0.700           0.239      0.220     0.577              0.817                    0.747
SDPO_hop_pin_bH3              0.502           0.311      0.204     0.336              0.612                    0.747
SDPO_hop_pin_bH10             0.464           0.373      0.154     0.282              0.447                    0.747
CTRL_hop_GA                   0.758           0.206      0.319     0.658              0.792                    0.747
CTRL_synB1_hop                0.817           1.450      0.528     0.109              0.909                    0.747
R_hopB_bsup_neg_AMdist        0.228           0.375      0.070     0.034              0.304                      NaN
R_hop_bsup_neg                0.455           0.805      0.103     0.064              0.349                      NaN
R_hop_bsup_neg_AL             0.483           0.887      0.043     0.052              0.084                      NaN
NPO_b100_neg                  0.493           0.913      0.018     0.049              0.067                      NaN
NPO_b10_neg_AMdist            0.230           0.352      0.074     0.049              0.296                      NaN
PSEUDO_vote_dist              0.236           0.382      0.076     0.040              0.294                      NaN
B0_frozen                     0.513           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_fresh_b1` (0.430)
- vs `NPO_b100_neg` (0.493): -0.0625 +- 0.0324, lower in 16/20 -> **tie**
- vs `R_hop_bsup_neg` (0.455): -0.0243 +- 0.0416, lower in 16/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.228): +0.2025 +- 0.0303, lower in 1/20 -> **worse**

### seeds 5-9: alt, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.554           0.615      0.084     0.254              0.382                    0.479
SDPO_fresh_b10                0.513           0.961      0.010     0.044              0.064                    0.479
SDPO_fresh_b25                0.512           0.987      0.003     0.030              0.022                    0.479
SDPO_fresh_b100               0.514           0.996      0.002     0.028              0.007                    0.479
SDPO_pin_b1                   0.676           0.348      0.197     0.503              0.607                    0.479
SDPO_pin_b25                  0.537           0.897      0.064     0.099              0.180                    0.479
SDPO_hop_pin_b1               0.675           0.257      0.334     0.539              0.759                    0.064
SDPO_hop_pin_b25              0.480           0.655      0.105     0.160              0.293                    0.064
SDPO_hop_pin_bH1              0.717           0.257      0.356     0.584              0.793                    0.064
SDPO_hop_pin_bH3              0.609           0.325      0.261     0.447              0.675                    0.064
SDPO_hop_pin_bH10             0.524           0.445      0.172     0.300              0.495                    0.064
CTRL_hop_GA                   0.657           0.098      0.210     0.609              0.662                    0.064
CTRL_synB1_hop                0.284           0.233      0.178     0.125              0.512                    0.064
R_hopB_bsup_neg_AMdist        0.450           0.722      0.203     0.064              0.358                      NaN
R_hop_bsup_neg                0.262           0.413      0.118     0.053              0.346                      NaN
R_hop_bsup_neg_AL             0.352           0.637      0.030     0.041              0.085                      NaN
NPO_b100_neg                  0.519           0.943      0.012     0.059              0.056                      NaN
NPO_b10_neg_AMdist            0.457           0.752      0.126     0.081              0.211                      NaN
PSEUDO_vote_dist              0.487           0.782      0.228     0.071              0.361                      NaN
B0_frozen                     0.516           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.480)
- vs `NPO_b100_neg` (0.519): -0.0395 +- 0.0135, lower in 16/20 -> **better**
- vs `R_hop_bsup_neg` (0.262): +0.2177 +- 0.0361, lower in 1/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.450): +0.0303 +- 0.0151, lower in 10/20 -> **worse**

### seeds 5-9: pairs, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.519           0.576      0.051     0.240              0.369                    0.503
SDPO_fresh_b10                0.503           0.955      0.008     0.040              0.060                    0.503
SDPO_fresh_b25                0.508           0.986      0.003     0.029              0.022                    0.503
SDPO_fresh_b100               0.511           0.996      0.001     0.028              0.007                    0.503
SDPO_pin_b1                   0.520           0.256      0.140     0.393              0.495                    0.503
SDPO_pin_b25                  0.470           0.760      0.043     0.101              0.147                    0.503
SDPO_hop_pin_b1               0.514           0.142      0.231     0.418              0.656                    0.510
SDPO_hop_pin_b25              0.420           0.515      0.087     0.170              0.296                    0.510
SDPO_hop_pin_bH1              0.545           0.139      0.262     0.455              0.681                    0.510
SDPO_hop_pin_bH3              0.473           0.167      0.216     0.371              0.601                    0.510
SDPO_hop_pin_bH10             0.427           0.261      0.140     0.295              0.475                    0.510
CTRL_hop_GA                   0.765           0.243      0.229     0.648              0.755                    0.510
CTRL_synB1_hop                0.551           0.884      0.208     0.118              0.641                    0.510
R_hopB_bsup_neg_AMdist        0.251           0.423      0.048     0.040              0.276                      NaN
R_hop_bsup_neg                0.350           0.606      0.096     0.057              0.338                      NaN
R_hop_bsup_neg_AL             0.430           0.783      0.027     0.050              0.082                      NaN
NPO_b100_neg                  0.504           0.918      0.011     0.059              0.056                      NaN
NPO_b10_neg_AMdist            0.311           0.531      0.035     0.054              0.152                      NaN
PSEUDO_vote_dist              0.288           0.495      0.049     0.045              0.217                      NaN
B0_frozen                     0.513           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.420)
- vs `NPO_b100_neg` (0.504): -0.0843 +- 0.0158, lower in 18/20 -> **better**
- vs `R_hop_bsup_neg` (0.350): +0.0699 +- 0.0252, lower in 6/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.251): +0.1689 +- 0.0366, lower in 4/20 -> **worse**

### seeds 5-9: b3p1, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.502           0.553      0.071     0.233              0.401                    0.482
SDPO_fresh_b10                0.498           0.950      0.010     0.036              0.056                    0.482
SDPO_fresh_b25                0.507           0.982      0.003     0.029              0.020                    0.482
SDPO_fresh_b100               0.511           0.994      0.002     0.028              0.008                    0.482
SDPO_pin_b1                   0.593           0.271      0.174     0.461              0.622                    0.482
SDPO_pin_b25                  0.493           0.775      0.058     0.117              0.183                    0.482
SDPO_hop_pin_b1               0.638           0.254      0.309     0.501              0.739                    0.574
SDPO_hop_pin_b25              0.458           0.568      0.102     0.182              0.315                    0.574
SDPO_hop_pin_bH1              0.655           0.236      0.322     0.528              0.764                    0.574
SDPO_hop_pin_bH3              0.586           0.254      0.271     0.440              0.680                    0.574
SDPO_hop_pin_bH10             0.476           0.365      0.172     0.289              0.482                    0.574
CTRL_hop_GA                   0.754           0.207      0.295     0.653              0.772                    0.574
CTRL_synB1_hop                0.656           1.097      0.238     0.122              0.735                    0.574
R_hopB_bsup_neg_AMdist        0.256           0.433      0.061     0.039              0.276                      NaN
R_hop_bsup_neg                0.407           0.699      0.131     0.066              0.377                      NaN
R_hop_bsup_neg_AL             0.440           0.805      0.026     0.048              0.084                      NaN
NPO_b100_neg                  0.507           0.930      0.011     0.054              0.050                      NaN
NPO_b10_neg_AMdist            0.301           0.502      0.079     0.055              0.186                      NaN
PSEUDO_vote_dist              0.285           0.487      0.063     0.044              0.239                      NaN
B0_frozen                     0.514           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.458)
- vs `NPO_b100_neg` (0.507): -0.0489 +- 0.0156, lower in 17/20 -> **better**
- vs `R_hop_bsup_neg` (0.407): +0.0511 +- 0.0302, lower in 7/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.256): +0.2018 +- 0.0331, lower in 2/20 -> **worse**

### seeds 10-14: random, n_forget=1

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.488           0.018      0.401     0.286              0.771                    0.117
SDPO_fresh_b10                0.139           0.608      0.076     0.037              0.215                    0.117
SDPO_fresh_b25                0.129           0.823      0.031     0.027              0.098                    0.117
SDPO_fresh_b100               0.139           0.926      0.013     0.027              0.041                    0.117
SDPO_pin_b1                   0.542           0.155      0.268     0.513              0.802                    0.117
SDPO_pin_b25                  0.153           0.480      0.088     0.072              0.273                    0.117
SDPO_hop_pin_b1               0.463           0.074      0.290     0.391              0.614                    0.043
SDPO_hop_pin_b25              0.160           0.470      0.086     0.069              0.205                    0.043
SDPO_hop_pin_bH1              0.515           0.071      0.318     0.447              0.652                    0.043
SDPO_hop_pin_bH3              0.347           0.117      0.238     0.257              0.507                    0.043
SDPO_hop_pin_bH10             0.213           0.245      0.144     0.112              0.330                    0.043
CTRL_hop_GA                   0.630           0.071      0.350     0.609              0.765                    0.043
CTRL_synB1_hop                0.680           0.554      0.676     0.093              0.833                    0.043
R_hopB_bsup_neg_AMdist        0.092           0.264      0.063     0.035              0.254                      NaN
R_hop_bsup_neg                0.177           0.286      0.111     0.080              0.290                      NaN
NPO_b10_neg_AMdist            0.102           0.303      0.074     0.041              0.251                      NaN
PSEUDO_vote_dist              0.102           0.336      0.073     0.037              0.247                      NaN
B0_frozen                     0.148           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_fresh_b25` (0.129)
- vs `R_hop_bsup_neg` (0.177): -0.0479 +- 0.0141, lower in 15/20 -> **better**
- vs `R_hopB_bsup_neg_AMdist` (0.092): +0.0366 +- 0.0060, lower in 2/20 -> **worse**

### seeds 10-14: random, n_forget=2

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.410           0.274      0.182     0.313              0.590                    0.242
SDPO_fresh_b10                0.241           0.815      0.040     0.039              0.135                    0.242
SDPO_fresh_b25                0.253           0.926      0.016     0.029              0.055                    0.242
SDPO_fresh_b100               0.263           0.974      0.006     0.027              0.019                    0.242
SDPO_pin_b1                   0.470           0.202      0.266     0.363              0.675                    0.242
SDPO_pin_b25                  0.239           0.643      0.104     0.071              0.243                    0.242
SDPO_hop_pin_b1               0.581           0.110      0.366     0.493              0.743                    0.109
SDPO_hop_pin_b25              0.228           0.497      0.105     0.085              0.267                    0.109
SDPO_hop_pin_bH1              0.591           0.115      0.359     0.519              0.752                    0.109
SDPO_hop_pin_bH3              0.455           0.145      0.275     0.347              0.645                    0.109
SDPO_hop_pin_bH10             0.276           0.274      0.172     0.147              0.432                    0.109
CTRL_hop_GA                   0.658           0.073      0.334     0.633              0.780                    0.109
CTRL_synB1_hop                0.496           0.546      0.471     0.126              0.768                    0.109
R_hopB_bsup_neg_AMdist        0.155           0.323      0.096     0.035              0.287                      NaN
R_hop_bsup_neg                0.188           0.350      0.117     0.053              0.323                      NaN
NPO_b10_neg_AMdist            0.174           0.383      0.106     0.042              0.266                      NaN
PSEUDO_vote_dist              0.174           0.381      0.109     0.039              0.280                      NaN
B0_frozen                     0.270           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.228)
- vs `R_hop_bsup_neg` (0.188): +0.0403 +- 0.0220, lower in 6/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.155): +0.0735 +- 0.0178, lower in 2/20 -> **worse**

### seeds 10-14: random, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.509           0.544      0.076     0.245              0.396                    0.484
SDPO_fresh_b10                0.500           0.952      0.010     0.038              0.065                    0.484
SDPO_fresh_b25                0.506           0.983      0.004     0.029              0.023                    0.484
SDPO_fresh_b100               0.510           0.994      0.002     0.028              0.007                    0.484
SDPO_pin_b1                   0.574           0.265      0.223     0.436              0.607                    0.484
SDPO_pin_b25                  0.474           0.857      0.052     0.058              0.147                    0.484
SDPO_hop_pin_b1               0.663           0.237      0.317     0.532              0.753                    0.511
SDPO_hop_pin_b25              0.459           0.594      0.106     0.171              0.274                    0.511
SDPO_hop_pin_bH1              0.720           0.204      0.341     0.606              0.794                    0.511
SDPO_hop_pin_bH3              0.567           0.265      0.276     0.420              0.650                    0.511
SDPO_hop_pin_bH10             0.493           0.337      0.191     0.320              0.468                    0.511
CTRL_hop_GA                   0.766           0.157      0.301     0.690              0.775                    0.511
CTRL_synB1_hop                0.582           0.944      0.333     0.101              0.710                    0.511
R_hopB_bsup_neg_AMdist        0.281           0.455      0.089     0.038              0.292                      NaN
R_hop_bsup_neg                0.384           0.678      0.140     0.042              0.330                      NaN
NPO_b10_neg_AMdist            0.318           0.512      0.094     0.048              0.233                      NaN
PSEUDO_vote_dist              0.312           0.503      0.101     0.043              0.273                      NaN
B0_frozen                     0.513           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.459)
- vs `R_hop_bsup_neg` (0.384): +0.0749 +- 0.0256, lower in 6/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.281): +0.1776 +- 0.0238, lower in 2/20 -> **worse**

### seeds 10-14: contig, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.434           0.429      0.075     0.225              0.388                    0.489
SDPO_fresh_b10                0.488           0.931      0.009     0.036              0.047                    0.489
SDPO_fresh_b25                0.503           0.977      0.003     0.029              0.017                    0.489
SDPO_fresh_b100               0.510           0.993      0.001     0.028              0.006                    0.489
SDPO_pin_b1                   0.549           0.202      0.278     0.405              0.770                    0.489
SDPO_pin_b25                  0.418           0.759      0.046     0.050              0.190                    0.489
SDPO_hop_pin_b1               0.684           0.322      0.307     0.503              0.759                    0.786
SDPO_hop_pin_b25              0.496           0.635      0.086     0.188              0.291                    0.786
SDPO_hop_pin_bH1              0.697           0.308      0.299     0.521              0.783                    0.786
SDPO_hop_pin_bH3              0.625           0.371      0.213     0.444              0.658                    0.786
SDPO_hop_pin_bH10             0.561           0.424      0.135     0.355              0.496                    0.786
CTRL_hop_GA                   0.797           0.304      0.303     0.650              0.782                    0.786
CTRL_synB1_hop                0.833           1.527      0.543     0.091              0.905                    0.786
R_hopB_bsup_neg_AMdist        0.230           0.389      0.049     0.036              0.285                      NaN
R_hop_bsup_neg                0.484           0.881      0.102     0.056              0.342                      NaN
NPO_b10_neg_AMdist            0.234           0.367      0.054     0.048              0.276                      NaN
PSEUDO_vote_dist              0.239           0.397      0.056     0.039              0.274                      NaN
B0_frozen                     0.513           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_pin_b25` (0.418)
- vs `R_hop_bsup_neg` (0.484): -0.0662 +- 0.0272, lower in 14/20 -> **better**
- vs `R_hopB_bsup_neg_AMdist` (0.230): +0.1880 +- 0.0346, lower in 4/20 -> **worse**

### seeds 10-14: alt, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.524           0.610      0.101     0.226              0.357                    0.479
SDPO_fresh_b10                0.512           0.966      0.012     0.040              0.064                    0.479
SDPO_fresh_b25                0.513           0.989      0.004     0.029              0.022                    0.479
SDPO_fresh_b100               0.515           0.996      0.002     0.028              0.007                    0.479
SDPO_pin_b1                   0.711           0.323      0.187     0.552              0.658                    0.479
SDPO_pin_b25                  0.524           0.916      0.059     0.076              0.164                    0.479
SDPO_hop_pin_b1               0.726           0.281      0.371     0.584              0.800                    0.096
SDPO_hop_pin_b25              0.469           0.657      0.094     0.147              0.275                    0.096
SDPO_hop_pin_bH1              0.756           0.268      0.400     0.622              0.819                    0.096
SDPO_hop_pin_bH3              0.618           0.339      0.316     0.444              0.689                    0.096
SDPO_hop_pin_bH10             0.509           0.406      0.174     0.309              0.480                    0.096
CTRL_hop_GA                   0.673           0.085      0.254     0.630              0.673                    0.096
CTRL_synB1_hop                0.305           0.290      0.167     0.121              0.512                    0.096
R_hopB_bsup_neg_AMdist        0.450           0.733      0.217     0.069              0.360                      NaN
R_hop_bsup_neg                0.272           0.454      0.113     0.039              0.315                      NaN
NPO_b10_neg_AMdist            0.470           0.769      0.147     0.089              0.228                      NaN
PSEUDO_vote_dist              0.493           0.803      0.242     0.077              0.377                      NaN
B0_frozen                     0.517           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.469)
- vs `R_hop_bsup_neg` (0.272): +0.1964 +- 0.0303, lower in 3/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.450): +0.0190 +- 0.0173, lower in 8/20 -> **tie**

### seeds 10-14: pairs, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.550           0.552      0.091     0.282              0.396                    0.494
SDPO_fresh_b10                0.508           0.947      0.010     0.048              0.083                    0.494
SDPO_fresh_b25                0.508           0.984      0.004     0.030              0.026                    0.494
SDPO_fresh_b100               0.512           0.995      0.002     0.028              0.009                    0.494
SDPO_pin_b1                   0.597           0.238      0.170     0.482              0.602                    0.494
SDPO_pin_b25                  0.466           0.857      0.051     0.050              0.114                    0.494
SDPO_hop_pin_b1               0.604           0.165      0.290     0.506              0.723                    0.525
SDPO_hop_pin_b25              0.422           0.574      0.072     0.143              0.244                    0.525
SDPO_hop_pin_bH1              0.651           0.155      0.276     0.572              0.769                    0.525
SDPO_hop_pin_bH3              0.505           0.201      0.245     0.387              0.616                    0.525
SDPO_hop_pin_bH10             0.447           0.303      0.143     0.293              0.446                    0.525
CTRL_hop_GA                   0.764           0.227      0.252     0.654              0.769                    0.525
CTRL_synB1_hop                0.588           0.957      0.227     0.105              0.654                    0.525
R_hopB_bsup_neg_AMdist        0.264           0.460      0.043     0.039              0.258                      NaN
R_hop_bsup_neg                0.370           0.648      0.125     0.052              0.366                      NaN
NPO_b10_neg_AMdist            0.331           0.574      0.036     0.052              0.143                      NaN
PSEUDO_vote_dist              0.306           0.534      0.045     0.046              0.208                      NaN
B0_frozen                     0.514           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.422)
- vs `R_hop_bsup_neg` (0.370): +0.0523 +- 0.0195, lower in 5/20 -> **worse**
- vs `R_hopB_bsup_neg_AMdist` (0.264): +0.1577 +- 0.0285, lower in 3/20 -> **worse**

### seeds 10-14: b3p1, n_forget=4

                        tv_to_ideal  neg_mass_ratio  retain_tv  mass_off  retain_noise_flip  sdpo_pos_frac_in_forget
arm                                                                                                                 
SDPO_fresh_b1                 0.516           0.565      0.076     0.243              0.369                    0.500
SDPO_fresh_b10                0.497           0.949      0.005     0.037              0.059                    0.500
SDPO_fresh_b25                0.505           0.982      0.003     0.029              0.020                    0.500
SDPO_fresh_b100               0.510           0.995      0.001     0.028              0.007                    0.500
SDPO_pin_b1                   0.624           0.244      0.207     0.503              0.695                    0.500
SDPO_pin_b25                  0.462           0.856      0.052     0.047              0.144                    0.500
SDPO_hop_pin_b1               0.662           0.266      0.329     0.532              0.771                    0.581
SDPO_hop_pin_b25              0.455           0.653      0.067     0.139              0.260                    0.581
SDPO_hop_pin_bH1              0.693           0.247      0.303     0.572              0.786                    0.581
SDPO_hop_pin_bH3              0.588           0.309      0.235     0.432              0.690                    0.581
SDPO_hop_pin_bH10             0.496           0.402      0.127     0.301              0.478                    0.581
CTRL_hop_GA                   0.753           0.244      0.266     0.631              0.748                    0.581
CTRL_synB1_hop                0.635           1.061      0.229     0.101              0.722                    0.581
R_hopB_bsup_neg_AMdist        0.255           0.442      0.044     0.038              0.257                      NaN
R_hop_bsup_neg                0.414           0.747      0.128     0.051              0.403                      NaN
NPO_b10_neg_AMdist            0.309           0.514      0.059     0.058              0.162                      NaN
PSEUDO_vote_dist              0.288           0.502      0.044     0.044              0.220                      NaN
B0_frozen                     0.512           1.000      0.000     0.028              0.000                      NaN

best SDPO: `SDPO_hop_pin_b25` (0.455)
- vs `R_hop_bsup_neg` (0.414): +0.0413 +- 0.0230, lower in 6/20 -> **tie**
- vs `R_hopB_bsup_neg_AMdist` (0.255): +0.2002 +- 0.0273, lower in 2/20 -> **worse**

