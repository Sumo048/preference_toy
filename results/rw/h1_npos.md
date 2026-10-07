# H1: scarce positives (n_pos sweep), pre-registered decision rule

## npos512  (120 arms, 45 runs per complete arm)

**NEG** best: `R_near_nf_nhood_neg` tv 0.267 (forget 0.120, retain TV 0.219, off 0.089)

- best rerouting `R_near_nf_nhood_neg` 0.267 vs best non-rerouting `B3_npo_b100_negonly` 0.289: paired diff -0.0225 +- 0.0358 (n=45, rerouting lower in 18); dominated by: -
- criteria 1/2/3: Y/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| R_near_nf_nhood_neg (R) | 0.267 | 0.120 | 0.219 | 0.089 | 0.00 | 0.01 |
| R_near_nf_log_neg (R) | 0.270 | 0.189 | 0.157 | 0.176 | 0.00 | 0.01 |
| B3_npo_b100_negonly | 0.289 | 0.739 | 0.028 | 0.058 | nan | nan |
| B5_kto_b100_negonly | 0.353 | 0.075 | 0.149 | 0.304 | nan | nan |
| B3_npo_b10_negonly | 0.416 | 0.124 | 0.213 | 0.287 | nan | nan |
| R_spread_nf_log_neg (R) | 0.448 | 0.085 | 0.355 | 0.259 | 0.00 | 0.00 |
| R_ours_nf_neg (R) | 0.457 | 0.109 | 0.302 | 0.318 | 0.00 | 0.00 |
| R_ours_neg (R) | 0.555 | 0.351 | 0.337 | 0.363 | 0.02 | 0.27 |

**NEG+U** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

**POS+NEG** best: `DPO_cpl_b1` tv 0.056 (forget 0.000, retain TV 0.046, off 0.021)

- best rerouting `R_near_nf_log_both` 0.070 vs best non-rerouting `DPO_cpl_b1` 0.056: paired diff +0.0144 +- 0.0025 (n=45, rerouting lower in 6); dominated by: B1_filtered
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| DPO_cpl_b1 | 0.056 | 0.000 | 0.046 | 0.021 | nan | nan |
| B1_filtered | 0.063 | 0.004 | 0.053 | 0.020 | nan | nan |
| R_near_nf_log_both (R) | 0.070 | 0.010 | 0.056 | 0.027 | 0.00 | 0.01 |
| B3_npo_b1 | 0.073 | 0.000 | 0.066 | 0.018 | nan | nan |
| B3_npo_b10 | 0.073 | 0.008 | 0.059 | 0.025 | nan | nan |
| B3_npo_b100 | 0.079 | 0.045 | 0.061 | 0.022 | nan | nan |
| R_near_nf_nhood_both (R) | 0.081 | 0.010 | 0.069 | 0.021 | 0.00 | 0.01 |
| B5_kto_b10 | 0.122 | 0.061 | 0.068 | 0.077 | nan | nan |

**ALL (POS+NEG+U)** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| DPO_cpl_b1 | 0.056 | 0.000 | 0.046 | 0.021 | nan | nan |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |

## npos64  (87 arms, 45 runs per complete arm)

incomplete arms (excluded): B0_frozen, B1_filtered, B3_npo_b0.1, B3_npo_b1, B3_npo_b10, B5_kto_b0.1, B5_kto_b1, B5_kto_b10

**NEG** best: `R_near_nf_nhood_neg` tv 0.267 (forget 0.120, retain TV 0.219, off 0.089)

- best rerouting `R_near_nf_nhood_neg` 0.267 vs best non-rerouting `B3_npo_b100_negonly` 0.289: paired diff -0.0225 +- 0.0358 (n=45, rerouting lower in 18); dominated by: -
- criteria 1/2/3: Y/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| R_near_nf_nhood_neg (R) | 0.267 | 0.120 | 0.219 | 0.089 | 0.00 | 0.01 |
| R_near_nf_log_neg (R) | 0.270 | 0.189 | 0.157 | 0.176 | 0.00 | 0.01 |
| B3_npo_b100_negonly | 0.289 | 0.739 | 0.028 | 0.058 | nan | nan |
| B5_kto_b100_negonly | 0.353 | 0.075 | 0.149 | 0.304 | nan | nan |
| B3_npo_b10_negonly | 0.416 | 0.124 | 0.213 | 0.287 | nan | nan |
| R_spread_nf_log_neg (R) | 0.448 | 0.085 | 0.355 | 0.259 | 0.00 | 0.00 |
| R_ours_nf_neg (R) | 0.457 | 0.109 | 0.302 | 0.318 | 0.00 | 0.00 |
| R_ours_neg (R) | 0.555 | 0.351 | 0.337 | 0.363 | 0.02 | 0.27 |

**NEG+U** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

**ALL (POS+NEG+U)** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

## npos16  (115 arms, 45 runs per complete arm)

incomplete arms (excluded): B4_naive_ga, DPO_cpl_b1, DPO_cpl_b25, RM_bon8, R_guide_log_both, R_guide_nhood_both, R_ours_both, R_ours_filt_both, R_spread_filt_log_both, R_spread_filt_nhood_both, R_spread_filt_nhood_neg_kl1, R_spread_inv_r0.5_both, R_spread_inv_r1.5_both, R_spread_inv_r3_both, R_spread_log_both, R_spread_nhood_both, R_spread_nhood_neg_kl0.1, R_spread_nhood_neg_kl0.3, R_spread_nhood_neg_kl1, R_spread_ninv_t0.4_fresh_both, R_spread_ninv_t0.4_mix_both, R_spread_ninv_t0.6_fresh_both, R_spread_ninv_t0.8_fresh_both, R_spread_reach_far_both, R_spread_reach_mid_both, R_spread_reach_near_both

**NEG** best: `R_near_nf_nhood_neg` tv 0.267 (forget 0.120, retain TV 0.219, off 0.089)

- best rerouting `R_near_nf_nhood_neg` 0.267 vs best non-rerouting `B3_npo_b100_negonly` 0.289: paired diff -0.0225 +- 0.0358 (n=45, rerouting lower in 18); dominated by: -
- criteria 1/2/3: Y/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| R_near_nf_nhood_neg (R) | 0.267 | 0.120 | 0.219 | 0.089 | 0.00 | 0.01 |
| R_near_nf_log_neg (R) | 0.270 | 0.189 | 0.157 | 0.176 | 0.00 | 0.01 |
| B3_npo_b100_negonly | 0.289 | 0.739 | 0.028 | 0.058 | nan | nan |
| B5_kto_b100_negonly | 0.353 | 0.075 | 0.149 | 0.304 | nan | nan |
| B3_npo_b10_negonly | 0.416 | 0.124 | 0.213 | 0.287 | nan | nan |
| R_spread_nf_log_neg (R) | 0.448 | 0.085 | 0.355 | 0.259 | 0.00 | 0.00 |
| R_ours_nf_neg (R) | 0.457 | 0.109 | 0.302 | 0.318 | 0.00 | 0.00 |
| R_ours_neg (R) | 0.555 | 0.351 | 0.337 | 0.363 | 0.02 | 0.27 |

**NEG+U** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

**POS+NEG** best: `B3_npo_b10` tv 0.157 (forget 0.009, retain TV 0.149, off 0.022)

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B3_npo_b10 | 0.157 | 0.009 | 0.149 | 0.022 | nan | nan |
| B3_npo_b100 | 0.160 | 0.038 | 0.149 | 0.020 | nan | nan |
| B5_kto_b10 | 0.169 | 0.078 | 0.115 | 0.081 | nan | nan |
| B3_npo_b1 | 0.177 | 0.000 | 0.173 | 0.017 | nan | nan |
| B1_filtered | 0.182 | 0.006 | 0.176 | 0.019 | nan | nan |
| B5_kto_b100 | 0.194 | 0.385 | 0.048 | 0.072 | nan | nan |
| B5_kto_b1 | 0.271 | 0.000 | 0.269 | 0.011 | nan | nan |
| B3_npo_b0.1 | 0.384 | 0.000 | 0.380 | 0.023 | nan | nan |

**ALL (POS+NEG+U)** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

## npos4  (115 arms, 45 runs per complete arm)

**NEG** best: `R_near_nf_nhood_neg` tv 0.267 (forget 0.120, retain TV 0.219, off 0.089)

- best rerouting `R_near_nf_nhood_neg` 0.267 vs best non-rerouting `B3_npo_b100_negonly` 0.289: paired diff -0.0225 +- 0.0358 (n=45, rerouting lower in 18); dominated by: -
- criteria 1/2/3: Y/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| R_near_nf_nhood_neg (R) | 0.267 | 0.120 | 0.219 | 0.089 | 0.00 | 0.01 |
| R_near_nf_log_neg (R) | 0.270 | 0.189 | 0.157 | 0.176 | 0.00 | 0.01 |
| B3_npo_b100_negonly | 0.289 | 0.739 | 0.028 | 0.058 | nan | nan |
| B5_kto_b100_negonly | 0.353 | 0.075 | 0.149 | 0.304 | nan | nan |
| B3_npo_b10_negonly | 0.416 | 0.124 | 0.213 | 0.287 | nan | nan |
| R_spread_nf_log_neg (R) | 0.448 | 0.085 | 0.355 | 0.259 | 0.00 | 0.00 |
| R_ours_nf_neg (R) | 0.457 | 0.109 | 0.302 | 0.318 | 0.00 | 0.00 |
| R_ours_neg (R) | 0.555 | 0.351 | 0.337 | 0.363 | 0.02 | 0.27 |

**NEG+U** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

**POS+NEG** best: `B5_kto_b100` tv 0.240 (forget 0.289, retain TV 0.143, off 0.089)

- best rerouting `R_ours_both` 0.386 vs best non-rerouting `B5_kto_b100` 0.240: paired diff +0.1468 +- 0.0211 (n=45, rerouting lower in 8); dominated by: -
- criteria 1/2/3: N/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b100 | 0.240 | 0.289 | 0.143 | 0.089 | nan | nan |
| B5_kto_b10 | 0.330 | 0.029 | 0.297 | 0.111 | nan | nan |
| R_ours_both (R) | 0.386 | 0.179 | 0.359 | 0.042 | 0.02 | 0.27 |
| R_spread_nhood_neg_kl1 (R) | 0.386 | 0.493 | 0.185 | 0.178 | 0.14 | 0.19 |
| R_spread_ninv_t0.4_fresh_both (R) | 0.397 | 0.188 | 0.353 | 0.079 | 0.14 | 0.19 |
| R_spread_nhood_both (R) | 0.401 | 0.181 | 0.359 | 0.078 | 0.14 | 0.19 |
| R_spread_ninv_t0.6_fresh_both (R) | 0.403 | 0.191 | 0.358 | 0.082 | 0.14 | 0.19 |
| R_spread_reach_mid_both (R) | 0.403 | 0.184 | 0.365 | 0.070 | 0.14 | 0.19 |

**ALL (POS+NEG+U)** best: `B5_kto_b10_negonly_Anf_dist` tv 0.040 (forget 0.005, retain TV 0.017, off 0.036)

- best rerouting `R_near_nf_log_neg_Anf_dist` 0.054 vs best non-rerouting `B5_kto_b10_negonly_Anf_dist` 0.040: paired diff +0.0142 +- 0.0022 (n=45, rerouting lower in 3); dominated by: B5_kto_b100_negonly_Anf_dist
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b10_negonly_Anf_dist | 0.040 | 0.005 | 0.017 | 0.036 | nan | nan |
| PSEUDO_nf_dist | 0.041 | 0.011 | 0.014 | 0.036 | nan | nan |
| B5_kto_b100_negonly_Anf_dist | 0.041 | 0.008 | 0.016 | 0.036 | nan | nan |
| R_near_nf_log_neg_Anf_dist (R) | 0.054 | 0.015 | 0.025 | 0.041 | 0.00 | 0.01 |
| B3_npo_b10_negonly_Anf_dist | 0.057 | 0.011 | 0.032 | 0.040 | nan | nan |
| B3_npo_b100_negonly_Anf_dist | 0.062 | 0.042 | 0.034 | 0.037 | nan | nan |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.063 | 0.014 | 0.040 | 0.034 | 0.00 | 0.01 |
| R_spread_nf_log_neg_Anf_dist (R) | 0.116 | 0.013 | 0.083 | 0.053 | 0.00 | 0.00 |

## Verdict table

| regime | setting | best rerouting | best other | paired diff (R - other) | rerouting best? |
|---|---|---|---|---|---|
| npos512 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| npos512 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos512 | POS+NEG | R_near_nf_log_both | DPO_cpl_b1 | +0.0144 +- 0.0025 | no |
| npos512 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos64 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| npos64 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos64 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos16 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| npos16 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos16 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos4 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| npos4 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| npos4 | POS+NEG | R_ours_both | B5_kto_b100 | +0.1468 +- 0.0211 | no |
| npos4 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
