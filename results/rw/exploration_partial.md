# Round rw: exploration (decision rule of docs/REROUTE_WINS_PREREG.md)

seeds [0, 1, 2, 3, 4], cells n_forget x n_neg = [(1, 1), (1, 4), (1, 16), (2, 1), (2, 4), (2, 16), (4, 1), (4, 4), (4, 16)]

## base  (120 arms, 45 runs per complete arm)

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

## contam0.2  (99 arms, 45 runs per complete arm)

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

**POS+NEG** best: `B3_npo_b1` tv 0.124 (forget 0.249, retain TV 0.058, off 0.032)

- best rerouting `R_near_nf_nhood_both` 0.159 vs best non-rerouting `B3_npo_b1` 0.124: paired diff +0.0343 +- 0.0084 (n=45, rerouting lower in 12); dominated by: -
- criteria 1/2/3: N/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B3_npo_b1 | 0.124 | 0.249 | 0.058 | 0.032 | nan | nan |
| B5_kto_b10 | 0.137 | 0.320 | 0.064 | 0.033 | nan | nan |
| R_near_nf_nhood_both (R) | 0.159 | 0.514 | 0.080 | 0.020 | 0.00 | 0.01 |
| R_near_nf_log_both (R) | 0.169 | 0.638 | 0.060 | 0.024 | 0.00 | 0.01 |
| DPO_cpl_b1 | 0.172 | 0.510 | 0.031 | 0.028 | nan | nan |
| R_ours_nf_both (R) | 0.190 | 0.445 | 0.115 | 0.033 | 0.00 | 0.00 |
| B5_kto_b100 | 0.214 | 0.593 | 0.029 | 0.045 | nan | nan |
| R_spread_nf_log_both (R) | 0.214 | 0.616 | 0.109 | 0.041 | 0.00 | 0.00 |

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

## cover0.25  (99 arms, 45 runs per complete arm)

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

**POS+NEG** best: `B5_kto_b100` tv 0.290 (forget 0.340, retain TV 0.166, off 0.092)

- best rerouting `R_spread_nhood_neg_kl1` 0.411 vs best non-rerouting `B5_kto_b100` 0.290: paired diff +0.1201 +- 0.0140 (n=45, rerouting lower in 4); dominated by: B5_kto_b100
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b100 | 0.290 | 0.340 | 0.166 | 0.092 | nan | nan |
| B5_kto_b10 | 0.403 | 0.009 | 0.371 | 0.098 | nan | nan |
| R_spread_nhood_neg_kl1 (R) | 0.411 | 0.500 | 0.211 | 0.205 | 0.14 | 0.19 |
| R_ours_nf_both (R) | 0.546 | 0.007 | 0.540 | 0.029 | 0.00 | 0.00 |
| R_near_nf_nhood_both (R) | 0.554 | 0.002 | 0.551 | 0.017 | 0.00 | 0.01 |
| RM_bon8 | 0.567 | 0.101 | 0.561 | 0.019 | nan | nan |
| R_ours_both (R) | 0.588 | 0.188 | 0.571 | 0.032 | 0.02 | 0.27 |
| R_spread_nhood_both (R) | 0.589 | 0.160 | 0.571 | 0.057 | 0.14 | 0.19 |

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

## cover0.5  (99 arms, 45 runs per complete arm)

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

**POS+NEG** best: `B5_kto_b100` tv 0.224 (forget 0.271, retain TV 0.144, off 0.102)

- best rerouting `R_ours_nf_both` 0.359 vs best non-rerouting `B5_kto_b100` 0.224: paired diff +0.1347 +- 0.0188 (n=45, rerouting lower in 9); dominated by: -
- criteria 1/2/3: N/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b100 | 0.224 | 0.271 | 0.144 | 0.102 | nan | nan |
| B5_kto_b10 | 0.316 | 0.016 | 0.288 | 0.119 | nan | nan |
| R_ours_nf_both (R) | 0.359 | 0.004 | 0.354 | 0.032 | 0.00 | 0.00 |
| R_near_nf_nhood_both (R) | 0.363 | 0.006 | 0.361 | 0.021 | 0.00 | 0.01 |
| R_spread_nhood_neg_kl1 (R) | 0.383 | 0.525 | 0.190 | 0.166 | 0.14 | 0.19 |
| DPO_cpl_b25 | 0.386 | 0.332 | 0.305 | 0.050 | nan | nan |
| R_spread_nhood_both (R) | 0.393 | 0.161 | 0.373 | 0.065 | 0.14 | 0.19 |
| R_ours_both (R) | 0.395 | 0.171 | 0.382 | 0.039 | 0.02 | 0.27 |

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

## std0.45  (83 arms, 45 runs per complete arm)

**NEG** best: `B3_npo_b100_negonly` tv 0.264 (forget 0.780, retain TV 0.031, off 0.015)

- best rerouting `R_near_nf_log_neg` 0.272 vs best non-rerouting `B3_npo_b100_negonly` 0.264: paired diff +0.0081 +- 0.0224 (n=45, rerouting lower in 18); dominated by: -
- criteria 1/2/3: N/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B3_npo_b100_negonly | 0.264 | 0.780 | 0.031 | 0.015 | nan | nan |
| R_near_nf_log_neg (R) | 0.272 | 0.230 | 0.228 | 0.044 | 0.00 | 0.04 |
| B5_kto_b100_negonly | 0.274 | 0.253 | 0.138 | 0.126 | nan | nan |
| R_near_nf_nhood_neg (R) | 0.279 | 0.256 | 0.253 | 0.006 | 0.00 | 0.04 |
| B3_npo_b10_negonly | 0.357 | 0.257 | 0.200 | 0.128 | nan | nan |
| R_spread_nf_log_neg (R) | 0.441 | 0.170 | 0.368 | 0.168 | 0.00 | 0.01 |
| R_ours_nf_neg (R) | 0.457 | 0.153 | 0.344 | 0.258 | 0.00 | 0.00 |
| R_ours_neg (R) | 0.495 | 0.403 | 0.311 | 0.278 | 0.01 | 0.27 |

**NEG+U** best: `B5_kto_b100_negonly_Anf_map` tv 0.133 (forget 0.117, retain TV 0.092, off 0.030)

- best rerouting `R_near_nf_log_neg_Anf_map` 0.167 vs best non-rerouting `B5_kto_b100_negonly_Anf_map` 0.133: paired diff +0.0334 +- 0.0082 (n=45, rerouting lower in 8); dominated by: -
- criteria 1/2/3: N/N/Y -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| B5_kto_b100_negonly_Anf_map | 0.133 | 0.117 | 0.092 | 0.030 | nan | nan |
| B5_kto_b10_negonly_Anf_map | 0.143 | 0.119 | 0.098 | 0.038 | nan | nan |
| R_near_nf_log_neg_Anf_map (R) | 0.167 | 0.278 | 0.108 | 0.016 | 0.00 | 0.04 |
| B3_npo_b10_negonly_Anf_map | 0.168 | 0.241 | 0.081 | 0.021 | nan | nan |
| R_spread_nf_log_neg_Anf_map (R) | 0.173 | 0.253 | 0.111 | 0.023 | 0.00 | 0.01 |
| R_near_nf_nhood_neg_Anf_map (R) | 0.175 | 0.186 | 0.142 | 0.010 | 0.00 | 0.04 |
| R_near_nf_nhood_neg_Anf_dist (R) | 0.176 | 0.033 | 0.171 | 0.008 | 0.00 | 0.04 |
| B3_npo_b100_negonly_Anf_map | 0.187 | 0.384 | 0.072 | 0.014 | nan | nan |

**POS+NEG** best: `DPO_cpl_b1` tv 0.050 (forget 0.008, retain TV 0.046, off 0.010)

- best rerouting `R_near_nf_log_both` 0.092 vs best non-rerouting `DPO_cpl_b1` 0.050: paired diff +0.0417 +- 0.0067 (n=45, rerouting lower in 5); dominated by: B1_filtered
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| DPO_cpl_b1 | 0.050 | 0.008 | 0.046 | 0.010 | nan | nan |
| B1_filtered | 0.059 | 0.032 | 0.053 | 0.007 | nan | nan |
| B3_npo_b1 | 0.075 | 0.002 | 0.073 | 0.008 | nan | nan |
| B3_npo_b10 | 0.075 | 0.075 | 0.060 | 0.008 | nan | nan |
| B3_npo_b100 | 0.084 | 0.119 | 0.063 | 0.008 | nan | nan |
| R_near_nf_log_both (R) | 0.092 | 0.040 | 0.084 | 0.009 | 0.00 | 0.04 |
| B5_kto_b10 | 0.097 | 0.145 | 0.063 | 0.016 | nan | nan |
| R_near_nf_nhood_both (R) | 0.107 | 0.061 | 0.096 | 0.007 | 0.00 | 0.04 |

**ALL (POS+NEG+U)** best: `DPO_cpl_b1` tv 0.050 (forget 0.008, retain TV 0.046, off 0.010)

- best rerouting `R_near_nf_log_both` 0.092 vs best non-rerouting `DPO_cpl_b1` 0.050: paired diff +0.0417 +- 0.0067 (n=45, rerouting lower in 5); dominated by: B1_filtered
- criteria 1/2/3: N/N/N -> **rerouting not best**

| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |
|---|---|---|---|---|---|---|
| DPO_cpl_b1 | 0.050 | 0.008 | 0.046 | 0.010 | nan | nan |
| B1_filtered | 0.059 | 0.032 | 0.053 | 0.007 | nan | nan |
| B3_npo_b1 | 0.075 | 0.002 | 0.073 | 0.008 | nan | nan |
| B3_npo_b10 | 0.075 | 0.075 | 0.060 | 0.008 | nan | nan |
| B3_npo_b100 | 0.084 | 0.119 | 0.063 | 0.008 | nan | nan |
| R_near_nf_log_both (R) | 0.092 | 0.040 | 0.084 | 0.009 | 0.00 | 0.04 |
| B5_kto_b10 | 0.097 | 0.145 | 0.063 | 0.016 | nan | nan |
| R_near_nf_nhood_both (R) | 0.107 | 0.061 | 0.096 | 0.007 | 0.00 | 0.04 |

## std0.6  (83 arms, 29 runs per complete arm)

incomplete arms (excluded): B1_filtered, B3_npo_b1, B3_npo_b10, B3_npo_b100, B3_npo_b100_negonly, B3_npo_b100_negonly_Anf_dist, B3_npo_b100_negonly_Anf_map, B3_npo_b100_negonly_Aself_dist, B3_npo_b100_negonly_Aself_map, B3_npo_b10_negonly, B3_npo_b10_negonly_Anf_dist, B3_npo_b10_negonly_Anf_map, B3_npo_b10_negonly_Aself_dist, B3_npo_b10_negonly_Aself_map, B5_kto_b10, B5_kto_b100, B5_kto_b100_negonly, B5_kto_b100_negonly_Anf_dist, B5_kto_b100_negonly_Anf_map, B5_kto_b100_negonly_Aself_dist, B5_kto_b100_negonly_Aself_map, B5_kto_b10_negonly, B5_kto_b10_negonly_Anf_dist, B5_kto_b10_negonly_Anf_map, B5_kto_b10_negonly_Aself_dist, B5_kto_b10_negonly_Aself_map, DPO_cpl_b1, DPO_cpl_b25, ORACLE_retain_bc, PSEUDO_nf_dist, PSEUDO_nf_map, RM_bon8, R_near_filt_log_both, R_near_filt_log_neg_Anf_map, R_near_nf_log_both, R_near_nf_log_neg, R_near_nf_log_neg_Anf_dist, R_near_nf_log_neg_Anf_map, R_near_nf_log_neg_Aself_dist, R_near_nf_log_neg_Aself_map, R_near_nf_nhood_both, R_near_nf_nhood_neg, R_near_nf_nhood_neg_Anf_dist, R_near_nf_nhood_neg_Anf_map, R_near_nf_nhood_neg_Aself_dist, R_near_nf_nhood_neg_Aself_map, R_near_nf_nhoodx_0.6_16_neg_Anf_map, R_near_nf_nhoodx_0.6_16_neg_Aself_map, R_near_nf_nhoodx_1.0_16_neg_Anf_map, R_near_nf_nhoodx_1.0_16_neg_Aself_map, R_ours_both, R_ours_neg, R_ours_neg_Anf_dist, R_ours_neg_Anf_map, R_ours_neg_Aself_dist, R_ours_neg_Aself_map, R_ours_nf_both, R_ours_nf_neg, R_ours_nf_neg_Anf_dist, R_ours_nf_neg_Anf_map, R_ours_nf_neg_Aself_dist, R_ours_nf_neg_Aself_map, R_spread_filt_log_both, R_spread_log_both, R_spread_log_neg, R_spread_log_neg_Anf_dist, R_spread_log_neg_Anf_map, R_spread_log_neg_Aself_dist, R_spread_log_neg_Aself_map, R_spread_nf_log_both, R_spread_nf_log_neg, R_spread_nf_log_neg_Anf_dist, R_spread_nf_log_neg_Anf_map, R_spread_nf_log_neg_Aself_dist, R_spread_nf_log_neg_Aself_map, R_spread_nhood_both, R_spread_nhood_neg, R_spread_nhood_neg_Anf_dist, R_spread_nhood_neg_Anf_map, R_spread_nhood_neg_Aself_dist, R_spread_nhood_neg_Aself_map, R_spread_nhood_neg_kl1

## Verdict table

| regime | setting | best rerouting | best other | paired diff (R - other) | rerouting best? |
|---|---|---|---|---|---|
| base | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| base | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| base | POS+NEG | R_near_nf_log_both | DPO_cpl_b1 | +0.0144 +- 0.0025 | no |
| base | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| contam0.2 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| contam0.2 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| contam0.2 | POS+NEG | R_near_nf_nhood_both | B3_npo_b1 | +0.0343 +- 0.0084 | no |
| contam0.2 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| cover0.25 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| cover0.25 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| cover0.25 | POS+NEG | R_spread_nhood_neg_kl1 | B5_kto_b100 | +0.1201 +- 0.0140 | no |
| cover0.25 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| cover0.5 | NEG | R_near_nf_nhood_neg | B3_npo_b100_negonly | -0.0225 +- 0.0358 | no |
| cover0.5 | NEG+U | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| cover0.5 | POS+NEG | R_ours_nf_both | B5_kto_b100 | +0.1347 +- 0.0188 | no |
| cover0.5 | ALL (POS+NEG+U) | R_near_nf_log_neg_Anf_dist | B5_kto_b10_negonly_Anf_dist | +0.0142 +- 0.0022 | no |
| std0.45 | NEG | R_near_nf_log_neg | B3_npo_b100_negonly | +0.0081 +- 0.0224 | no |
| std0.45 | NEG+U | R_near_nf_log_neg_Anf_map | B5_kto_b100_negonly_Anf_map | +0.0334 +- 0.0082 | no |
| std0.45 | POS+NEG | R_near_nf_log_both | DPO_cpl_b1 | +0.0417 +- 0.0067 | no |
| std0.45 | ALL (POS+NEG+U) | R_near_nf_log_both | DPO_cpl_b1 | +0.0417 +- 0.0067 | no |
