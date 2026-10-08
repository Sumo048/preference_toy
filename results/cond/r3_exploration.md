# Round 3 / layouts (r3_exploration.md), seeds [0, 1, 2, 3, 4], 4480 runs

## layout=random, n_forget=1 (B0 0.149; 20 runs per arm)

best rerouting `R_hopB_bsup_neg_AMdist` vs best other `NPO_b10_neg_AMdist`: -0.0091 +- 0.0027 (lower in 15/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hopB_bsup_neg_AMdist (R) | 0.106 | 0.248 | 0.077 | 0.034 | 0.29 | 0.07/0.29 | 0.01 | 0.52/0.74 | nan |
| NPO_b10_neg_AMdist | 0.115 | 0.282 | 0.085 | 0.042 | 0.28 | 0.14/0.31 | nan | 0.52/0.74 | nan |
| PSEUDO_vote_dist | 0.116 | 0.315 | 0.086 | 0.038 | 0.28 | 0.18/0.33 | nan | 0.52/0.74 | nan |
| KTO_b10_neg_AL | 0.119 | 0.162 | 0.051 | 0.085 | 0.13 | 0.02/0.20 | nan | nan/nan | 1083 |
| NPO_b10_neg_AL | 0.119 | 0.183 | 0.051 | 0.084 | 0.13 | 0.03/0.23 | nan | nan/nan | 1083 |
| NPO_b100_neg_AL | 0.121 | 0.274 | 0.046 | 0.078 | 0.12 | 0.12/0.32 | nan | nan/nan | 1083 |
| R_hopB_bsup_mine_neg_AMmap (R) | 0.123 | 0.265 | 0.078 | 0.045 | 0.17 | 0.08/0.27 | 0.01 | 0.52/0.74 | nan |
| R_hopB_bsup_neg_AMmap (R) | 0.128 | 0.484 | 0.038 | 0.057 | 0.07 | 0.19/0.68 | 0.01 | 0.52/0.74 | nan |
| NPO_b10_neg_AMmap | 0.129 | 0.417 | 0.026 | 0.075 | 0.07 | 0.08/0.67 | nan | 0.52/0.74 | nan |
| R_hopB_bsup_neg_AML (R) | 0.130 | 0.477 | 0.041 | 0.055 | 0.06 | 0.19/0.69 | 0.01 | 0.52/0.74 | 1083 |
| NPO_b100_neg_AMmap | 0.132 | 0.515 | 0.023 | 0.068 | 0.07 | 0.23/0.71 | nan | 0.52/0.74 | nan |
| KTO_b10_neg_AMmap | 0.133 | 0.423 | 0.022 | 0.079 | 0.07 | 0.06/0.70 | nan | 0.52/0.74 | nan |
| R_hopB_bsup_neg_AL (R) | 0.135 | 0.309 | 0.062 | 0.069 | 0.12 | 0.10/0.32 | 0.01 | nan/nan | 1083 |
| R_hop_bsup_neg_AL (R) | 0.140 | 0.332 | 0.065 | 0.068 | 0.11 | 0.13/0.32 | 0.08 | nan/nan | 1083 |

## layout=random, n_forget=2 (B0 0.270; 20 runs per arm)

best rerouting `R_hopB_bsup_mine_neg_AMmap` vs best other `NPO_b10_neg_AL`: -0.0254 +- 0.0083 (lower in 16/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hopB_bsup_mine_neg_AMmap (R) | 0.175 | 0.422 | 0.098 | 0.039 | 0.25 | 0.22/0.37 | 0.03 | 0.51/0.73 | nan |
| R_hopB_bsup_neg_AL (R) | 0.193 | 0.509 | 0.052 | 0.064 | 0.14 | 0.40/0.59 | 0.03 | nan/nan | 2159 |
| R_hopB_bsup_neg_AMdist (R) | 0.197 | 0.341 | 0.137 | 0.037 | 0.36 | 0.18/0.36 | 0.03 | 0.51/0.73 | nan |
| R_hopB_bsup_neg_AMmap (R) | 0.198 | 0.546 | 0.053 | 0.058 | 0.12 | 0.36/0.68 | 0.03 | 0.51/0.73 | nan |
| R_hopB_bsup_neg_AML (R) | 0.198 | 0.565 | 0.041 | 0.058 | 0.10 | 0.43/0.73 | 0.03 | 0.51/0.73 | 2159 |
| NPO_b10_neg_AL | 0.200 | 0.423 | 0.047 | 0.097 | 0.14 | 0.33/0.46 | nan | nan/nan | 2159 |
| R_hop_bsup_neg_AL (R) | 0.201 | 0.516 | 0.070 | 0.055 | 0.10 | 0.42/0.57 | 0.06 | nan/nan | 2159 |
| R_hopB_bsup_mine_neg (R) | 0.206 | 0.351 | 0.156 | 0.028 | 0.43 | 0.15/0.24 | 0.03 | 0.51/0.73 | nan |
| NPO_b100_neg_AL | 0.209 | 0.495 | 0.044 | 0.088 | 0.13 | 0.40/0.52 | nan | nan/nan | 2159 |
| NPO_b10_neg_AMdist | 0.217 | 0.356 | 0.151 | 0.044 | 0.35 | 0.23/0.36 | nan | 0.51/0.73 | nan |
| PSEUDO_vote_dist | 0.219 | 0.378 | 0.154 | 0.040 | 0.36 | 0.23/0.36 | nan | 0.51/0.73 | nan |
| NPO_b10_neg_AMmap | 0.220 | 0.516 | 0.043 | 0.094 | 0.10 | 0.34/0.66 | nan | 0.51/0.73 | nan |
| KTO_b10_neg_AL | 0.227 | 0.505 | 0.036 | 0.105 | 0.13 | 0.43/0.47 | nan | nan/nan | 2159 |
| R_hop_nhood_tx_neg (R) | 0.228 | 0.651 | 0.078 | 0.062 | 0.28 | 0.40/0.56 | 0.27 | nan/nan | nan |

## layout=random, n_forget=4 (B0 0.516; 20 runs per arm)

best rerouting `R_hopB_bsup_mine_neg` vs best other `PSEUDO_vote_dist`: -0.0561 +- 0.0182 (lower in 18/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hopB_bsup_mine_neg (R) | 0.304 | 0.502 | 0.145 | 0.039 | 0.45 | 0.32/0.47 | 0.22 | 0.66/0.65 | nan |
| R_hopB_bsup_neg (R) | 0.314 | 0.433 | 0.166 | 0.074 | 0.48 | 0.25/0.39 | 0.22 | nan/nan | nan |
| R_hopB_bsup_neg_AMdist (R) | 0.318 | 0.514 | 0.112 | 0.042 | 0.30 | 0.40/0.63 | 0.22 | 0.66/0.65 | nan |
| R_hopVB_bsup_neg (R) | 0.323 | 0.427 | 0.180 | 0.076 | 0.50 | 0.25/0.44 | 0.22 | nan/nan | nan |
| R_hopB_bsup_mine_neg_AMmap (R) | 0.324 | 0.574 | 0.066 | 0.041 | 0.22 | 0.47/0.67 | 0.22 | 0.66/0.65 | nan |
| R_hopB_bsup_neg_AMmap (R) | 0.345 | 0.588 | 0.057 | 0.056 | 0.14 | 0.50/0.79 | 0.22 | 0.66/0.65 | nan |
| R_hopV_bsup_neg (R) | 0.345 | 0.563 | 0.151 | 0.058 | 0.41 | 0.38/0.54 | 0.37 | nan/nan | nan |
| R_hop_bsup_neg (R) | 0.355 | 0.611 | 0.132 | 0.052 | 0.35 | 0.44/0.51 | 0.41 | nan/nan | nan |
| PSEUDO_vote_dist | 0.360 | 0.587 | 0.136 | 0.048 | 0.29 | 0.48/0.65 | nan | 0.66/0.65 | nan |
| R_hopB_bsup_neg_AL (R) | 0.367 | 0.639 | 0.049 | 0.055 | 0.14 | 0.60/0.74 | 0.22 | nan/nan | 4323 |
| NPO_b10_neg_AMdist | 0.368 | 0.598 | 0.118 | 0.053 | 0.21 | 0.53/0.65 | nan | 0.66/0.65 | nan |
| R_hop_nhood_neg (R) | 0.369 | 0.657 | 0.124 | 0.044 | 0.31 | 0.49/0.59 | 0.41 | nan/nan | nan |
| R_hopB_bsup_neg_AML (R) | 0.374 | 0.658 | 0.037 | 0.053 | 0.11 | 0.61/0.84 | 0.22 | 0.66/0.65 | 4323 |
| R_hop_bsup_neg_AL (R) | 0.401 | 0.727 | 0.046 | 0.046 | 0.08 | 0.70/0.76 | 0.41 | nan/nan | 4323 |

## layout=contig, n_forget=4 (B0 0.513; 20 runs per arm)

best rerouting `R_hopB_bsup_neg_AMdist` vs best other `NPO_b10_neg_AMdist`: -0.0066 +- 0.0031 (lower in 13/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hopB_bsup_neg_AMdist (R) | 0.203 | 0.330 | 0.055 | 0.036 | 0.31 | 0.32/0.50 | 0.36 | 0.88/0.76 | nan |
| NPO_b10_neg_AMdist | 0.210 | 0.314 | 0.065 | 0.046 | 0.30 | 0.29/0.50 | nan | 0.88/0.76 | nan |
| PSEUDO_vote_dist | 0.215 | 0.338 | 0.067 | 0.040 | 0.30 | 0.32/0.51 | nan | 0.88/0.76 | nan |
| R_hopB_bsup_mine_neg (R) | 0.236 | 0.391 | 0.085 | 0.038 | 0.38 | 0.35/0.59 | 0.36 | 0.88/0.76 | nan |
| R_hopB_bsup_mine_neg_AMmap (R) | 0.274 | 0.472 | 0.071 | 0.045 | 0.15 | 0.44/0.69 | 0.36 | 0.88/0.76 | nan |
| R_hopB_bsup_neg (R) | 0.302 | 0.380 | 0.117 | 0.099 | 0.42 | 0.33/0.59 | 0.36 | nan/nan | nan |
| R_hopB_bsup_neg_AMmap (R) | 0.336 | 0.549 | 0.067 | 0.067 | 0.10 | 0.51/0.82 | 0.36 | 0.88/0.76 | nan |
| R_hopB_bsup_neg_AML (R) | 0.376 | 0.643 | 0.047 | 0.063 | 0.11 | 0.62/0.85 | 0.36 | 0.88/0.76 | 4333 |
| R_hopB_bsup_neg_AL (R) | 0.378 | 0.640 | 0.050 | 0.068 | 0.15 | 0.62/0.79 | 0.36 | nan/nan | 4333 |
| NPO_b10_neg | 0.431 | 0.449 | 0.129 | 0.213 | 0.50 | 0.43/0.30 | nan | nan/nan | nan |
| KTO_b10_neg_AMmap | 0.432 | 0.491 | 0.065 | 0.193 | 0.06 | 0.44/0.80 | nan | 0.88/0.76 | nan |
| NPO_b10_neg_AMmap | 0.461 | 0.728 | 0.046 | 0.108 | 0.07 | 0.68/0.82 | nan | 0.88/0.76 | nan |
| KTO_b100_neg | 0.471 | 0.521 | 0.085 | 0.218 | 0.32 | 0.50/0.66 | nan | nan/nan | nan |
| R_hop_bsup_neg (R) | 0.471 | 0.819 | 0.108 | 0.073 | 0.33 | 0.74/0.71 | 0.75 | nan/nan | nan |

## layout=alt, n_forget=4 (B0 0.514; 20 runs per arm)

best rerouting `R_hop_bsup_neg` vs best other `KTO_b10_neg_AMmap`: -0.1422 +- 0.0228 (lower in 18/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hop_bsup_neg (R) | 0.261 | 0.422 | 0.118 | 0.049 | 0.33 | 0.18/0.39 | 0.08 | nan/nan | nan |
| R_hop_bsup_neg_AL (R) | 0.347 | 0.630 | 0.031 | 0.041 | 0.09 | 0.56/0.72 | 0.08 | nan/nan | 4330 |
| KTO_b10_neg_AMmap | 0.403 | 0.562 | 0.043 | 0.128 | 0.10 | 0.46/0.78 | nan | 0.50/0.68 | nan |
| R_hopB_bsup_neg_AL (R) | 0.410 | 0.717 | 0.074 | 0.060 | 0.17 | 0.58/0.74 | 0.17 | nan/nan | 4330 |
| R_hopB_bsup_neg_AML (R) | 0.416 | 0.734 | 0.062 | 0.058 | 0.14 | 0.61/0.85 | 0.17 | 0.50/0.68 | 4330 |
| R_hopB_bsup_neg_AMmap (R) | 0.427 | 0.716 | 0.120 | 0.073 | 0.26 | 0.50/0.82 | 0.17 | 0.50/0.68 | nan |
| NPO_b10_neg_AL | 0.430 | 0.669 | 0.025 | 0.104 | 0.12 | 0.61/0.64 | nan | nan/nan | 4330 |
| R_hopB_bsup_neg (R) | 0.431 | 0.631 | 0.257 | 0.083 | 0.57 | 0.28/0.50 | 0.17 | nan/nan | nan |
| NPO_b100_neg_AL | 0.438 | 0.710 | 0.023 | 0.092 | 0.11 | 0.65/0.67 | nan | nan/nan | 4330 |
| R_hopB_bsup_mine_neg_AMmap (R) | 0.439 | 0.764 | 0.155 | 0.062 | 0.32 | 0.50/0.75 | 0.17 | 0.50/0.68 | nan |
| NPO_b10_neg_A | 0.441 | 0.727 | 0.009 | 0.087 | 0.07 | 0.65/0.85 | nan | nan/nan | nan |
| R_hopB_bsup_mine_neg (R) | 0.442 | 0.731 | 0.248 | 0.064 | 0.54 | 0.33/0.50 | 0.17 | 0.50/0.68 | nan |
| KTO_b100_neg_AMmap | 0.451 | 0.679 | 0.019 | 0.120 | 0.08 | 0.60/0.84 | nan | 0.50/0.68 | nan |
| R_hopB_bsup_neg_AMdist (R) | 0.457 | 0.732 | 0.218 | 0.066 | 0.39 | 0.43/0.72 | 0.17 | 0.50/0.68 | nan |

## layout=pairs, n_forget=4 (B0 0.516; 20 runs per arm)

best rerouting `R_hopB_bsup_mine_neg` vs best other `PSEUDO_vote_dist`: -0.0853 +- 0.0114 (lower in 19/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hopB_bsup_mine_neg (R) | 0.203 | 0.323 | 0.083 | 0.039 | 0.43 | 0.26/0.45 | 0.16 | 0.78/0.66 | nan |
| R_hopB_bsup_neg (R) | 0.212 | 0.271 | 0.101 | 0.067 | 0.44 | 0.19/0.33 | 0.16 | nan/nan | nan |
| R_hopB_bsup_neg_AMdist (R) | 0.246 | 0.422 | 0.040 | 0.039 | 0.26 | 0.37/0.67 | 0.16 | 0.78/0.66 | nan |
| R_hopB_bsup_mine_neg_AMmap (R) | 0.264 | 0.470 | 0.036 | 0.035 | 0.19 | 0.43/0.66 | 0.16 | 0.78/0.66 | nan |
| PSEUDO_vote_dist | 0.288 | 0.501 | 0.046 | 0.043 | 0.21 | 0.47/0.70 | nan | 0.78/0.66 | nan |
| NPO_b10_neg_AMdist | 0.317 | 0.548 | 0.036 | 0.049 | 0.15 | 0.52/0.69 | nan | 0.78/0.66 | nan |
| R_hopB_bsup_neg_AMmap (R) | 0.319 | 0.537 | 0.036 | 0.057 | 0.13 | 0.48/0.81 | 0.16 | 0.78/0.66 | nan |
| R_hop_bsup_neg (R) | 0.345 | 0.580 | 0.108 | 0.061 | 0.31 | 0.45/0.53 | 0.45 | nan/nan | nan |
| R_hopB_bsup_neg_AL (R) | 0.361 | 0.614 | 0.039 | 0.062 | 0.17 | 0.59/0.69 | 0.16 | nan/nan | 4338 |
| R_hopB_bsup_neg_AML (R) | 0.367 | 0.638 | 0.029 | 0.055 | 0.13 | 0.60/0.82 | 0.16 | 0.78/0.66 | 4338 |
| KTO_b10_neg_AMmap | 0.380 | 0.571 | 0.025 | 0.101 | 0.06 | 0.52/0.81 | nan | 0.78/0.66 | nan |
| NPO_b10_neg_AL | 0.417 | 0.673 | 0.018 | 0.088 | 0.11 | 0.66/0.68 | nan | nan/nan | 4338 |
| NPO_b10_neg_AMmap | 0.419 | 0.663 | 0.019 | 0.096 | 0.07 | 0.60/0.82 | nan | 0.78/0.66 | nan |
| R_hop_bsup_neg_AL (R) | 0.425 | 0.764 | 0.029 | 0.052 | 0.09 | 0.74/0.73 | 0.45 | nan/nan | 4338 |

## layout=b3p1, n_forget=4 (B0 0.515; 20 runs per arm)

best rerouting `R_hopB_bsup_mine_neg` vs best other `PSEUDO_vote_dist`: -0.0652 +- 0.0098 (lower in 18/20); dominated by - -> **PASS**

| arm | tv | forget | retain TV | off | flips | still-forget near/far | tgt in F | vote prec/rec | local n |
|---|---|---|---|---|---|---|---|---|---|
| R_hopB_bsup_mine_neg (R) | 0.214 | 0.347 | 0.079 | 0.033 | 0.42 | 0.27/0.52 | 0.18 | 0.78/0.68 | nan |
| R_hopB_bsup_neg (R) | 0.247 | 0.287 | 0.122 | 0.080 | 0.46 | 0.21/0.38 | 0.18 | nan/nan | nan |
| R_hopB_bsup_neg_AMdist (R) | 0.251 | 0.421 | 0.055 | 0.037 | 0.26 | 0.34/0.64 | 0.18 | 0.78/0.68 | nan |
| R_hopB_bsup_mine_neg_AMmap (R) | 0.268 | 0.469 | 0.054 | 0.038 | 0.19 | 0.42/0.71 | 0.18 | 0.78/0.68 | nan |
| PSEUDO_vote_dist | 0.280 | 0.472 | 0.052 | 0.044 | 0.24 | 0.42/0.64 | nan | 0.78/0.68 | nan |
| NPO_b10_neg_AMdist | 0.300 | 0.492 | 0.063 | 0.058 | 0.17 | 0.46/0.64 | nan | 0.78/0.68 | nan |
| R_hopB_bsup_neg_AMmap (R) | 0.325 | 0.538 | 0.038 | 0.062 | 0.13 | 0.48/0.84 | 0.18 | 0.78/0.68 | nan |
| R_hopB_bsup_neg_AL (R) | 0.364 | 0.610 | 0.038 | 0.066 | 0.16 | 0.59/0.75 | 0.18 | nan/nan | 4334 |
| R_hopB_bsup_neg_AML (R) | 0.368 | 0.632 | 0.028 | 0.060 | 0.12 | 0.59/0.87 | 0.18 | 0.78/0.68 | 4334 |
| R_hop_bsup_neg (R) | 0.376 | 0.643 | 0.120 | 0.061 | 0.36 | 0.51/0.58 | 0.48 | nan/nan | nan |
| KTO_b10_neg_AMmap | 0.398 | 0.534 | 0.042 | 0.138 | 0.06 | 0.48/0.82 | nan | 0.78/0.68 | nan |
| R_hop_bsup_neg_AL (R) | 0.439 | 0.790 | 0.024 | 0.054 | 0.09 | 0.76/0.81 | 0.48 | nan/nan | 4334 |
| NPO_b10_neg_AMmap | 0.440 | 0.706 | 0.029 | 0.096 | 0.08 | 0.64/0.81 | nan | 0.78/0.68 | nan |
| KTO_b100_neg_AMmap | 0.445 | 0.577 | 0.038 | 0.164 | 0.06 | 0.53/0.82 | nan | 0.78/0.68 | nan |

## Verdict

| layout | n_forget | best rerouting | tv | best other | tv | paired diff | rerouting best? |
|---|---|---|---|---|---|---|---|
| random | 1 | R_hopB_bsup_neg_AMdist | 0.106 | NPO_b10_neg_AMdist | 0.115 | -0.0091 +- 0.0027 | **YES** |
| random | 2 | R_hopB_bsup_mine_neg_AMmap | 0.175 | NPO_b10_neg_AL | 0.200 | -0.0254 +- 0.0083 | **YES** |
| random | 4 | R_hopB_bsup_mine_neg | 0.304 | PSEUDO_vote_dist | 0.360 | -0.0561 +- 0.0182 | **YES** |
| contig | 4 | R_hopB_bsup_neg_AMdist | 0.203 | NPO_b10_neg_AMdist | 0.210 | -0.0066 +- 0.0031 | **YES** |
| alt | 4 | R_hop_bsup_neg | 0.261 | KTO_b10_neg_AMmap | 0.403 | -0.1422 +- 0.0228 | **YES** |
| pairs | 4 | R_hopB_bsup_mine_neg | 0.203 | PSEUDO_vote_dist | 0.288 | -0.0853 +- 0.0114 | **YES** |
| b3p1 | 4 | R_hopB_bsup_mine_neg | 0.214 | PSEUDO_vote_dist | 0.280 | -0.0652 +- 0.0098 | **YES** |
