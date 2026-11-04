# Phase 1 reproduction (repo code, scripts/cond_cell.py)

Rows: 120 of 120. D = `tv_to_ideal`, Removed = 1 - `neg_mass_ratio`, Collateral = `retain_tv`; means over seeds 0-4 and npm 1/4/16/64; SEM over seeds of the npm-mean. Reference: `reference/method_map.json` `values`.

| Setting | Arm | toypref config | n | D | ± SEM | Reference D | Miss | Removed | Collateral | mass_off | D by npm (1/4/16/64) | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k4 | B0 | `B0_frozen` | 20 | 0.516 | 0.002 | 0.516 | +0.000 | 0.000 | 0.000 | 0.027 | 0.516/0.516/0.516/0.516 | 2 |
| k4 | B1a | `R_hopB_bsup_neg_AMdist` | 20 | 0.318 | 0.019 | 0.318 | -0.000 | 0.486 | 0.112 | 0.042 | 0.489/0.322/0.239/0.220 | 188 |
| k4 | B1b | `R_hopB_bsup_mine_neg` | 20 | 0.304 | 0.014 | 0.304 | +0.000 | 0.498 | 0.145 | 0.039 | 0.503/0.304/0.239/0.172 | 194 |
| alt | B0 | `B0_frozen` | 20 | 0.514 | 0.001 | 0.514 | -0.000 | 0.000 | 0.000 | 0.027 | 0.514/0.514/0.514/0.514 | 3 |
| alt | B1a | `R_hopB_bsup_neg_AMdist` | 20 | 0.457 | 0.008 | 0.457 | +0.000 | 0.268 | 0.218 | 0.066 | 0.499/0.452/0.439/0.439 | 199 |
| alt | B1b | `R_hop_bsup_neg` | 20 | 0.261 | 0.007 | 0.261 | +0.000 | 0.578 | 0.118 | 0.049 | 0.502/0.378/0.115/0.048 | 58 |

Misses above 0.025: none.
