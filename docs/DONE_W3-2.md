STATUS: PASS-WITH-ISSUES
W3-2 ran PREREG share 2 (Phase 6, seeds 5-9): k1 n4/n16 A3+A2+A1, contig and pairs n4/n16 A3, code arms B4/R0/D1/B5 on all 30 cells.
All 50 LLM runs finished with a final evaluation; 119/120 code units finished (jobs.py, 3 workers, no method/threshold/prompt changes).
Issue: cond_contig_s8_n4 R0 raised ValueError 'unknown velocity-network layout (no inp/blocks)' (toyrepair/policy.py _core) twice; marked invalid (results/grid/contig/s8/n4/R0/INVALID.json).
Mean D: B0 0.392 (n=30), B1a 0.166 (n=30), B1b 0.145 (n=30), B4 0.238 (n=30), R0 0.381 (n=29), D1 0.196 (n=30), B5 0.145 (n=30), A3 0.122 (n=30), A2 0.097 (n=10), A1 0.123 (n=10).
A3_rep1 (PREREG order k1, contig, pairs) until 23:00 ET: 9 replicate runs evaluated, mean D 0.100.
Share cost: $127.15 = LLM subagents $81.59 (triage+surgeon, from run ledgers) + main agent $45.56 (transcripts spend; includes replicate driving).
Replicate subagent cost so far: $5.82; grand total $132.97. Code arms ran on CPU (no API cost).
Isolation audits: audit_hits 0 on every recorded session. Patch stacks for each finished run are in <arm>/patches/ (no caches).
Other issues: cloud_setup.sh failed on pandas==2.2.2 (py3.13); installed scikit-learn/matplotlib/pytest/numpy==2.2.6 by hand, no repo code change.
Several surgeons stalled after backgrounded calls; resumed with the prescribed message (logged in docs/STATE_W3-2.md).
Replicate runs were started in free slots once all 50 share runs were launched (never more than 8 sessions at once).
Results: results/grid/<setting>/s<seed>/n<npm>/<arm>/ ; table results/grid/share2_table.{md,json}; driver scripts/grid_share2.py.


# Share 2 (W3-2): D per cell and arm (seeds 5-9)

| cell | B0 | B1a | B1b | B4 | R0 | D1 | B5 | A3 | A2 | A1 | A3_rep1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_k1_s5_n4 | 0.152 | 0.110 | 0.110 | 0.111 | 0.152 | 0.113 | 0.113 | 0.040 | 0.137 | 0.109 | 0.139 |
| cond_k1_s5_n16 | 0.152 | 0.086 | 0.086 | 0.093 | 0.152 | 0.114 | 0.086 | 0.111 | 0.101 | 0.101 | 0.095 |
| cond_k1_s6_n4 | 0.149 | 0.127 | 0.127 | 0.121 | 0.149 | 0.120 | 0.127 | 0.082 | 0.081 | 0.149 | 0.133 |
| cond_k1_s6_n16 | 0.149 | 0.134 | 0.134 | 0.124 | 0.149 | 0.124 | 0.127 | 0.133 | 0.099 | 0.149 | 0.085 |
| cond_k1_s7_n4 | 0.147 | 0.103 | 0.103 | 0.103 | 0.147 | 0.095 | 0.094 | 0.097 | 0.108 | 0.143 | 0.105 |
| cond_k1_s7_n16 | 0.147 | 0.086 | 0.086 | 0.088 | 0.147 | 0.095 | 0.086 | 0.051 | 0.096 | 0.083 | 0.038 |
| cond_k1_s8_n4 | 0.148 | 0.108 | 0.108 | 0.108 | 0.148 | 0.106 | 0.108 | 0.132 | 0.099 | 0.148 | 0.077 |
| cond_k1_s8_n16 | 0.148 | 0.095 | 0.095 | 0.095 | 0.148 | 0.099 | 0.095 | 0.087 | 0.037 | 0.083 | 0.087 |
| cond_k1_s9_n4 | 0.148 | 0.106 | 0.106 | 0.148 | 0.148 | 0.148 | 0.148 | 0.073 | 0.105 | 0.148 | 0.137 |
| cond_k1_s9_n16 | 0.148 | 0.105 | 0.105 | 0.112 | 0.148 | 0.110 | 0.105 | 0.086 | 0.106 | 0.113 | - |
| cond_contig_s5_n4 | 0.512 | 0.261 | 0.261 | 0.266 | 0.512 | 0.258 | 0.261 | 0.101 | - | - | - |
| cond_contig_s5_n16 | 0.512 | 0.134 | 0.134 | 0.142 | 0.512 | 0.134 | 0.132 | 0.103 | - | - | - |
| cond_contig_s6_n4 | 0.512 | 0.221 | 0.221 | 0.230 | 0.512 | 0.203 | 0.229 | 0.143 | - | - | - |
| cond_contig_s6_n16 | 0.512 | 0.148 | 0.148 | 0.179 | 0.512 | 0.162 | 0.148 | 0.158 | - | - | - |
| cond_contig_s7_n4 | 0.518 | 0.239 | 0.239 | 0.239 | 0.518 | 0.239 | 0.239 | 0.104 | - | - | - |
| cond_contig_s7_n16 | 0.518 | 0.144 | 0.144 | 0.146 | 0.518 | 0.168 | 0.144 | 0.075 | - | - | - |
| cond_contig_s8_n4 | 0.518 | 0.253 | 0.253 | 0.230 | - | 0.224 | 0.220 | 0.303 | - | - | - |
| cond_contig_s8_n16 | 0.518 | 0.137 | 0.137 | 0.157 | 0.518 | 0.137 | 0.137 | 0.127 | - | - | - |
| cond_contig_s9_n4 | 0.506 | 0.175 | 0.175 | 0.183 | 0.506 | 0.183 | 0.175 | 0.159 | - | - | - |
| cond_contig_s9_n16 | 0.506 | 0.115 | 0.115 | 0.159 | 0.506 | 0.147 | 0.115 | 0.137 | - | - | - |
| cond_pairs_s5_n4 | 0.510 | 0.272 | 0.203 | 0.510 | 0.510 | 0.314 | 0.203 | 0.104 | - | - | - |
| cond_pairs_s5_n16 | 0.510 | 0.129 | 0.074 | 0.510 | 0.510 | 0.510 | 0.074 | 0.121 | - | - | - |
| cond_pairs_s6_n4 | 0.516 | 0.199 | 0.178 | 0.516 | 0.516 | 0.182 | 0.178 | 0.114 | - | - | - |
| cond_pairs_s6_n16 | 0.516 | 0.185 | 0.142 | 0.516 | 0.516 | 0.215 | 0.142 | 0.104 | - | - | - |
| cond_pairs_s7_n4 | 0.517 | 0.272 | 0.147 | 0.308 | 0.517 | 0.305 | 0.147 | 0.160 | - | - | - |
| cond_pairs_s7_n16 | 0.517 | 0.207 | 0.147 | 0.264 | 0.447 | 0.216 | 0.147 | 0.269 | - | - | - |
| cond_pairs_s8_n4 | 0.514 | 0.245 | 0.209 | 0.514 | 0.514 | 0.514 | 0.209 | 0.090 | - | - | - |
| cond_pairs_s8_n16 | 0.514 | 0.166 | 0.100 | 0.514 | 0.514 | 0.227 | 0.100 | 0.073 | - | - | - |
| cond_pairs_s9_n4 | 0.508 | 0.242 | 0.177 | 0.257 | 0.400 | 0.249 | 0.177 | 0.191 | - | - | - |
| cond_pairs_s9_n16 | 0.508 | 0.162 | 0.093 | 0.196 | 0.508 | 0.179 | 0.093 | 0.140 | - | - | - |

| arm | cells | mean D | mean Removed | mean Collateral |
|---|---|---|---|---|
| B0 | 30 | 0.392 | 0.000 | 0.000 |
| B1a | 30 | 0.166 | 0.735 | 0.081 |
| B1b | 30 | 0.145 | 0.768 | 0.081 |
| B4 | 30 | 0.238 | 0.552 | 0.068 |
| R0 | 29 | 0.381 | 0.013 | 0.006 |
| D1 | 30 | 0.196 | 0.646 | 0.075 |
| B5 | 30 | 0.145 | 0.755 | 0.078 |
| A3 | 30 | 0.122 | 0.889 | 0.073 |
| A2 | 10 | 0.097 | 0.921 | 0.071 |
| A1 | 10 | 0.123 | 0.485 | 0.052 |
| A3_rep1 | 9 | 0.100 | 0.927 | 0.075 |
