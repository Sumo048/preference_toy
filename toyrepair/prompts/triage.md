# Triage: static instructions

You choose the next failure cluster to repair in a policy-repair run.

Input below: the atlas (one card per open cluster of logged bad actions) and the ledger summary (recent repair attempts and accepted edits). Card fields:

- `r_ref` / `r_cur`: residual basin mass under the original and the current policy (share of fresh decodes at the members' conditions that land in their bad basins). Higher means more bad behaviour left.
- `n`: member count. `c`: 5th to 95th percentile of member conditions. `coh`: cohesion (mean affinity within the cluster, 0 to 1).
- `k_commit`: earliest decode step after which a renoised branch rarely escapes; `class` R (a scripted primitive rescues at least half of the members) or U (none does) or ? (not yet measured).
- `cooldown`: clusters that failed recently are not eligible and are not listed.

Prefer clusters with large remaining mass, class R, and a clear structure. Avoid clusters that failed repeatedly in the ledger.

Output: exactly one line of JSON and nothing else:

{"cluster": <id>, "expected_class": "R" | "U", "hypothesis": "<one line>"}
