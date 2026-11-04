# Phase 2 exit: hand-written graft on uncond base_k2 (W2a, by hand, no subagent)

Run `p2exit_base_k2_A2` (arm A2, uncond base_k2, seed 0, npm 4), full path:
`orchestrate init` -> `next` (triage token; triage done by hand, recorded) -> `next` (surgeon token)
-> `toyrepair-tool prompt / sandbox / graft_gated_path / quick_check / submit_edit` -> `orchestrate record`.

- Atlas v0: 2 open clusters (n=4 each). Target: cluster 1 (r_ref 0.127, k_commit 12, class R, best push(d0,2r,k8,h3)).
- Sandbox: logistic probe on `out_act` at step 8 for "decode ends in the cluster's basins"
  (8,192 fresh noises; fires 12.1%, precision 0.988, recall 0.98). Basin radius r = 0.394.
- Graft: layer `out`, 1 unit, `act linear`, w_in = 0, b_in = 1 (constant unit), w_out = -(2r)·d0/(3·dt)
  = (-5.246, 0); detector = probe on `out_act`, threshold 0, k in [8, 11). This is push(d0,2r,k8,h3) inside the net.
- quick_check: still_bad 0.000, residual 0.036 (from 0.127), drift 0.120.
- Gate (cluster scope, N_G 16,384): ACCEPT, all six tests pass. Target mass 0.126 -> 0.042 (fall 66%,
  CI [0.079, 0.088]); D 0.273 -> 0.229 (dD UB -0.040); Collateral 0.087 (UB 0.091); between rise UB 0.0004.
- Bug found and fixed on the way: the CLI renamed a patch after the edit, so a graft's id (part of the
  version hash) changed and `rebuild` failed its sha check (candidate v1). The CLI now passes the patch
  id into the edit call.

`ledger.jsonl` is the run's ledger (copied from `runs/repair/p2exit_base_k2_A2/`).
