# STATE W3-2 (share 2 of PREREG.md)

Branch: local `art/grid-2` from origin/agentic-repair-toy 4fca1dd (DONE_W2b). Push to `art/grid-2` and the
session branch `claude/session-w3-2-5ff936`. Hard deadline Wed 23:00 ET.

Share 2: k1 npm 4,16 (A3, A2, A1); contig npm 4,16 (A3); pairs npm 4,16 (A3); code arms B4, R0, D1, B5 on all
30 cells; seeds 5-9. Replicates A3_rep1 in PREREG order after the share.

## Done
- Waited for DONE_W2b (07:41 ET, PASS-WITH-ISSUES, nothing blocking).
- Driver `scripts/grid_share2.py` = W3-1's `grid_share1.py` with share-2 cells, per-row LLM arms, B1b map
  (k1, contig: frozen recipe; pairs: R_hopB_bsup_mine_neg). Run ids `grid_cond_<setting>_s<seed>_n<npm>_<arm>[_rep1]`.
- Baselines B0/B1a/B1b seeds 5-9 are on origin/art/arms (results/baselines); `BASELINES=<dir>` for `table`.

## Running
- cloud_setup.sh

## Next
1. caches manifest -> jobs; code manifest -> `jobs.py run results/grid/_share2/code.jsonl --workers 3 --timeout 5400`.
2. LLM loop: `tick` -> surgeon/triage subagents (background) -> `record --token T --summary "<line>"` -> tick.
   Resume a surgeon idle > 10 min ("your background command finished; continue in the foreground").
3. When share is done: `tick --rep` until 23:00 ET. Then table, DONE_W3-2.md, push.

## Agent map (session-local; for resume messages)
- wave 1 surgeons: a5b91 s5n4A3, a3a98 s5n4A2, a692b s5n4A1, acbc1 s5n16A2, aeee0 s5n16A1, a7b06 s6n4A3, ad01f s6n4A2, a7275 s5n16A3 (started ~08:20 ET)
- wave 2 (~08:45 ET): surgeons a21ee s5n4A3, ad0ee s5n4A1, a86aa s5n16A2, acfc3 s6n4A3, aadea s6n4A1, a1b9c s6n16A3; triage abb64 s5n16A3; still running aeee0 s5n16A1
- 13:35 UTC resumed surgeon a86aa (k1 s5 n16 A2, s004) idle 11 min after backgrounded call (DEV_LOG rule)
- 13:45 UTC in flight: surgeons ae5de s5n4A3, aaddc s5n16A3, a4326 s6n16A3, a225e s5n16A1, adb02 s5n16A2, a0c5c s6n4A1, aa49d s6n16A1, a4ab2 s7n4A3. Idle check: scratchpad/idle.sh
- Done runs: k1 s5n4 A2, s6n4 A2, s5n4 A1, s6n4 A3, s6n16 A2 (evals auto-started by tick)
- 13:51 UTC resumed 4 surgeons idle ~10 min after backgrounded calls: k1 s5n4A3, s5n16A3, s6n16A3, s5n16A1
- 14:10 UTC in flight: surgeons a0210 s5n16A1, a7e30 s6n16A3, ae450 s7n4A2, af699 s7n4A1; triage ac490 s6n16A1, a5599 s7n16A3, a48f0 s7n16A2, a6982 s5n4A3
- Done runs so far: k1 s5n4 A2/A1, s5n16 A2/A3, s6n4 A2/A3/A1, s6n16 A2, s7n4 A3
