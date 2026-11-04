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
- 14:15 UTC surgeons a84bf s5n4A3, a962f s6n16A1, a2c15 s7n16A3, ae387 s7n16A2 (+ a0210 a7e30 ae450 af699)
- 14:45 UTC surgeons in flight: af699->done; a6d12 s7n4A1, a51c7 s7n16A1, a9e66 s8n4A3, a6b96 s6n16A1, a85ab s8n4A2, a8b04 s8n4A1; triage abfb2 s8n16A3, aa86c s5n16A1
- 15:10 UTC in flight: a6b96 s6n16A1, a8b04 s8n4A1, a6a51 s5n16A1, ab991 s7n4A1, a576f s7n16A1, ab92f s8n16A3; triage a7df5 s8n16A2, ad49f s8n4A2
- 15:20 UTC in flight: a6a51 s5n16A1, ab991 s7n4A1, a576f s7n16A1, ad21d s8n4A2, ab141 s8n16A2; triage a1bb6 s8n4A1, a36fd s8n16A1, a8867 s9n4A3
- 15:30 UTC +surgeons a8d82 s8n4A1, adfd7 s8n16A1, afa22 s9n4A3
- 15:55 UTC in flight: a8d82 s8n4A1, afa22 s9n4A3, a65b4 s9n4A2; triage a88e0 s8n4A2, a7174 s8n16A1, afc35 s7n4A1, a09ac s9n4A1, a780e s9n16A3
- 16:05 UTC in flight: a65b4 s9n4A2, a1572 s7n4A1, ae097 s8n4A2, afded s8n16A1, a369d s9n4A1, a6f55 s9n16A3; triage aae40 s8n4A1, a2f7c s9n4A3
- 16:10 UTC +a6110 s9n4A3, a619a s8n4A1
- 16:35 UTC contig started. in flight: a1572 s7n4A1 (bg wait), a6d2c s9n16A2; triage a30c7 k1s9n4A1, a5589 k1s9n16A3, a66a0 k1s9n16A1, ac650 contig s5n4, a43db contig s5n16, a370e contig s6n4
- 16:45 UTC resumed a1572 (k1 s7n4 A1) idle >10 min
- 16:50 UTC in flight: a1572 k1s7n4A1, a6d2c k1s9n16A2, a1262 k1s9n4A1, a6b9d k1s9n16A3, a4b06 k1s9n16A1, ab418 contig s5n4, a469c contig s5n16, aeb79 contig s6n4
- 17:00 UTC triage a3a31 k1s9n16A3, aa51a contig s6n16, afee1 contig s7n4
- 17:15 UTC in flight: a1262 k1s9n4A1, a4b06 k1s9n16A1, a469c contig s5n16, a7301 k1s9n16A3, aa56c contig s6n16, a010a contig s7n4; triage ae468 contig s5n4, ac06a contig s6n4
- 17:25 UTC +triage ae876 k1s9n4A1, surgeons afc06 contig s5n4, a8acf contig s6n4
- 17:40 UTC in flight: a7301 k1s9n16A3, aa56c contig s6n16, afc06 contig s5n4, a8acf contig s6n4, ab31d k1s9n4A1; triage a3bf2 k1s9n16A1, ac204 contig s5n16, a90e6 contig s7n4
- 17:55 UTC in flight: aa56c contig s6n16, ab31d k1s9n4A1, a236b k1s9n16A1, a8010 contig s5n16, a5c3c contig s7n4; triage afa6d contig s5n4, af001 contig s6n4, a9cdc contig s7n16
