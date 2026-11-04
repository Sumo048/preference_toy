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
- 18:05 UTC in flight: ab31d k1s9n4A1, a236b k1s9n16A1, a8010 contig s5n16, a5c3c contig s7n4, ab690 contig s5n4, abb5c contig s6n4, a99a9 contig s7n16; triage a4dea contig s6n16
- 18:10 UTC +ae432 contig s6n16
- 18:30 UTC in flight: ae432 contig s6n16 surgeon; triage acb38 c-s5n16, a13e5 c-s6n4, ad91b c-s7n16, af742 k1s9n16A1, a160a c-s8n4, a0a71 c-s8n16, a5d3f c-s9n4
- 18:40 UTC in flight: ac942 k1s9n16A1, a8acf c-s5n16, a530e c-s6n4, acfe5 c-s7n16, a4a25 c-s8n4, a1447 c-s8n16, a8442 c-s9n4; triage ac791 c-s6n16
- 18:45 UTC +a7fb1 c-s6n16
- 18:55 UTC in flight: ac942 k1s9n16A1, a8acf c-s5n16 (bg wait 18:52), a1447 c-s8n16 (bg wait 18:52); triage a6275 c-s6n4, a4786 c-s6n16, a4707 c-s7n16, a0bc9 c-s8n4, a1d0e c-s9n4
- 19:05 UTC resumed a8acf (c-s5n16) and a1447 (c-s8n16), idle >10 min
- 19:10 UTC in flight: a8acf c-s5n16, a1447 c-s8n16, a9c3c c-s6n4, aebd7 c-s6n16, ade54 c-s7n16, a7577 c-s8n4, a7173 c-s9n4; triage ae2db k1s9n16A1
- 19:15 UTC +ada4a k1s9n16A1
- 19:30 UTC in flight: a8acf c-s5n16 (bg), a1447 c-s8n16 (bg), ada4a k1s9n16A1; triage a0ac5 c-s6n4, ab0a3 c-s6n16, ada34 c-s7n16, af40a c-s8n4, aa42d c-s9n4
- 19:40 UTC resumed a8acf and a1447 again
- 19:45 UTC in flight: a8acf c-s5n16, a1447 c-s8n16, add78 c-s6n4, af6ad c-s6n16, a0849 c-s7n16, a800c c-s8n4, a779a c-s9n4; triage acd24 k1s9n16A1
- 19:55 UTC +ae6cd k1s9n16A1, triage a3b34 c-s8n16
- 20:10 UTC code unit cond_contig_s8_n4_R0 failed (ValueError unknown velocity-network layout in policy._core); retrying once per W3 rule
- 20:15 UTC pairs started. in flight: ae6cd k1s9n16A1, ad710 c-s8n16; triage a2343 c-s5n16, ab652 c-s8n4, a1f01 c-s9n4, a70d4 c-s9n16, ae4ce pairs s5n4, a5bd4 c-s6n16
- 20:30 UTC in flight: ad710 c-s8n16, a1709 c-s5n16, afade c-s6n16, a904c c-s8n4, a4a5b c-s9n4, a8cdf c-s9n16, a8afa pairs s5n4; triage a5053 k1s9n16A1
- (11:20 ET) +af489 k1s9n16A1. NOTE earlier 'UTC' stamps in this file are approximate
- (11:35 ET) in flight: a8cdf c-s9n16 (bg wait), af489 k1s9n16A1; triage a3ce8 c-s5n16, ae314 c-s8n4, a123a c-s8n16, a7d69 c-s9n4, af32e pairs s5n4, aef65 c-s6n16
- (11:45 ET) resumed a8cdf c-s9n16
- (11:50 ET) k1 all 30 LLM runs done. in flight: a8cdf c-s9n16, a6a38 c-s5n16, a1f16 c-s6n16, a4b07 c-s8n4, a4332 c-s8n16, a3df3 c-s9n4, acd46 pairs s5n4; triage a3b96 pairs s5n16
- (11:55 ET) +aa772 pairs s5n16
- (12:05 ET) in flight: a8cdf c-s9n16 (bg), a1f16 c-s6n16, a4b07 c-s8n4 (bg), a4332 c-s8n16, a3df3 c-s9n4, aa772 pairs s5n16; triage a857a pairs s6n4, a2687 pairs s5n4
- (12:15 ET) resumed a8cdf c-s9n16 and a4b07 c-s8n4
- (12:20 ET) in flight: a8cdf c-s9n16, a4b07 c-s8n4, aa772 pairs s5n16, ac0cb pairs s5n4, a570a pairs s6n4; triage a6dbf c-s6n16, a37b2 c-s8n16, a8793 pairs s6n16
