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
- (12:25 ET) +a35ff c-s6n16, a0e1b c-s8n16, a1af5 pairs s6n16
- (12:45 ET) user request: commit each finished run's patch stack to results/grid/.../<arm>/patches/ (stack.json + p*.json/npz of final version chain; no caches). Implemented copy_patches in grid_share2.py (code + eval paths) and backfilled 115 results
- (12:50 ET) in flight: a35ff c-s6n16 (bg), a1af5 pairs s6n16 (bg), a9c73 c-s8n4, a1546 c-s8n16, a2f7f c-s9n16, a78cf pairs s5n4, ac00d pairs s5n16, addad pairs s6n4
- (12:57 ET) resumed a1af5 pairs s6n16, a35ff c-s6n16
- (13:00 ET) triage a2992 c-s8n16, a36d6 pairs s5n16
- (13:10 ET) in flight: surgeons ad5fb c-s8n16, a1cb5 pairs s5n16; triage a9bc7 c-s8n4, a5e86 c-s9n16, a65d8 pairs s5n4, a3371 pairs s6n4, abe98 pairs s6n16, a5276 pairs s7n4
- (13:20 ET) in flight: ad5fb c-s8n16, a5fe9 c-s8n4, aa142 c-s9n16, a6d71 pairs s5n4, af86d pairs s6n4, a11a7 pairs s6n16, a8dcc pairs s7n4
- (13:30 ET) +af1b7 pairs s5n16, triage a399a pairs s5n4
- (13:40 ET) in flight: a5fe9 c-s8n4, aa142 c-s9n16, af86d pairs s6n4, a11a7 pairs s6n16, af1b7 pairs s5n16, a45b1 pairs s5n4; triage a18fc c-s8n16, acf25 pairs s7n4
- (13:55 ET) in flight: a45b1 pairs s5n4, a6874 c-s8n16, a3623 pairs s7n4; triage a9f90 c-s9n16, a95f5 pairs s6n4, a29ba pairs s6n16, a3990 pairs s5n16, ac831 pairs s7n16
- (14:05 ET) in flight: a6874 c-s8n16, a3623 pairs s7n4, a4ec7 c-s9n16, aeb42 pairs s5n16, a16e3 pairs s6n4, a7a74 pairs s6n16, a1084 pairs s7n16; triage a2593 pairs s5n4
- (14:15 ET) +a01a3 pairs s5n4; triage a536d pairs s6n4, abab0 pairs s6n16
- (14:30 ET) in flight: a6874 c-s8n16 (bg since 14:20), aeb42 pairs s5n16, a1084 pairs s7n16, a01a3 pairs s5n4, a3138 pairs s6n4, a68d6 pairs s6n16; triage a356d pairs s7n4, a2dea c-s9n16
- (14:40 ET) resumed a6874 c-s8n16
- (14:45 ET) in flight: a6874 c-s8n16, adfa0 c-s9n16, ad1b0 pairs s7n4; triage ae9f6 p-s5n16, ab564 p-s6n4, ade62 p-s6n16, affc0 p-s7n16, a2d45 p-s8n4
- 17:46 UTC: recorded triage pairs s5n16/s6n4/s6n16/s7n16/s8n4; contig_s8_n16 s014 no-accept. Agents: triage contig_s8_n16=a4eba0dd; surgeons pairs s5n16=aec46fe4, s6n4=a364ea1c, s6n16=aa588302, s7n16=a6bb783a, s8n4=a80f0324; still running contig_s9_n16=adfa0daa, pairs_s7_n4=ad1b0852.
- 17:50: surgeon contig_s8_n16 s015=a590b47a
- ~18:15: recorded pairs s6n16 v7, s7n4 v7, s8n4 v2, s6n4 v10, s5n16 v9 (accepted); contig s9n16, s8n16 (s016), pairs s7n16 no-accept. DONE: pairs s6n4, s6n16, s5n16 (no_open_cluster), contig s8n16 (no_accept_streak). Triage agents: contig s9n16=a462ae43, pairs s7n4=ae82251f, s8n4=a80cb22b, s7n16=a404b45c, s8n16=a9a7f1a6, s9n4=ae79f862.
- ~18:20: triage recorded pairs s7n4(c5) s8n4(c2) s7n16(c1) s8n16(c6) s9n4(c0), contig s9n16(c1). Agents: triage pairs s9n16=a9af67fe; surgeons contig s9n16=aa41f765, pairs s7n4=ab8c450f, s7n16=add2ef06, s8n4=a8c9bfc1, s8n16=ad54c047, s9n4=a531b3fd.
- ~18:25: all 50 share LLM runs launched (7 in flight, surgeon pairs s9n16=aa18b702). Filling free slots with A3_rep1 (tick --rep --slots 8-share_inflight); total sessions kept <= 8.
- ~18:45: recorded 7 accepted share surgeons (pairs s8n16 v2, s7n4 v8, s7n16 v7, s8n4 v5, s9n4 v3, s9n16 v3; contig s9n16 v15) + rep k1_s5_n4 triage c0. Triage agents for those 7; surgeon k1_s5_n4_rep1=a19e8a7d. Driver: scratchpad/tick2.sh (share tick, then rep tick with 8-share_inflight).
- ~18:50: triage recorded for 7 share runs (pairs s7n4 c4 U, s9n4 c1, s8n16 c1, contig s9n16 c3, s7n16 c3, s8n4 c1, s9n16 c4). Surgeons: contig s9n16=a43f4edb, pairs s7n4=a72b22f3, s7n16=a6230b6c, s8n4=accc5581, s8n16=ab3485e3, s9n4=a8eaeefb, s9n16=a8c0b76f; rep k1_s5_n4=a19e8a7d.
- ~19:05: recorded 7 accepted (rep k1_s5_n4 v1 -> DONE; contig s9n16 v16; pairs s8n4 v6, s9n16 v4, s7n16 v8, s7n4 v10 -> DONE, s9n4 v4) + pairs s8n16 no-accept. Triage agents for contig s9n16, pairs s7n16/s8n4/s8n16/s9n4/s9n16, rep k1_s5_n16.
- ~19:10: triage recorded (pairs s9n4 c5, s7n16 c4, s9n16 c3, s8n16 c0, s8n4 c3; contig s9n16 c0; rep k1_s5_n16 c0). Agents: triage rep k1_s6_n4=a2c97c51; surgeons contig s9n16=ad907079, pairs s7n16=a570e902, s8n4=a2743ba1, s8n16=a1e4058d, s9n4=ad578b4d, s9n16=a1227d0e, rep k1_s5_n16=ab37e0c5.
- ~19:15: rep k1_s6_n4 triage c1; surgeon=aa129d6c.
- ~19:35: recorded pairs s8n4 v7, s9n16 v5, s9n4 v5 (DONE), s8n16 v9; contig s9n16 v18 (DONE); pairs s7n16 no-accept; rep k1_s6_n4 v3 (DONE). Triage agents: pairs s7n16/s8n4/s8n16/s9n16, rep k1_s6_n16, k1_s7_n4.
- ~19:40: triage recorded (rep k1_s6_n16 c0, k1_s7_n4 c1; pairs s9n16 c5, s7n16 c2 U, s8n4 c0, s8n16 c5). Agents: triage rep k1_s7_n16=a2ba1088; surgeons pairs s7n16=ab49a043, s8n4=a7dc3b65, s8n16=a834f5a0, s9n16=adee44e2, rep k1_s6_n16=a1814480, k1_s7_n4=a3286b6c, k1_s5_n16=ab37e0c5.
- ~20:00: recorded rep k1_s5_n16 v3 (DONE), k1_s6_n16 v3, k1_s7_n4 no-accept, k1_s7_n16 triage c1; pairs s8n4 v8, s9n16 v8 (DONE), s7n16 no-accept. Agents: triage pairs s7n16=a1f62fe5, s8n4=aaa49d53, rep k1_s6_n16=aaf1af9d, k1_s7_n4=ae1c6173, k1_s8_n4=a356d342; surgeon rep k1_s7_n16=a64c938e; pairs s8n16 surgeon=a834f5a0 still running.
- ~20:05: triage recorded rep k1_s7_n4 c1, k1_s6_n16 c0, k1_s8_n4 c0; pairs s8n4 c5, s7n16 c7. Agents: triage rep k1_s8_n16=ad90a445; surgeons pairs s7n16=a5c272db, s8n4=a677200a, rep k1_s6_n16=a1c7ee0b, k1_s7_n4=a70606cb, k1_s8_n4=a5af9277.
- ~20:10: rep k1_s8_n16 triage c0, surgeon=a97c1b39. Code manifest drained except 2 running; rerunning failed cond_contig_s8_n4_R0 once by hand (log _share2/logs/rerun_contig_s8_n4_R0.log).
- ~20:30: rerun of contig_s8_n4_R0 failed with the same ValueError -> marked invalid (results/grid/contig/s8/n4/R0/INVALID.json). Code manifest otherwise complete. Triage agents for pairs s8n16 + reps k1 s7n16,s8n16,s9n4,s9n16, contig s5n4,s5n16,s6n4.
- ~20:35: 8 triages recorded; surgeons: pairs s8n16=af5ecfda; reps k1 s7n16=ae1b010e, s8n16=abc6395e, s9n4=a9011587, s9n16=aaeffc89; contig s5n4=a8f7fc3d, s5n16=aa8b7912, s6n4=a7f40c86. Only open share run: pairs_s8_n16.
- ~20:50: recorded pairs s8n16 v12; reps k1 s8n16 v3 (DONE), s9n4 v2, s7n16 v5 (DONE), s9n16 v4, contig s6n4 v4, s5n16 v4, s5n4 no-accept. 8 triages launched (pairs s8n16; reps k1 s9n4,s9n16; contig s5n4,s5n16,s6n4,s6n16,s7n4).
- NOTE: real clock check 19:16 UTC (14:16 ET); earlier '~HH:MM' stamps in this file after 17:46 ran ahead of real time. Deadline 23:00 ET = 04:00 UTC.
- 19:17Z: triage recorded (reps k1 s9n16 c0, s9n4 c0, contig s6n4 c1, s6n16 c2, s7n4 c2, s5n16 c0, s5n4 c0; pairs s8n16 c6). Surgeons: k1 s9n4=a838d84f, s9n16=ad6845ab, contig s6n4=ab5b61dc, s6n16=a1a96426, s7n4=ab01465a, s5n4=a79032b9, s5n16=ab7cfba4; pairs s8n16=af3af902.
- 19:28Z: recorded pairs s8n16 v13; reps contig s5n4 v6, s5n16 v6, k1 s9n4 v4 (DONE), no-accept contig s6n4, s7n4, s6n16, k1 s9n16. 8 triages launched (pairs s8n16; reps k1 s9n16, contig s5n4,s5n16,s6n4,s6n16,s7n4,s7n16).
- 19:29Z: 8 triages recorded; surgeons pairs s8n16=ae688f18, reps k1 s9n16=aa415758, contig s5n4=a55a890f, s5n16=abb69ad6, s6n4=af64e43f, s6n16=a9b30901, s7n4=a2045036, s7n16=a06730f2.
- 19:40Z: ALL 50 SHARE LLM RUNS DONE (pairs_s8_n16 last, v14). Reps recorded: contig s6n4 v8, s7n4 v6, s5n4 v9, s7n16 v3, s6n16 v8; no-accept contig s5n16, k1 s9n16. 8 rep triages launched (k1 s9n16; contig s5n4,s5n16,s6n4,s6n16,s7n4,s7n16,s8n4).
- 19:40Z: share complete (pairs s8n16 eval written). Surgeons for 8 reps (k1 s9n16; contig s5n4,s5n16,s6n4,s6n16,s7n4,s7n16,s8n4).
- 19:41Z: wrote draft docs/DONE_W3-2.md + share2_table (regenerate with scratchpad/done.py at the end).
- 19:52Z: reps accepted k1 s9n16 v12, contig s7n16 v4, s5n16 v11, s8n4 v1, s6n16 v10, s7n4 v8, s6n4 v9. 7 triages launched; contig s5n4 surgeon a7e8e96f still running.
- 19:58Z: 7 rep triages recorded + contig s5n4 rep no-accept. Surgeons: k1 s9n16=af0b84dc, contig s5n16=a71e5a64, s6n4=ae81a717, s6n16=abfb23d9, s7n4=a9eba9af, s7n16=a821c77c, s8n4=ad54e208; triage contig s5n4=a7815556.
