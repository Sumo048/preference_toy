# Share 3 run notes (W3-3)

- 13:05 UTC: resume message ("Your background command finished; continue in the foreground.") sent to s002 of
  g3_cond_k4_s6_n16_A3-nogate, g3_cond_k4_s5_n16_A3-nobranch, g3_cond_k4_s6_n16_A3-nobranch. Each was idle
  >10 min after a backgrounded `targeted_train` finished (DEV_LOG W2b rule). No limit or prompt change.
- 12:55 UTC: Claude Code flagged s002 of g3_cond_k4_s6_n16_A3-nobranch ("Security Weaken"). Main agent reviewed
  its commands: `prompt`, a `sandbox` script that printed the negatives and `os.listdir('.')` of its own export
  dir, `targeted_train`, `dossier`. All are allowed `toyrepair-tool` calls; no oracle path. Run kept; the
  transcript audit at `record` decides.
- 13:20 UTC: second resume message to s002 of g3_cond_k4_s6_n16_A3-nogate (idle again since 13:05).
- 13:35 UTC: Claude Code flagged g3_cond_k4_s7_n16_A3-nogate surgeon ('Create RCE Surface'); its commands were only `prompt` and `targeted_train`. Run kept.
- 13:45 UTC: resume message to the k4s7 nogate surgeon (idle 9-10 min after a backgrounded targeted_train).
- ~13:40 UTC: resume message to the k4s8 nobranch s002 surgeon (idle 10 min after backgrounded targeted_train; earlier 'Security Weaken' flag reviewed: only prompt/targeted_train/dossier calls). Note: earlier note times are approximate.
- ~13:50 UTC: resume message to the k4s6 nobranch surgeon (idle 10 min).
- 14:12 UTC: resume message to the k4s6 nobranch surgeon (idle 10 min).
- 14:50 UTC: resume message to the k4s8 nogate surgeon (idle 10 min).
- Observed (W3-3, no change made): `targeted_train --params grafts` retrains every graft of the current version, not only the newest one (k4s9 nobranch s012: v13 cancelled v6's earlier graft, target_fall -0.111). Listed for the report.
- 14:58 UTC: resume message to the b3p1 s7 n16 A3 surgeon (idle 10 min).
- 15:15 UTC: resume message to the k4s8 nogate surgeon s012 (idle 10 min).
- 15:15 UTC: resume message to the b3p1 s8 n16 A3 surgeon (idle 10 min).
- 15:31 UTC: resume message to the k4 s5 n64 A3 surgeon (idle 10 min).
- 15:31 UTC: resume message to the b3p1 s9 n4 A3 surgeon (idle 11 min).
- 15:44 UTC: second resume message to the b3p1 s9 n4 A3 surgeon s006.
- 15:47 UTC: resume message to the k4 s6 n64 A3 surgeon (idle 11 min).
- 16:04 UTC: resume message to the k4 s5 n64 A3 surgeon (idle 10 min).
- 16:05 UTC: resume message to the b3p1 s9 n4 A3 surgeon s008 (idle 10 min).
- 16:15 UTC: resume message to the k4 s6 n64 A3 surgeon s004 (idle 10 min).
- 16:24 UTC: resume message to the k4 s7 n64 A3 surgeon s002 (idle 10 min).
- 16:44 UTC g3_cond_k4_s8_n64_A3 surgeon flagged 'Create RCE Surface'; reviewed: only toyrepair-tool prompt/targeted_train/dossier calls. Benign.
