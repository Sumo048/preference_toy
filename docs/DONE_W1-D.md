STATUS: PASS-WITH-ISSUES — RealBackend is written against W0 signatures but untested (W1-A1/A2/B modules were stubs); cloud_setup.sh fails on a fresh VM at pandas==2.2.2 (no py3.13 wheel; I installed pandas 2.2.3 binary)

# W1-D summary (branch art/llm)

1. Works: `toyrepair-tool` (alphanumeric session tokens; 20 calls / 3 submissions; per-arm command sets A1, A2, A3, A3-nobranch, A3-sonnet, A3-nogate; triage tokens run only `triage-input`; scratch-only path checks incl. symlinks; a `tool` ledger record for every call, refused ones too; row and token caps that state how many rows were cut; `prompt` = static, then dossier). Dossier builder with static ≤ 4,000 and variable ≤ 6,000 tokens. `orchestrate init/next/record/status`, with the four stop rules, a 2-iteration cool-down, the triage fallback, per-run file locks (tested with 8 runs in parallel) and invalidation on an audit hit. Transcript reader: usage, cost, context check, isolation audit, chars/token. Static prompts are in `toyrepair/prompts/`. MockBackend fakes every tool.
2. Tests: `tests/toyrepair/test_llm_harness.py`, 22 passed; with `tests/toyrepair` and `tests/infra`, 39 passed.
3. Chars per token, measured on 2 real surgeon transcripts: 1.95 and 1.84 (the 6.7k-char prompt alone: 1.81). `dossier.CHARS_PER_TOKEN` = 1.8, not W0's 3.5. The A3 static part is 2,481 tokens without EDIT_REFERENCE, so EDIT_REFERENCE gets ≤ 1,500 tokens = 2,700 chars.
4. End to end on mock runs, with real subagents (the project hook fired: it blocked a surgeon call containing `+`):
   - triage: 4.9k tokens, $0.014 and $0.009; valid JSON both times.
   - surgeon, run 1: 15 requests, 13 calls, 2 submissions, accepted. In 30 / cache write 13.9k / cache read 151k / out 3.7k tokens, $0.173.
   - surgeon, run 2 (final prompt): 13 requests, 12 calls, 2 submissions, accepted. In 26 / cache write 9.8k / cache read 114k / out 3.7k tokens, $0.145; peak context 12.2k.
5. Fixed from the E2E runs: anchor `local+map` is spelled `local_map` on the CLI (the hook allows no `+`). The surgeon may write more than one final line, so `record` keeps the RESULT line.
6. Open interface requests: `docs/INTERFACE_REQUESTS_W1-D.md`. Key ones:
   - EDIT_REFERENCE ≤ 2,700 chars.
   - The sandbox audit hook must import torch before blocking subprocess: torch runs ldconfig.
   - `Run.scratch` sits inside the mode-700 repo, so the `sandbox` user can't reach it.
   - `flowstate.inspect` signature.
   - The ledger summary must use the `4_other` test label.
7. Cost: main agent about $6, subagents $0.34.
8. For W2a: `python -m toyrepair.orchestrate init --run <distinctive id> --arm A1 ... [--backend mock]`, then loop `next`, invoke the printed agent with "Run <id>. Token <t>. Start with `toyrepair-tool prompt --token <t>`.", then `record --token <t>`. Run state is in `runs/repair/<id>/` (run.json, versions.json, sessions/, atlas/, branch/).
