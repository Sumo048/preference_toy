# Cloud session prompts: agentic repair on the toypref toy

Companion to `docs/PLAN.md` (read its Amendment 1). Repo: `Sumo048/preference_toy`, default branch `main`.

Each session gets a one-line prompt:

> You are session W1-A1. Read docs/SESSION_PROMPTS.md and do section W1-A1 exactly.

## Aggressive mode

The goal is to spend the $250 credit on useful work before it expires: November 4, 11:59 PM PT (Thursday 2:59 AM in Pittsburgh).

Start every session now. A session whose inputs are not ready waits for them through git (rule 4), then starts work by itself. No one needs to start the later waves by hand.

| Session | Waits for | Main model | Work time box | Hard deadline (ET) |
|---|---|---|---|---|
| W0 | nothing (already running) | Opus 5.5 | 1.5 h | — |
| W1-A1, W1-A2, W1-B, W1-C, W1-D | `DONE_W0` | Opus 5.5 | 5 h | Wed 08:00 |
| E1, E2, E3 | nothing | Opus 5.5 | 6 h | Wed 06:00 |
| W2a | all five `DONE_W1-*` | Opus 5.5 | 4 h | Wed 12:00 |
| W2b | `DONE_W2a` | Opus 5.5 | 5 h | Wed 17:00 |
| W3-1, W3-2, W3-3 | `DONE_W2b` | Opus 5.5 | until deadline | Wed 23:00 |
| W4 | all three `DONE_W3-*` | Opus 5.5 | 2 h | Thu 02:30 |

After the credit ends, running sessions count against your Max plan limits. Archive any session you don't want to continue on those limits.

## Setup (done once)

1. `main` is the default branch on GitHub.
2. The files of `agentic_repair_files.zip` are committed on `main`: `docs/PLAN.md`, `docs/SESSION_PROMPTS.md`, `.claude/agents/surgeon.md`, `surgeon-sonnet.md`, `triage.md`, and `tools/surgeon_hook.py`.
3. The Claude GitHub App is installed on `Sumo048/preference_toy`, so sessions can push branches.
4. W0's role hook is on `main`: `.claude/settings.json` and `tools/role_hook_dispatch.py`. The `hooks:` field in agent definitions does not run in cloud sessions, and a session reads hooks only when it starts.
5. Sessions are plain cloud sessions: claude.ai/code (repo `Sumo048/preference_toy`, branch `main`, Default environment) or `claude --cloud "<prompt>"`. Not inside a Project, a Routine, or Remote Control.

## Rules for every session

1. Read `docs/PLAN.md` in full, including Amendment 1, before any work.
2. **Wait first.** If your section has a "Waits for" line, run the wait loop (rule 4) before anything else. Read the first line of each input `DONE` file. If one says `STATUS: FAIL`, read its reason. Work around it if you can within your time box and say so; otherwise write your own `DONE` file with `STATUS: FAIL` and stop.
3. Then check out your branch as your section says, and run `bash scripts/cloud_setup.sh` (W0 writes it; E sessions, see their section).
4. **Wait loop.** Run this as one Bash call with a 600,000 ms timeout. List the `DONE` file names you wait for. It searches every branch, because some sessions can push only to their own `claude/...` branch. Exit 0 means ready: it prints each file with the branch that holds it, newest first. Those printed branches are your inputs. Exit 124 means not yet: run it again. Each call adds only one short result to your context.

   ```bash
   timeout 580 bash -c 'while true; do git fetch -q origin "+refs/heads/*:refs/remotes/origin/*" 2>/dev/null; ok=1; out=; for f in "$@"; do hit=; for b in $(git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/remotes/origin); do git cat-file -e "$b:docs/$f" 2>/dev/null && { hit=$b; break; }; done; if [ -n "$hit" ]; then out="$out$f $hit\n"; else ok=0; fi; done; [ $ok = 1 ] && { printf "$out"; exit 0; }; sleep 60; done' _ DONE_X.md DONE_Y.md
   ```

5. **DONE file.** When you finish, or at your deadline, commit `docs/DONE_<your session>.md` and push.

   **Pushing.** Push to the branch your section names: `git push origin HEAD:refs/heads/<name>`. If your session lets you push only to its own `claude/...` branch, push there instead and say so in your `DONE` file. Wait loops find `DONE` files on any branch.

   The `DONE` file's first line is `STATUS: PASS`, `STATUS: PASS-WITH-ISSUES — <issues>`, or `STATUS: FAIL — <reason>`. The rest is your summary (rule 14). Later sessions start from this file.
6. Keep `docs/STATE_<your session>.md` current: what is done, what runs, what is next. After a context compaction, read it again before you act.
7. Work only on your branch. Edit only the files your section gives you. Read other files freely.
8. If you need a change to another session's interface, write it in `docs/INTERFACE_REQUESTS_<your session>.md` and use a local adapter until the merge.
9. CPU jobs go through `scripts/jobs.py` (W0 writes it): a resumable manifest runner with 4 workers, one torch thread each. No single job unit runs longer than 20 minutes. Results are written per unit, so finished units are skipped on restart.
10. While jobs or subagents run, do not end your turn. Poll with `sleep 90` followed by `python scripts/jobs.py ensure <manifest>`, which restarts dead workers. An idle session pauses its VM and stops its jobs. Background commands stop after about 30 minutes, which is why job units are short.
11. Run at most 8 surgeon subagents at a time.
12. At each commit, run `python -m toyrepair.transcripts spend --session <your session>`. It appends your main-agent cost to `results/spend.jsonl`. Commit at each milestone. Push your branch at least once an hour.
13. Use Sonnet subagents for code search, log summaries, and plots. Do not open pull requests. Do not push to `main`. Do not edit `.claude/`, `tools/surgeon_hook.py`, or `tools/role_hook_dispatch.py`.
14. End with a summary of 15 lines or fewer: what works, key numbers, what failed, open interface requests, measured cost, and anything the next session must know. Put the same summary in your `DONE` file.

## W0

Scope: Phase 0 only. Time box: 1.5 hours.

Branch: `git checkout -b agentic-repair-toy origin/main`

1. **Environment.** Record CPU count, RAM, Python and torch versions. Write `scripts/cloud_setup.sh`. It must do all of these and finish in under 5 minutes on a fresh VM:
   - run `pip install -e .` (add a `pyproject.toml` that installs `toyrepair` and the `toyrepair-tool` script)
   - install `requirements.txt` plus scikit-learn and pytest
   - check `ckpt/SHA256SUMS`, then copy `ckpt/cond_base` and `ckpt/base` into `runs/`, as `README.md` shows
   - create the unprivileged user `sandbox`, and set the repo directory to mode 700, owned by the user that runs Claude Code, so `sandbox` cannot read it (PLAN.md Section 6.6). Check that Claude Code itself can still read and write the repo afterwards.
2. **Run paths.** Make `toyrepair/__init__.py` set `TOYPREF_RUNS` to `<repo>/runs` when it is unset.
3. **Survey.** Write `docs/SURVEY.md`, one page. Check every entry point and the network description in PLAN.md Section 5.1. Record the CPU time of one frozen-recipe arm run and one `compute` call. Do not retrain π_ref: `ckpt/` has it. Record which `compute` keys are Removed and Collateral.
4. **Reproduce** B0 and `R_hopB_bsup_neg_AMdist` on `scripts/cond_cell.py --seed 0 --n_forget 4 --n_neg 4`. Compare with the k4 seeds 0–4 means in `reference/method_map.json`: 0.515 (B0) and 0.318 (frozen recipe). A single cell will differ from a mean; report the value.
5. **Scaffold.** Write the package of PLAN.md Section 6.11: every module with typed signatures, dataclasses, and docstrings that state inputs, outputs, and units, with `NotImplementedError` bodies. Five parallel sessions build against these signatures, so make them exact. Implement `toyrepair/cell.py` and the seed logic of `pools.py` fully, with tests. Add the console script `toyrepair-tool`, with `--help` only.
6. **Infra.** Write `scripts/jobs.py` (rule 9) with `run`, `ensure`, and `status`. Write a minimal `toyrepair/transcripts.py` with the `spend` command (rule 12).
7. **Isolation check.** Invoke the `surgeon` subagent and ask it to run `cat toyrepair/cell.py`, then `toyrepair-tool --help`. The first must be blocked and the second must run. Invoke `surgeon-sonnet` and `triage` once each the same way. Find the subagent transcripts on disk and confirm they hold token-usage fields. Record the path pattern and field names in SURVEY.md. If the hook does not block, use `STATUS: PASS-WITH-ISSUES — audit-only isolation` (Amendment 1).
8. Commit, write `docs/DONE_W0.md` (rule 5), and push `agentic-repair-toy`.

## W1-A1: simulation-side harness

Waits for: `DONE_W0.md`

Branch: `git checkout -b art/harness-sim <branch printed for DONE_W0.md>`

You own: `toyrepair/policy.py`, `flowstate.py`, `miner.py`, `quick_check.py`, `pools.py` (beyond W0's seeds), and their tests. W0's signatures are the contract.

1. Policy server and flow states (Sections 6.1, 5.4). Test: restore at every step and continue gives bit-identical actions on 10,000 noises.
2. Miner, failure cards, and atlas, with token caps (Section 6.2). Log oracle purity to a CSV only, through W1-A2's oracle interface (use a local adapter until the merge).
3. Branch tool: renoise, Wilson top-up, k_commit, primitives, failure class, and the 40-row table (Section 6.3). Inspect tool (Section 6.4).
4. quick_check (Section 6.8).

Smoke-test each component on unconditional `base_k2` first. Report: cluster counts on `k4` and `alt`, seeds 0–4, npm 4 and 16; k_commit and failure class of the largest cluster in each of those cells.

## W1-A2: gate and evaluation

Waits for: `DONE_W0.md`

Branch: `git checkout -b art/harness-gate <branch printed for DONE_W0.md>`

You own: `toyrepair/oracle/`, `gate.py`, `ledger.py`, `budget.py`, `eval_final.py`, and their tests. W0's signatures are the contract.

1. Oracle and final evaluation (Section 4, rule 8): wrap `CondMixture.assign` and `experiment.Bank`/`compute`. Only `gate.py` and `eval_final.py` import `oracle/`. Expose a purity function for the miner.
2. The per-run ledger (Section 6.9) and the budget counter (decodes, gate checks, cost).
3. Gate with both scopes, the six tests, and the paired bootstrap (Section 10). For test 6, call the unit-test hook that W1-B writes; until the merge, stub it with shape and NaN checks. Decode versions through W0's `policy` signatures, with a local adapter until the merge.
4. Gate reproducibility and power check (Section 10.3). Report the N_G you need and the gate's CPU time per check.

## W1-B: edit API and training

Waits for: `DONE_W0.md`

Branch: `git checkout -b art/edits <branch printed for DONE_W0.md>`

You own: `toyrepair/edits.py`, `patches.py`, `train.py`, `sandbox.py`, the gate's unit-test hook (a function in `edits.py`), `docs/EDIT_REFERENCE.md`, and their tests. W0's signatures are the contract.

1. `graft_gated_path` with a hard gate (Section 6.7). Tests: graft identity on 10,000 reference flow states with the detector off; a zero-output graft gives bit-identical decodes; the API rejects a detector that reads anything other than network inputs and activations. Layers to support: `inp`, `blocks.{0,1,2}.fc1`, `blocks.{0,1,2}.fc2`, `out`, and the `cond` vector as a detector input.
2. `rank1`, `ablate_direction`, and `set_units`, with tests.
3. Patch files and rebuild from π_ref plus a patch stack. Test: the rebuild equals the live version bit for bit.
4. `targeted_train` over the repo's losses (rerouting with each target rule and support, NPO, KTO) and its anchors. Test: `targeted_train` configured as `R_hopB_bsup_neg_AMdist` on seed 0, n_forget 4, npm 4 matches `scripts/cond_cell.py` for that arm.
5. Sandbox (Section 6.6): the export (weights, a self-contained copy of `CondVelocityNet` with no toypref import, negatives, basins, a decode function), the `sandbox` user, the audit hook, the runner, and the 2,000-token output cap. Tests: as user `sandbox`, reading `src/toypref/cond/task.py` fails; importing `toypref` and `toyrepair.oracle` fails; subprocess and socket fail; a least-squares fit of a graft's `w_out` runs.
6. `EDIT_REFERENCE.md`: the edit calls and the sandbox, with inputs, units, and one example each, in 1,500 tokens or fewer. It goes into the surgeon's static prompt.

## W1-C: reproduction and code arms

Waits for: `DONE_W0.md`

Branch: `git checkout -b art/arms <branch printed for DONE_W0.md>`

You own: `toyrepair/arms/`, `scripts/run_arm.py`, `scripts/phase1_repro.py`, `results/phase1/`, `results/baselines/`, `docs/DEV_LOG.md`, and their tests.

1. **Phase 1 reproduction** with the repo's code directly (`scripts/cond_cell.py`), through `scripts/jobs.py`. Run B0, B1a (`R_hopB_bsup_neg_AMdist`), and B1b on `k4` (n_forget 4, random) and `alt` (n_forget 4, layout alt), seeds 0–4, npm 1, 4, 16, 64. Compare against `reference/method_map.json` (`values` = seeds 0–4, mean over npm): k4 B0 0.515, B1a 0.318, B1b `R_hopB_bsup_mine_neg` 0.304; alt B0 0.515, B1a 0.457, B1b `R_hop_bsup_neg` 0.261. A miss of more than 0.025 is an issue to report, not a stop (Amendment 1).
2. **Arms** B0, B1a, B1b, B4, B5, R0, and D1 (Section 9.2), behind one entry point: `scripts/run_arm.py --arm <name> --setting <s> --seed <n> --npm <m>`. B5 configs per setting (top 3 on seeds 0–4):
   - k1: `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AMdist`, `PSEUDO_vote_dist`
   - k2: `R_hopB_bsup_mine_neg_AMmap`, `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AL`
   - k4, pairs, b3p1: `R_hopB_bsup_mine_neg`, `R_hopB_bsup_neg_AMdist`, `R_hopB_bsup_mine_neg_AMmap`
   - contig: `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AMdist`, `PSEUDO_vote_dist`
   - alt: `R_hop_bsup_neg`, `KTO_b10_neg_AMmap`, `NPO_b10_neg_AL`
3. B4, B5, R0, and D1 need the branch tool (W1-A1), the gate (W1-A2), and `targeted_train` (W1-B). Write them against W0's signatures and test them with small fakes. W2a runs them for real.
4. Fix the D1 variant list (3 variants). Write it in `DEV_LOG.md` with the reason.
5. When steps 1–4 are done, run B0, B1a, and B1b with the repo's code on all 140 conditional cells of seeds 5–9 (7 settings × 4 npm × 5 seeds), into `results/baselines/`. Phase 6 needs them, and they cost only CPU.

Report the reproduction table and the CPU time per arm run.

## W1-D: LLM roles, CLI, and orchestrator

Waits for: `DONE_W0.md`

Branch: `git checkout -b art/llm <branch printed for DONE_W0.md>`

You own: `toyrepair/cli.py`, `dossier.py`, `orchestrate.py`, `transcripts.py` (beyond W0's `spend`), `toyrepair/prompts/` (the static prompt texts), and their tests.

1. `toyrepair-tool`: session tokens; per-session limits (20 calls, 3 submissions); per-arm command sets (A1, A2, A3, A3-nobranch, A3-nogate); path checks that keep every path argument inside the run's scratch directory; a ledger record for every call; row caps that state the number of cut rows. Commands include `prompt` (static part, then the dossier) and `triage-input`.
2. Dossier builder with both caps (Section 7.3). Count tokens with a chars-per-token ratio measured once on a real transcript.
3. `orchestrate next`, `record`, and `status` (Sections 6.10 and 7.4): stop rules, cool-down, and the triage fallback. `status` prints one line per run. Support up to 8 concurrent sessions.
4. Transcript reader: usage and cost per subagent with the Section 2 price table, plus the isolation audit (Section 7.5).
5. Static prompt texts in `toyrepair/prompts/`: the surgeon's (4,000 tokens or fewer: the policy interface from SURVEY.md, the CLI reference, a placeholder for `EDIT_REFERENCE.md`, and the session rules), and the triage's (JSON output only). No ring geometry and no mode count.
6. A mock harness that fakes tool outputs. Run one real `surgeon` session and one `triage` call against the mock to test the mechanics end to end. Report their tokens and cost.

## E1, E2, E3: exploration (start now, no wait)

These sessions explore by hand what a surgeon could do. Each one tests one family of edits. Their results are outside the claims (Amendment 1). They give W2b ideas for the Phase 4 tools and give the report a sense of the headroom.

Branch: `git checkout -b art/explore-EN origin/main`, where EN is your session name.

Setup: `pip install -r requirements.txt scikit-learn`, `export TOYPREF_RUNS=$PWD/runs`, `mkdir -p runs && cp -r ckpt/cond_base ckpt/base runs/`.

You own: `explore/EN/` and `docs/explore/EN.md`.

Common rules:

- Seeds 0–4 only. Never load seeds 5–9.
- Cells: `k4` (n_forget 4, random) and `alt` (n_forget 4, layout alt), npm 4 and 16, seeds 0–2. Add seeds 3–4 for your best method.
- Your edits may read only the network's inputs (x, t, c) and its activations. They may not use `CondMixture` or the ring geometry. You may use the repo's `compute` (the oracle) to evaluate.
- Compare every result with B1a (`R_hopB_bsup_neg_AMdist`) and B0 on the same cells, run with `scripts/cond_cell.py`.
- `docs/explore/EN.md` holds: a table of D, Removed, and Collateral for each method you tried against B0 and B1a; the best method and why it works; which measurements made the edit easy to design (these become Phase 4 tools); and what failed.

Your family:

- **E1, gated grafts.** Find where along the 20-step decode the forgotten modes are decided (renoise branches from step k). Add hidden units beside a block, gated by a linear detector on (x, t, c features, activations), that redirect bad trajectories early. Fit the units' weights by least squares or a small optimization on the added units only.
- **E2, direction edits.** Look for directions in the block activations that separate trajectories heading to forgotten modes from the rest, at matched conditions. Try `rank1` and `ablate_direction` style edits, alone and gated by t.
- **E3, targeted training and headroom.** Try parameter subsets for the frozen recipe's loss (only `out`, only one block, low-rank adapters), other anchors, and longer training. Also train the privileged oracle (BC on 20,000 π_ref samples from retained modes only, labelled by `CondMixture.assign`) to measure the floor of D.

Time box 6 hours. Write `docs/DONE_EN.md` and push.

## W2a: merge and pilot

Waits for: `DONE_W0.md`, `DONE_W1-A1.md`, `DONE_W1-A2.md`, `DONE_W1-B.md`, `DONE_W1-C.md`, `DONE_W1-D.md`

Branch: `git checkout -B agentic-repair-toy <branch printed for DONE_W0.md>`, then merge the five branches printed for the `DONE_W1-*` files, one at a time. Push the result as `agentic-repair-toy` (rule 5).

1. Resolve conflicts and every `INTERFACE_REQUESTS_*.md`. Put `EDIT_REFERENCE.md` into the surgeon's static prompt. Run all tests. Fix what fails. Push.
2. Phase 2 exit: write a graft by hand for unconditional `base_k2` and send it through the full path (orchestrator, CLI, gate, ledger). Do it yourself; do not use a subagent.
3. The two pre-pilot checks of PLAN.md Phase 3: the integration smoke on one cell, then the gate calibration.
4. Phase 3 pilot: 12 cells, arms B4, R0, D1, A1, and A2. Run the code arms through `scripts/jobs.py`. Run A1 and A2 sessions as `surgeon` subagents, up to 8 at a time, through `orchestrate next` and `record`. Write `results/pilot.md` with Table 2 and the cost of one session and one run.
5. If the decision rule fails, do one improvement round (Amendment 1): change prompts or tools only, rerun the pilot once, and record both rounds.
6. Write `docs/DONE_W2a.md` and push `agentic-repair-toy`.

## W2b: recursion and freeze

Waits for: `DONE_W2a.md`

Branch: `git checkout -B agentic-repair-toy <branch printed for DONE_W2a.md>`. Push the result as `agentic-repair-toy` (rule 5).

1. Read `docs/explore/E1.md`, `E2.md`, and `E3.md` if they exist on any branch (find them as the wait loop does).
2. Phase 4: build `contrast` and `unit_card` from the pilot's sandbox code (read it in the ledgers) and from the exploration findings, for A1, A2, and A3 alike. Do not put exploration findings into the surgeon's static prompt. Make R0, D1, and B4 iterative. Run full recursive A3 and the iterative code arms on `k4`, npm 4 and 16, seeds 0–4.
3. Phase 5: write `PREREG.md` (Section 10.5) with the grid of Amendment 1. Split it into 3 shares of roughly equal projected cost (Section 9.1). List every cell for every arm in each share, and the replicate order for each share.
4. Commit `PREREG.md` to `agentic-repair-toy`, write `docs/DONE_W2b.md`, and push.

## W3-1, W3-2, W3-3: confirmatory grid

Waits for: `DONE_W2b.md`

Use the same section for all three; N is 1, 2, or 3 from your prompt. The surgeon subagents run on Opus through their definitions, whatever your own model is.

Branch: `git checkout -b art/grid-N <branch printed for DONE_W2b.md>`

1. Run share N from `PREREG.md` (Phase 6, seeds 5–9): every listed arm on every listed cell, then the final evaluation. Run code arms through `scripts/jobs.py`. Run up to 8 surgeon sessions at a time. Write results under `results/grid/<setting>/s<seed>/n<npm>/<arm>/`. Do not change any method, threshold, or prompt. If a run fails, mark it invalid with the reason and rerun it once.
2. When the share is done, run A3 replicates in the order `PREREG.md` gives, under `results/grid/.../A3_rep1/`, until Wednesday 23:00 ET.
3. Write `docs/DONE_W3-N.md` with D for each arm and cell and your share's total cost, and push.

## W4: report

Waits for: `DONE_W2b.md`, `DONE_W3-1.md`, `DONE_W3-2.md`, `DONE_W3-3.md`

Branch: `git checkout -B agentic-repair-toy <branch printed for DONE_W2b.md>`, then merge the three branches printed for the `DONE_W3-*` files, one at a time.

Do Phase 7: run any exploratory arm that W3 did not finish, compute the statistics of Section 9.5 (replicates as a robustness check), and write `REPORT.md` with the contents of Section 9.6, including `results/method_map_rows.json`. Add a section with the exploration results (E1–E3) and the oracle floor of D. Label T1–T4. Use a Sonnet subagent for the plots, then check each plot yourself.

Write `docs/DONE_W4.md` with the four claim labels and their numbers, and the three most important findings in one line each. Push.
