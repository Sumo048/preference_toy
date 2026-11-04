# Cloud session prompts: agentic repair on the toypref toy

Companion to `TOY_PLAN.md`. Repo: `Sumo048/preference_toy`, default branch `cond-reroute`.

Each session gets a one-line prompt, for example:

> You are session W1-A. Read docs/agentic_repair/SESSION_PROMPTS.md and do section W1-A exactly.

The full instructions for each session are in the sections below.

## Setup (one time, about 15 minutes)

### 1. Commit the base checkpoints and per-cell results (on the cluster)

The cloud VM cannot reach the cluster. If π_ref is retrained there on other hardware, it is a different network, and B0 and B1a will not reproduce the method map exactly. The checkpoints are small (under 1 MB each).

```bash
cd /home/soumojib/repos/toypref-cond          # your clone of Sumo048/preference_toy
git checkout cond-reroute && git pull
R=/grogu/datasets/rollout/soumojib/toypref/runs
mkdir -p ckpt
rsync -a --include='*/' --include='policy.pt' --include='*report.json' --exclude='*' $R/cond_base/ ckpt/cond_base/
rsync -a --include='*/' --include='policy.pt' --include='*report.json' --exclude='*' $R/base/ ckpt/base/
tar czf ckpt/cond_cells.tgz -C $R/cond .
du -sh ckpt ckpt/*
```

If `ckpt/cond_cells.tgz` is over 90 MB, delete it (GitHub rejects files over 100 MB). The checkpoints alone are enough.

```bash
git add ckpt && git commit -m "Base checkpoints and per-cell results for cloud runs" && git push origin cond-reroute
```

### 2. Commit the plan, the agent definitions, and the command filter

Unzip `agentic_repair_files.zip` at the repo root. It adds:

- `docs/agentic_repair/TOY_PLAN.md`
- `docs/agentic_repair/SESSION_PROMPTS.md`
- `.claude/agents/surgeon.md`, `surgeon-sonnet.md`, `triage.md`
- `tools/surgeon_hook.py`

```bash
git pull
unzip -o agentic_repair_files.zip
chmod +x tools/surgeon_hook.py
git add docs/agentic_repair .claude/agents tools
git commit -m "Agentic repair: toy plan, session prompts, agent definitions"
git push origin cond-reroute
```

These files must be on `cond-reroute` before any session starts. Claude Code loads agent definitions when a session starts.

### 3. Connect GitHub for cloud sessions

The repo is public, so cloud sessions can clone it. To push branches, they need write access. Use one of these:

- At claude.ai/code, install the Claude GitHub App on `Sumo048/preference_toy`.
- Or, in a local terminal with `gh` signed in, run `claude` and then `/web-setup`.

### 4. Start sessions so the $250 credit pays

Use plain cloud sessions only:

- **claude.ai/code** in the browser: select `Sumo048/preference_toy` and branch `cond-reroute`, keep the **Default** environment (Trusted network, so PyPI works), choose the model, and paste the one-line prompt.
- **Terminal**: from the repo on branch `cond-reroute`, run `claude --cloud "You are session W0. Read docs/agentic_repair/SESSION_PROMPTS.md and do section W0 exactly."`. Each `--cloud` call starts a separate parallel session.

Do not start them inside a claude.ai Project, a Claude Code project, a Routine, or Remote Control: the credit excludes those. Local `claude` runs are not covered either.

The credit expires on November 4 at 11:59 PM PT (Thursday 2:59 AM in Pittsburgh). After that, sessions count against your Max plan limits with no extra charge, so unfinished waves still run.

Check the credit balance in the usage menu 10 minutes after Wave 1 starts. Rough cost guesses from the price table: Wave 0 $5–15; each Wave 1 session $10–40; one surgeon session $1–3; Wave 2a $30–70. These are guesses, not measurements. `results/spend.jsonl` records measured costs from Wave 0 on.

## Schedule

Each wave has a time box. When it ends, the session commits, pushes, writes its summary, and stops.

| Wave | Sessions | Model | Time box | Start when | Plan phases |
|---|---|---|---|---|---|
| W0 | 1 | Opus 5.5 | 1.5 h | setup is done | 0 |
| W1-A, W1-B, W1-C, W1-D | 4 in parallel | Opus 5.5 | 5 h | W0 has pushed `agentic-repair-toy` | 1, 2 |
| W2a | 1 | Opus 5.5 | 4 h | all four `art/*` branches are pushed | 2 exit, 3 |
| W2b | 1 | Opus 5.5 | 5 h | W2a reports that the pilot passed | 4, 5 |
| W3-1, W3-2, W3-3 | 3 in parallel | Sonnet 5.5 (surgeons stay Opus) | 6 h | W2b has pushed `PREREG.md` | 6 |
| W4 | 1 | Opus 5.5 | 2 h | all three `art/grid-*` branches are pushed | 7 |

A realistic timeline: W0 tonight, W1 overnight, W2a and W2b on Wednesday, W3 on Wednesday evening, W4 late Wednesday.

## Rules for every session

1. Read `docs/agentic_repair/TOY_PLAN.md` in full before any work.
2. Check out your branch first, as your section says. Then run `bash scripts/cloud_setup.sh` (from W1 on; W0 writes it).
3. Keep `docs/agentic_repair/STATE_<your session>.md` current: what is done, what runs, what is next. After a context compaction, read it again before you act.
4. Work only on your branch. Edit only the files your section gives you. Read other files freely.
5. If you need a change to another session's interface, write it in `docs/agentic_repair/INTERFACE_REQUESTS_<your session>.md` and use a local adapter until the merge.
6. CPU jobs go through `scripts/jobs.py` (W0 writes it): a resumable manifest runner with 4 workers, one torch thread each. No single job unit runs longer than 20 minutes. Results are written per unit, so finished units are skipped on restart.
7. While jobs or subagents run, do not end your turn. Poll with `sleep 90` followed by `python scripts/jobs.py ensure <manifest>`, which restarts dead workers. An idle session pauses its VM and stops its jobs. Background commands stop after about 30 minutes, which is why job units are short.
8. At each commit, run `python -m toyrepair.transcripts spend --session <your session>`. It appends your main-agent cost to `results/spend.jsonl`.
9. Commit at each milestone. Push your branch at least once an hour.
10. Use Sonnet subagents for code search, log summaries, and plots.
11. Do not open pull requests. Do not push to `cond-reroute`. Do not edit `.claude/agents/` or `tools/surgeon_hook.py`.
12. Stop at your time box. End with a summary of 15 lines or fewer: what works, key numbers, what failed, open interface requests, measured cost, and blockers for the next wave.

## W0

Scope: Phase 0 only. Time box: 1.5 hours.

Branch: `git checkout -b agentic-repair-toy origin/cond-reroute`

1. **Environment.** Record CPU count, RAM, Python and torch versions. Install the CPU torch wheel and the repo's dependencies (torch, numpy, pandas, matplotlib, scikit-learn) if missing. Write `scripts/cloud_setup.sh`. It must do all of these and finish in under 5 minutes on a fresh VM:
   - install the dependencies and run `pip install -e .` for `toyrepair`
   - unpack `ckpt/` into `runs/` (`runs/cond_base/s*/policy.pt`, `runs/base/s*/policy.pt`, and `runs/cond/` from `ckpt/cond_cells.tgz` if present)
   - create the unprivileged user `sandbox`, and set the repo directory to mode 700, owned by the user that runs Claude Code, so `sandbox` cannot read it (TOY_PLAN.md Section 6.6). Check that Claude Code itself can still read and write the repo afterwards.
2. **Run paths.** Make `toyrepair/__init__.py` set `TOYPREF_RUNS` to `<repo>/runs` when it is unset.
3. **Survey.** Write `docs/agentic_repair/SURVEY.md`, one page. Check every entry point and the network description in TOY_PLAN.md Section 5.1. Record the CPU time of one π_ref training (if `ckpt/` has no checkpoints, train seeds 0–9 now with `scripts/cond_base.py`, 4 in parallel), one frozen-recipe arm run, and one `compute` call. Record which `compute` keys are Removed and Collateral.
4. **Reproduce** B0 and `R_hopB_bsup_neg_AMdist` on `scripts/cond_cell.py --seed 0 --n_forget 4 --n_neg 4`. If `runs/cond/cells/s0_k4_n4/` holds the cluster results, compare each JSON field. Otherwise compare with the k4 seeds 0–4 means: 0.515 (B0) and 0.318 (frozen recipe). A single cell will differ from a mean; report the value.
5. **Scaffold.** Write the package of TOY_PLAN.md Section 6.11: every module with typed signatures, dataclasses, and docstrings that state inputs, outputs, and units, with `NotImplementedError` bodies. Four parallel sessions build against these signatures, so make them exact. Implement `toyrepair/cell.py` and the seed logic of `pools.py` fully, with tests. Add the console script `toyrepair-tool`, with `--help` only.
6. **Infra.** Write `scripts/jobs.py` (rule 6) with `run`, `ensure`, and `status`. Write a minimal `toyrepair/transcripts.py` with the `spend` command (rule 8).
7. **Isolation check.** Invoke the `surgeon` subagent and ask it to run `cat toyrepair/cell.py`, then `toyrepair-tool --help`. The first must be blocked and the second must run. Invoke `surgeon-sonnet` and `triage` once each the same way. Find the subagent transcripts on disk and confirm they hold token-usage fields. Record the path pattern and field names in SURVEY.md. If the hook does not block, stop and report: this is an escalation trigger.
8. Commit and push `agentic-repair-toy`.

## W1-A: harness core

Branch: `git fetch origin && git checkout -b art/harness origin/agentic-repair-toy`

You own: `toyrepair/policy.py`, `flowstate.py`, `miner.py`, `pools.py` (beyond W0's seeds), `oracle/`, `gate.py`, `quick_check.py`, `ledger.py`, `budget.py`, `eval_final.py`, and their tests. W0's signatures are the contract.

1. Policy server and flow states (Sections 6.1, 5.4). Test: restore at every step and continue gives bit-identical actions on 10,000 noises.
2. Oracle and final evaluation (Section 4, rule 8): wrap `CondMixture.assign` and `experiment.Bank`/`compute`. Only `gate.py` and `eval_final.py` import `oracle/`.
3. Miner, failure cards, and atlas, with token caps (Section 6.2). Log oracle purity to a CSV only.
4. Branch tool: renoise, Wilson top-up, k_commit, primitives, failure class, and the 40-row table (Section 6.3). Inspect tool (Section 6.4).
5. quick_check (Section 6.8), the per-run ledger (Section 6.9), and the budget counter (decodes, gate checks, cost).
6. Gate with both scopes, the six tests, and the paired bootstrap (Section 10). For test 6, call the unit-test hook that W1-B writes; until the merge, stub it with shape and NaN checks.
7. Gate reproducibility and power check (Section 10.3). Report the N_G you need.

Smoke-test each component on unconditional `base_k2` first. Report: miner purity and cluster counts on `k4` and `alt`, seeds 0–4, npm 4 and 16; k_commit and failure class of the largest cluster in each of those cells; the power result.

## W1-B: edit API and training

Branch: `git fetch origin && git checkout -b art/edits origin/agentic-repair-toy`

You own: `toyrepair/edits.py`, `patches.py`, `train.py`, `sandbox.py`, the gate's unit-test hook (a function in `edits.py`), `docs/agentic_repair/EDIT_REFERENCE.md`, and their tests. W0's signatures are the contract.

1. `graft_gated_path` with a hard gate (Section 6.7). Tests: graft identity on 10,000 reference flow states with the detector off; a zero-output graft gives bit-identical decodes; the API rejects a detector that reads anything other than network inputs and activations. Layers to support: `inp`, `blocks.{0,1,2}.fc1`, `blocks.{0,1,2}.fc2`, `out`, and the `cond` vector as a detector input.
2. `rank1`, `ablate_direction`, and `set_units`, with tests.
3. Patch files and rebuild from π_ref plus a patch stack. Test: the rebuild equals the live version bit for bit.
4. `targeted_train` over the repo's losses (rerouting with each target rule and support, NPO, KTO) and its anchors. Test: `targeted_train` configured as `R_hopB_bsup_neg_AMdist` on seed 0, n_forget 4, npm 4 matches `scripts/cond_cell.py` for that arm.
5. Sandbox (Section 6.6): the export (weights, a self-contained copy of `CondVelocityNet` with no toypref import, negatives, basins, a decode function), the `sandbox` user, the audit hook, the runner, and the 2,000-token output cap. Tests: as user `sandbox`, reading `src/toypref/cond/task.py` fails; importing `toypref` and `toyrepair.oracle` fails; subprocess and socket fail; a least-squares fit of a graft's `w_out` runs.
6. `EDIT_REFERENCE.md`: the edit calls and the sandbox, with inputs, units, and one example each, in 1,500 tokens or fewer. It goes into the surgeon's static prompt.

## W1-C: reproduction and code arms

Branch: `git fetch origin && git checkout -b art/arms origin/agentic-repair-toy`

You own: `toyrepair/arms/`, `scripts/run_arm.py`, `scripts/phase1_repro.py`, `results/phase1/`, `docs/agentic_repair/DEV_LOG.md`, and their tests.

1. **Phase 1 reproduction** with the repo's code directly (`scripts/cond_cell.py`), through `scripts/jobs.py`. Run B0, B1a (`R_hopB_bsup_neg_AMdist`), and B1b on `k4` (n_forget 4, random) and `alt` (n_forget 4, layout alt), seeds 0–4, npm 1, 4, 16, 64. Method-map values (seeds 0–4, mean over npm): k4 B0 0.515, B1a 0.318, B1b `R_hopB_bsup_mine_neg` 0.304; alt B0 0.515, B1a 0.457, B1b `R_hop_bsup_neg` 0.261. If `runs/cond/` holds the cluster's cell JSONs, also compare each cell. Exit: each mean is within 0.025. If one is not, stop and report: this is an escalation trigger.
2. **Arms** B0, B1a, B1b, B4, B5, R0, and D1 (Section 9.2), behind one entry point: `scripts/run_arm.py --arm <name> --setting <s> --seed <n> --npm <m>`. B5 configs per setting (top 3 on seeds 0–4):
   - k1: `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AMdist`, `PSEUDO_vote_dist`
   - k2: `R_hopB_bsup_mine_neg_AMmap`, `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AL`
   - k4, pairs, b3p1: `R_hopB_bsup_mine_neg`, `R_hopB_bsup_neg_AMdist`, `R_hopB_bsup_mine_neg_AMmap`
   - contig: `R_hopB_bsup_neg_AMdist`, `NPO_b10_neg_AMdist`, `PSEUDO_vote_dist`
   - alt: `R_hop_bsup_neg`, `KTO_b10_neg_AMmap`, `NPO_b10_neg_AL`
3. B4, B5, R0, and D1 need the branch tool and gate (W1-A) and `targeted_train` (W1-B). Write them against W0's signatures and test them with small fakes. W2a runs them for real.
4. Fix the D1 variant list (3 variants). Write it in `DEV_LOG.md` with the reason.

Report the reproduction table and the CPU time per arm run.

## W1-D: LLM roles, CLI, and orchestrator

Branch: `git fetch origin && git checkout -b art/llm origin/agentic-repair-toy`

You own: `toyrepair/cli.py`, `dossier.py`, `orchestrate.py`, `transcripts.py` (beyond W0's `spend`), `toyrepair/prompts/` (the static prompt texts), and their tests.

1. `toyrepair-tool`: session tokens; per-session limits (20 calls, 3 submissions); per-arm command sets (A1, A2, A3, A3-nobranch, A3-nogate); path checks that keep every path argument inside the run's scratch directory; a ledger record for every call; row caps that state the number of cut rows. Commands include `prompt` (static part, then the dossier) and `triage-input`.
2. Dossier builder with both caps (Section 7.3). Count tokens with a chars-per-token ratio measured once on a real transcript.
3. `orchestrate next`, `record`, and `status` (Sections 6.10 and 7.4): stop rules, cool-down, and the triage fallback. `status` prints one line per run.
4. Transcript reader: usage and cost per subagent with the Section 2 price table, plus the isolation audit (Section 7.5).
5. Static prompt texts in `toyrepair/prompts/`: the surgeon's (4,000 tokens or fewer: the policy interface from SURVEY.md, the CLI reference, a placeholder for `EDIT_REFERENCE.md`, and the session rules), and the triage's (JSON output only). No ring geometry and no mode count.
6. A mock harness that fakes tool outputs. Run one real `surgeon` session and one `triage` call against the mock to test the mechanics end to end. Report their tokens and cost.

## W2a: merge and pilot

Branch: `git fetch origin && git checkout agentic-repair-toy && git merge origin/art/harness origin/art/edits origin/art/arms origin/art/llm`

1. Resolve conflicts and every `INTERFACE_REQUESTS_*.md`. Put `EDIT_REFERENCE.md` into the surgeon's static prompt. Run all tests. Push.
2. Phase 2 exit: write a graft by hand for unconditional `base_k2` and send it through the full path (orchestrator, CLI, gate, ledger). Do it yourself; do not use a subagent.
3. The two pre-pilot checks of TOY_PLAN.md Phase 3: the integration smoke on one cell, then the gate calibration.
4. Phase 3 pilot: 12 cells, arms B4, R0, D1, A1, and A2. Run the code arms through `scripts/jobs.py`. Run A1 and A2 sessions as `surgeon` subagents, at most 4 at a time, through `orchestrate next` and `record`. Write `results/pilot.md` with Table 2 and the cost of one session and one run.
5. Check the escalation triggers (Section 11): the decision rule, mean session cost above $5, and any audit hit.

Report: the pilot table, the decision, and the measured costs. If a trigger fires, say so in the first line of the summary.

## W2b: recursion and freeze

Branch: `git fetch origin && git checkout agentic-repair-toy && git pull`

1. Phase 4: build `contrast` and `unit_card` from the pilot's sandbox code (read it in the ledgers), for A1, A2, and A3 alike. Make R0, D1, and B4 iterative. Run full recursive A3 and the iterative code arms on `k4`, npm 4 and 16, seeds 0–4.
2. Phase 5: write `PREREG.md` (Section 10.5). Compute the grid from Section 9.1 with the $120 cap, including main-agent cost from `results/spend.jsonl`. Split it into 3 shares (Section 9.1) and list every cell for every arm in each share.

Report: the Phase 4 D values against B1a, B5, and D1 on seeds 0–4, the cost per A3 run, and the shares.

## W3-1, W3-2, W3-3: confirmatory grid

Use the same section for all three; N is 1, 2, or 3 from your prompt. Run the main agent on Sonnet 5.5. The surgeon subagents stay on Opus through their definitions.

Branch: `git fetch origin && git checkout -b art/grid-N origin/agentic-repair-toy`

Run share N from `PREREG.md` (Phase 6, seeds 5–9): every listed arm on every listed cell, then the final evaluation. Run code arms through `scripts/jobs.py`. Run surgeon sessions as subagents, at most 4 at a time. Write results under `results/grid/<setting>/s<seed>/n<npm>/<arm>/`. Do not change any method, threshold, or prompt. If a run fails, mark it invalid with the reason and rerun it once.

Report per cell: D for each arm, and the total cost of your share.

## W4: report

Branch: `git fetch origin && git checkout agentic-repair-toy && git pull && git merge origin/art/grid-1 origin/art/grid-2 origin/art/grid-3`

Do Phase 7: run any exploratory arm that W3 did not finish, compute the statistics of Section 9.5, and write `REPORT.md` with the contents of Section 9.6, including `results/method_map_rows.json`. Label T1–T4. Use a Sonnet subagent for the plots, then check each plot yourself.

Report: the four claim labels with their numbers, and the three most important findings in one line each.
