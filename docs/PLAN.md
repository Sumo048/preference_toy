# Plan: Agentic Policy Repair on the toypref Flow Toy

Date: 2026-11-03. Parent plan: "Plan: Agentic Policy Repair on Isaac Lab Factory GearMesh" (the GearMesh plan).

> Repo note (2026-11-03): this compact repo is the testbed the plan calls "toypref". Where the plan
> says "default branch `cond-reroute`", read `main` of this repo (code = `cond-reroute` @ c48b697,
> unchanged). See `README.md` for what `ckpt/` holds and what is missing. The plan text below is
> otherwise verbatim.

## 1. Purpose

This plan applies the GearMesh plan to the toypref flow toy before any Isaac work.

The toy runs on CPU. The whole loop runs inside Claude Code cloud sessions: base policies, edits, gate, evaluation, and the LLM roles.

The toy has two jobs. First, it builds and debugs the harness, the edit API, and the surgeon loop cheaply. Second, it answers a research question on its own.

The research question: can an LLM surgeon that edits weights remove forgotten modes from a conditional flow policy better than the best toypref recipes, with the same negatives? Does the counterfactual branch tool help?

The work has two stages. The pilot (Phase 3) asks one question: can the surgeon turn a rescue that the branch tool finds into a reliable edit, with no damage to retained behaviour? The confirmatory grid (Phase 6) runs only if the pilot passes its decision rule.

Read the sections in order. Section 2 defines each term once.

The user approves Section 12 by starting the first cloud session. Do not wait for a second approval unless an escalation trigger fires (Section 11).

## 2. Terms

Each term has one meaning in this plan.

| Term | Definition |
|---|---|
| toypref | The existing repo with the 2-D flow-policy preference-editing toy and the Toypref Method Map results. |
| Mode | One of the 8 behaviours of the toy. The modes sit on a ring. |
| Forgotten mode | A mode that the edit must remove. |
| Retained mode | A mode that the edit must keep. |
| Condition c | The scalar policy input in [0, 1). The ring rotates by 1.5 mode spacings and grows as c grows. |
| Conditional toy | The toy with a condition. It has no positive labels. It is the main testbed. |
| Unconditional toy | The toy with no condition. It is the smoke test. |
| Setting | One data situation of the method map, for example conditional `k4` (forget 4 of 8, random set). |
| npm | Negatives per forgotten mode: 1, 4, 16 or 64 in the conditional toy. |
| Cell | One (setting, seed, npm) combination. |
| π_ref | The base flow policy of a cell, before any edit. |
| Logged negative | A tuple (c_i, ξ_i, a_i) from toypref: π_ref produced action a_i from noise ξ_i at condition c_i, and a label marks a_i as bad. |
| Decode | One integration of the velocity field from noise to action: 20 Euler steps. |
| Step k | Integration step k ∈ {0, …, 20}. Step 0 is the noise. Step 20 is the action. t_k = 1 − k/20. |
| Flow state | (x_k, t_k, c) at one step of one decode. |
| Network inputs | The inputs of the velocity network: x_k, t_k, c (as toypref encodes them). Deployed edits read network inputs and activations only. |
| Oracle | Code that maps (action, c) to a mode index or to "between modes". Only the gate and final evaluation use it. |
| Removed | The share of π_ref's mass on forgotten modes that an edit removes (method-map definition). |
| Collateral | The total-variation distance between the edited policy and π_ref over retained modes plus the region between modes, each renormalised (method-map definition). |
| D | Distance to ideal: the method-map score. The ideal puts equal mass on each retained mode and no mass elsewhere. Lower is better. |
| Bad basin B_i | The toypref basin of logged negative i: the linked group of 128 π_ref samples at c_i that contains a_i. It uses no labels. |
| Bad vote | The toypref vote v(ξ) over all logged negatives' basins. It uses no labels. |
| Residual basin mass r_i(π) | The share of 256 fresh noises at c_i that π decodes into B_i. It uses no labels. |
| Version | One policy state: π_ref plus a stack of accepted edits. Each version has an id. |
| Branch | A decode that restarts from a saved flow state. |
| Renoise | The branch noise. At step k: x̂ = x_k − t_k·v(x_k, t_k, c), then x′ = t_k·ε + (1 − t_k)·x̂ with fresh ε ~ N(0, I). At step 0, x′ is fresh noise. |
| Rescue | A branch that ends outside B_i at a typical action. Typical: the 5th-nearest-neighbour distance to π_ref samples at c_i is at or below their 90th percentile. |
| p(k) | Rescue curve: the share of M renoise branches from step k that end in a rescue, averaged over a cluster's members. |
| k_commit | The earliest sampled step after which every later p(k) is below 0.2. If no step qualifies, k_commit is the last sampled step. |
| Primitive | A scripted change to a decode: `push(u, α, k, h)` adds a total displacement α along unit vector u over steps k to k+h−1; `hop(τ, u)` sets ξ ← ξ + τu at step 0. |
| Failure cluster | A group of logged negatives that the miner joins (Section 6.2). |
| Atlas | The set of failure cards for one version. |
| Failure class | R: a primitive at or before k_commit rescues at least 50% of a cluster's members. U: no tested primitive does. The toy has no S class, because fresh noise at step 0 always escapes one basin. |
| Edit | One change through the edit API. Each edit makes a new version. |
| Detector | A linear function of network inputs and activations, with a threshold. |
| Gated path | Hidden units added beside a layer that are active only when their detector fires. The original units do not change. |
| Targeted training | Gradient updates on a selected parameter subset, with a selected data set, with the locality penalty. |
| Locality penalty | A toypref anchor loss (vote-masked density anchor by default) on a fixed reference set. |
| Gate | Code that accepts or rejects an edit with fixed tests (Section 10). |
| Gate check | One run of the gate on one fresh gate batch. |
| Session | One surgeon context, from its dossier to an accept or a limit. |
| Surgeon | The LLM role that designs and submits edits for one failure cluster. |
| Triage | The LLM role that selects the next failure cluster. |
| Dossier | The compact input that the harness gives to one session. |
| Tool CLI | `toyrepair-tool`: the only interface the surgeon has. |
| Ledger | The append-only JSONL record of all sessions, edits, and gate checks. |
| Run | One arm on one cell, from π_ref to a stop rule. |
| Arm | One method in the experiment grid. |
| Frozen recipe | The round-3 toypref recipe: hop + veto + basin support + vote-masked density anchor (`R_hopB_bsup_neg_AMdist`). |
| Cost | The estimated API-price cost of LLM tokens: Opus 5.5 at $4 input, $5 cache write, $0.20 cache read, $20 output per million tokens; Sonnet 5.5 at $2, $2.50, $0.10, $10. |

## 3. Goal and claims

The goal: find out whether an LLM surgeon with counterfactual diagnosis and weight edits removes forgotten modes better than the best toypref recipes, and which edit type does the work.

A3 is the practical system. The central comparisons are A3 against B1a and B5, A3 against D1, A2 against A1, and A2 against R0.

Pre-register four claims. Label each claim supported, unsupported, or inconclusive (Section 9.5).

T1 (system). A3 reaches lower D than B1a and lower D than B5 on the confirmatory cells.

T2 (LLM value). A3 reaches lower D than D1.

T3 (edit type). A2 and A1 differ in D. The test is two-sided.

T4 (embedding). A2 and R0 differ in D. The test is two-sided. It asks whether a fix inside the weights improves on the same fix that code executes at sampling time.

Exploratory comparisons get estimates and intervals but no labels: A3 against A3-nobranch, A3 against A3-sonnet, and A3 against A3-nogate.

A negative or inconclusive result is a result. Report it with the same detail as a positive result.

## 4. Rules for the implementation agent

1. Survey the repo before you write code. Record what exists in `SURVEY.md`.
2. Put all new code in the package `toyrepair/`. Do not change toypref code. Import it.
3. Use Sonnet subagents for simple tasks: code search, log summaries, plots, and config checks.
4. Do not add standalone diagnostic phases. Each diagnostic is a metric that a run logs.
5. Write `PREREG.md` in Phase 5. After that, record each change as a dated amendment with its reason.
6. Never delete a result. If a result is invalid, mark it invalid and record the reason.
7. Keep each edit reproducible. Store each edit as a patch file and its arrays. Rebuild any version from π_ref and its patch stack.
8. Use the toypref evaluation code for final D, Removed, and Collateral, so the numbers match the method map.
9. The surgeon and the triage never see the oracle, oracle-based numbers other than gate results, the forgotten-mode set, the ring geometry, or other runs. Section 7.5 states how the harness enforces this.
10. Deployed edits read network inputs and activations only.
11. Commit at the end of each phase. Push to the branch named in the session prompt.
12. Ask the user only at an escalation trigger (Section 11).
13. This plan marks some facts with **(verify)**. Verify each one before you use it.

## 5. Testbed

### 5.1 Policy and data

The repo is `Sumo048/preference_toy`, default branch `cond-reroute`. Use its flow policy, data generation, settings, and seeds without change.

Entry points (verify each):

- `src/toypref/cond/task.py`: `CondMixture` (ring geometry and `assign`, the oracle), `CondVelocityNet`, and `CondFlow` (20-step Euler sampler).
- `src/toypref/cond/experiment.py`: `train_base`, `make_pool`, `forget_modes` and `LAYOUTS`, `editset`, the arm registry `ARMS`, `train_arm`, `Bank`, and `compute` (final metrics, including `tv_to_ideal`).
- `scripts/cond_base.py --seed s` trains π_ref (40,000 steps). `scripts/cond_cell.py --seed s --n_forget k --n_neg m --layout L --arms a,b` runs arms on one cell, skip-if-exists.
- Run outputs go to `$TOYPREF_RUNS`. Its default is a cluster path, so every cloud session sets `TOYPREF_RUNS` to a directory inside the clone.

The settings map onto the repo's arguments: `k1`, `k2`, `k4` are `--n_forget 1, 2, 4 --layout random`; `contig`, `alt`, `pairs`, `b3p1` are `--n_forget 4 --layout <name>`. npm is `--n_neg`.

The velocity network (verify): `inp` Linear(2→128); 3 residual blocks, each LayerNorm → `fc1` Linear(128 + 128 → 128) → SiLU → `fc2` Linear(128 → 128), added to the input; `out_norm` LayerNorm; `out` Linear(128 → 2). The 128-dim `cond` vector concatenates a time MLP over a sinusoidal embedding of t with a condition MLP over Fourier features of c. It enters every block's `fc1`.

`task.py` holds both the network and the ring geometry. So the surgeon's sandbox gets a copy of the network class, never `task.py`.

If the user commits the cluster's base checkpoints and per-cell results under `ckpt/` (see `SESSION_PROMPTS.md`), use them. Then π_ref is the same network the method map used, and B0 and B1a reproduce exactly. Otherwise, train π_ref for seeds 0–9 with `scripts/cond_base.py`.

Record the CPU time of one π_ref training, one frozen-recipe run (5,000 steps), and one final evaluation on the cloud VM.

### 5.2 Settings

The conditional toy is the main testbed. Its settings are `k1`, `k2`, `k4`, `contig`, `alt`, `pairs`, and `b3p1`, with npm ∈ {1, 4, 16, 64}.

The unconditional toy `base_k2` is the smoke test for each new component. Its best methods already sit near the oracle (D ≈ 0.04 against 0.03), so it has no headroom for claims.

Development uses seeds 0–4. Confirmation uses seeds 5–9. This matches the method map.

Reference values from the method map (conditional, seeds 5–9, mean over npm):

| Setting | B0 (π_ref) | Frozen recipe | Best config on the map |
|---|---|---|---|
| k1 | 0.149 | 0.107 | 0.107 (frozen recipe) |
| k4 | 0.515 | 0.265 | 0.265 (frozen recipe) |
| alt | 0.515 | 0.450 | 0.262 (hop, basin support, no anchor) |
| contig | 0.513 | 0.228 | 0.228 (frozen recipe) |
| pairs | 0.514 | 0.251 | 0.188 (hop + veto + mined negatives) |

The best recipes remove 50–75% of forgotten mass with Collateral 0.04–0.15. The headroom is large: the floor is near 0.03.

### 5.3 Pools

The cell index is 1000·g + 100·s + 10·seed + n. Here g is 0 for the conditional toy and 1 for the unconditional toy, s is the setting's position in the lists of Section 5.2 (k1 = 0, k2 = 1, k4 = 2, contig = 3, alt = 4, pairs = 5, b3p1 = 6; unconditional base_k1 = 0, base_k2 = 1, base_k4 = 2), and n is the npm position (1, 4, 16, 64 → 0, 1, 2, 3).

Each cell keeps three independent random streams for (c, ξ) draws, made with `numpy.random.SeedSequence([pool, cell index, check index])`: pool 1 is mining, pool 2 is gate, and pool 3 is final evaluation. The check index is 0 for mining and final evaluation.

The logged negatives and π_ref of a cell come from toypref with its own seed rule. Final evaluation uses the toypref evaluation code. If that code draws its own samples, verify that it uses fixed seeds, so all arms see the same final samples.

Final-evaluation numbers never go back into any loop.

### 5.4 Branch validity

Branch validity holds by construction in the toy: the decode is a deterministic CPU computation, and a flow state restores exactly.

Phase 2 confirms it with one test: a decode restored at each step k and continued with no perturbation gives the same action as the uninterrupted decode, bit for bit, on 10,000 noises.

## 6. Harness

The harness is Python code. The LLM roles make decisions. The code does all computation.

### 6.1 Policy server

The policy server loads a version, decodes batches, captures activations at named layers, and saves flow states. It counts decodes per run.

### 6.2 Failure miner

The miner reads the logged negatives of the cell and the current version.

It computes a label-free affinity between negatives i and j: A_ij = ½(1[f_ref(ξ_j, c_i) ∈ B_i] + 1[f_ref(ξ_i, c_j) ∈ B_j]).

It clusters on 1 − A with HDBSCAN. If HDBSCAN gives fewer than 2 clusters, use agglomerative clustering with average linkage, cut at distance 0.7.

For each cluster, it computes the residual basin mass under the current version: the mean of r_i over its members.

A cluster is open while its residual basin mass is at least 0.02.

Code writes one failure card per open cluster: id, member count, residual basin mass under π_ref and under the current version, the c range of members (5th to 95th percentile), cohesion (mean A within the cluster), k_commit, failure class, and 3 exemplar negative ids. The cap for one card is 150 tokens. The cap for the atlas is 1500 tokens.

The miner also logs cluster purity with the oracle. Purity goes to the ledger and the report only, never to a card.

### 6.3 Branch tool

The branch tool runs on all members of one cluster. Its inputs are the cluster id, the steps (default k ∈ {0, 2, 4, 6, 8, 10, 12, 14, 16}), M (default 32), and a primitive list.

Compute p(k) at each step with renoise.

If the 95% Wilson interval of p(k) contains 0.2, add 64 branches at that step.

Find k_commit. Report the full curve, because p(k) is not always monotonic.

Run each primitive at k_commit and the two sampled steps before it. The default primitive list: `push(u, α, k, h=3)` for 8 directions u and α ∈ {0.5, 1, 2} × the basin radius (the 90th-percentile distance of B_i's samples from their mean), and `hop(τ, u)` with the toypref hop search.

Set the failure class (Section 2).

The output is one table: step, t_k, p(k) with its interval, and the rescue rate of the best 5 primitives. The cap is 40 rows.

The surgeon can add a primitive as code through the sandbox. The ledger records it as a change to the tool set, not to the policy.

### 6.4 Inspect tool

The inspect tool returns one decode of one negative, downsampled to 21 rows or fewer: step, t_k, x_k, predicted action x̂, |v|, whether x̂ is in B_i, and selected unit activations.

### 6.5 Activation tools

Build these tools in Phase 4. Base them on the analyses that the surgeon writes in the sandbox in Phase 3.

`contrast(cluster, layer, k)`: compare activations at step k between decodes that end in the cluster's basins and decodes at nearby conditions that end outside. Return the top 10 units by effect size, a logistic-probe direction with its held-out AUC, and the difference of velocity means.

`unit_card(layer, idx)`: return the correlation of one unit with network inputs, step, and basin membership of x̂.

The cap for each output is 800 tokens.

### 6.6 Sandbox

The sandbox runs Python with numpy and torch in a subprocess, with `python -I`, in the run's scratch directory.

Its inputs are an export: the current version's weights, a self-contained model class with no toypref import, the logged negatives, the basins, and a decode function. Each sandbox decode counts.

The sandbox subprocess runs as a separate unprivileged user, `sandbox`. The repo directory, which holds `toypref` and `toyrepair/oracle/`, is mode 700 and owned by the user that runs Claude Code. The `sandbox` user can read only the export and write only the scratch directory. This is the isolation layer.

A Python audit hook adds a second layer. It blocks imports of `toypref` and `toyrepair.oracle`, network access, and subprocess calls.

The sandbox has no write access to weights. It writes arrays to the scratch directory. Edit calls read arrays from there. The cap for each output is 2000 tokens.

The sandbox may fit numbers for a weight write: least squares or a small optimization. The difference between A1 and A2 is the structure of the change, not the use of gradients.

### 6.7 Edit API

Each call makes a new version id and writes a patch file.

| Call | Effect |
|---|---|
| `graft_gated_path(layer, detector, n_units, w_in, w_out)` | Adds n_units gated units beside a layer. The gate is hard: the units' output is exactly 0 when the detector is off. The detector reads network inputs and activations only. |
| `rank1(layer, u, v, s)` | W ← W + s·u·vᵀ |
| `ablate_direction(layer, v, side)` | Normalizes v. side = pre: W ← (I − v·vᵀ)·W and b ← (I − v·vᵀ)·b. side = post: the next layer's input weights become W_next·(I − v·vᵀ). |
| `set_units(layer, idx, w_in, b, w_out)` | Writes the weights of selected units directly. |
| `targeted_train(params, data, objective, steps, anchor)` | Targeted training. Objectives: BC to rescue targets (toypref rerouting, with its logged-noise, jitter, or basin supports), NPO, KTO. Anchors: the toypref anchors. |

For each edit, record the fraction of weights changed, the units added, the Frobenius norm of the change, and the decodes used.

The edit API rejects a detector that reads anything other than network inputs and activations.

A unit test confirms the graft identity: with the detector off, the new and current versions give bit-identical velocities on 10,000 reference flow states. A graft with zero output weights gives bit-identical decodes everywhere.

### 6.8 quick_check

`quick_check` runs a candidate version with no oracle. It returns three numbers: the share of the cluster's logged noises that still decode into their basins; the cluster's residual basin mass; and the drift, the share of 4096 mining-pool pairs (c, ξ) whose decode moves by more than the median basin radius from the current version's decode.

### 6.9 Ledger

The ledger is append-only JSONL, one file per run: `runs/<run id>/ledger.jsonl`. It has one record per session, edit, gate check, and tool call. For each dossier, code writes a ledger summary of the recent attempts and all accepted edits. The cap is 2000 tokens.

### 6.10 Orchestrator

1. Run the miner on the current version. Build the atlas.
2. Triage selects one open cluster.
3. Code runs the branch tool on the cluster. Code builds the dossier. A session starts.
4. The surgeon submits edits to the gate. The gate accepts or rejects each one.
5. After an accept, the new version becomes current. Go to step 1.
6. If the session ends without an accept, put the cluster on cool-down for 2 iterations. Go to step 2.

The loop stops when one stop rule triggers:

- no cluster is open
- 3 sessions in a row end without an accept
- the run's gate checks reach 3 × (number of clusters in the first atlas + 2)
- the run's cost cap is spent.

### 6.11 Package layout and interfaces

Wave 0 writes these modules as stubs with signatures and docstrings. Later waves fill them in. File ownership per wave is in `SESSION_PROMPTS.md`.

| Module | Contents |
|---|---|
| `toyrepair/cell.py` | `load_cell(grid, setting, seed, npm) -> Cell`: π_ref, logged negatives, basins, vote; the oracle stays private. |
| `toyrepair/policy.py` | `Version`, `decode(version, xi, c, save_states=False, capture=())`, `velocity(...)`, `layers(version)`. |
| `toyrepair/flowstate.py` | `restore_and_decode`, `renoise`, primitives, `branch(...) -> BranchTable`. |
| `toyrepair/miner.py` | `mine(cell, version) -> Atlas`. |
| `toyrepair/pools.py` | `draw(cell, pool, n, index) -> (c, xi)`. |
| `toyrepair/oracle/` | Labels, Removed, Collateral, D, and the toypref evaluation wrapper. Only `gate.py` and `eval_final.py` import it. |
| `toyrepair/gate.py` | `gate_check(cell, base, current, new, cluster) -> GateResult`. |
| `toyrepair/edits.py`, `patches.py` | Edit API, patch files, rebuild. |
| `toyrepair/train.py` | `targeted_train` on toypref losses. |
| `toyrepair/sandbox.py` | Export, audit hook, runner. |
| `toyrepair/quick_check.py`, `ledger.py`, `budget.py` | As named. |
| `toyrepair/cli.py` | `toyrepair-tool`, with per-session limits. |
| `toyrepair/dossier.py`, `orchestrate.py` | Dossier builder and the code half of the loop. |
| `toyrepair/arms/` | B0, B1a, B1b, B4, B5, R0, D1. |
| `toyrepair/eval_final.py` | Final evaluation of a version on a cell. |
| `toyrepair/transcripts.py` | Token usage, cost, and the isolation audit from subagent transcripts. |
| `.claude/agents/*.md`, `tools/surgeon_hook.py` | LLM role definitions and the command filter. They are on the default branch already. No wave edits them. |

## 7. LLM roles and cost

The LLM roles run as Claude Code subagents inside the cloud session. The cloud-session credit pays for them. No API key is needed.

Each surgeon session is a new subagent invocation, so each starts with a fresh context.

The agent definitions `.claude/agents/surgeon.md`, `surgeon-sonnet.md`, and `triage.md`, and the hook `tools/surgeon_hook.py`, are on the repo's default branch before the first session. Claude Code loads agent definitions at session start, so no wave changes these files. The definitions are thin. The full static prompt comes from `toyrepair-tool prompt`, so code can change it.

### 7.1 Triage

Triage is the subagent `triage` with model `sonnet`. Its only command is `toyrepair-tool triage-input`, which prints the atlas and the ledger summary. Its output is JSON: one cluster id, the expected failure class, and a one-line hypothesis.

If the triage output does not parse, code selects the open cluster with the largest residual basin mass that is not on cool-down.

### 7.2 Surgeon

The surgeon is the subagent `surgeon` with model `opus`. Its only tool is Bash, restricted by a hook to `toyrepair-tool` commands (Section 7.5).

The tool CLI commands are: `prompt`, `dossier`, `atlas`, `branch`, `inspect`, `sandbox`, `quick_check`, the edit calls, `submit_edit`, and (from Phase 4) `contrast` and `unit_card`. Arms remove commands (Section 9.2). A removed command returns an error.

`submit_edit` sends a candidate version to the gate. The surgeon gets back pass or fail for each of the six tests, plus the number of test 1: the paired fall in target-mode mass with its 95% interval. It gets no other oracle number.

The session limits are 20 CLI calls, 3 gate submissions, and 150k tokens of context. The CLI enforces the first two. Code checks the third from the transcript and marks a session that exceeds it.

### 7.3 Dossier

The dossier has a static part and a variable part. `toyrepair-tool prompt` prints both, static part first, so prompt caching covers the static part. The static cap is 4000 tokens. It holds the policy interface (2-D action, scalar condition, 20-step decode, layer names and sizes) and the CLI and edit API reference. It gives no ring geometry and no number of modes.

The surgeon can discover the ring and the mode count by sampling π_ref through the sandbox. This is allowed. The baselines also query π_ref without labels.

The variable part comes from `toyrepair-tool dossier`, which `prompt` also calls. Its cap is 6000 tokens. It holds, in order: the failure card of the target cluster, its branch table, the ledger summary, and the remaining gate checks and cost.

### 7.4 Session mechanics

The main agent of the cloud session runs `python -m toyrepair.orchestrate next --run <id>`. The command prints the next action.

For a session, the command prints a one-time session token. The main agent invokes the surgeon subagent with one short message: the run id, the token, and the instruction to start with `toyrepair-tool prompt --token <token>`. For the A3-sonnet arm it invokes `surgeon-sonnet` instead.

The CLI maps the token to (run, session) and enforces the limits. The surgeon cannot reach another run with its token.

The surgeon ends with a one-line summary. The main agent then runs `python -m toyrepair.orchestrate record --token <token>`. Code reads the subagent transcript, sums its token usage and cost, audits it (Section 7.5), and writes the ledger record.

The main agent runs at most 4 surgeon subagents at the same time, on different runs. It keeps no session details in its own context beyond the one-line summaries.

### 7.5 Isolation

The surgeon and triage definitions allow only Bash. The PreToolUse hook `tools/surgeon_hook.py` blocks every command except a single `toyrepair-tool` call: no `;`, `&&`, `||`, pipes, redirection, backticks, `$(`, or quotes. A heredoc body for `sandbox --stdin` is allowed.

The CLI checks that each path argument is inside the run's scratch directory.

Wave 0 verifies that the hook runs inside the cloud session. If it does not, stop and report: this is an escalation trigger.

After each session, code greps the subagent transcript in `~/.claude/projects/*/*/subagents/` for paths under `toyrepair/oracle/`, toypref setting files, and other runs. A hit marks the run invalid.

### 7.6 Cost rules

The ledger records input, cache-write, cache-read, and output tokens and the cost of each session.

Each cloud session also appends its own main-agent cost to `results/spend.jsonl`, read from its transcript in `~/.claude/projects/`, at each commit. Grid budgets include main-agent cost.

Until Phase 5, the cost cap of one run is $8.

Phase 5 sets the grid size from pilot costs (Section 9.1).

## 8. Phases

The pilot (Phase 3) runs as soon as the minimum harness works. The confirmatory grid runs only if the pilot passes.

### Phase 0: Survey and scaffold (Wave 0, 1.5 hours maximum)

Read toypref: data generation, the velocity network, settings, seeds, the methods of the method map, and the evaluation code.

Write `SURVEY.md` (one page). Include the timing numbers of Section 5.1.

Reproduce B0 and the frozen recipe on conditional `k4`, seed 0, npm 4. Compare D with the method map, or with the cluster's per-cell result if `ckpt/` has it.

Write the package skeleton of Section 6.11 with signatures, docstrings, and `NotImplementedError` bodies. Implement `cell.py` and the seed logic of `pools.py` (Section 5.3) fully.

Verify that the hook blocks a forbidden command and that the subagents `surgeon`, `surgeon-sonnet`, and `triage` run.

Exit: the skeleton, `cell.py`, `pools.py` seeds, `SURVEY.md`, and `scripts/cloud_setup.sh` are on the branch `agentic-repair-toy`.

### Phase 1: Reproduction and pools (Wave 1)

Make the pools (Section 5.3).

Reproduce B0, B1a, and B1b on seeds 0–4 for `k4` and `alt`, all npm values.

Exit: each mean D (seeds 0–4, mean over npm) is within 0.025 of the method-map value for seeds 0–4.

### Phase 2: Minimum harness (Wave 1)

Build the policy server, flow states, branch tool, miner, inspect tool, sandbox, quick_check, gate, ledger, budget counter, tool CLI (including `prompt` and `triage-input`), dossier builder, orchestrator, and transcript reader.

Build `graft_gated_path` and `targeted_train` first. Build `rank1`, `ablate_direction`, and `set_units` in the same wave if time allows.

Build arms B0, B1a, B1b, B4, B5, R0, and D1.

Run the branch-validity test (Section 5.4), the graft identity test, the gate reproducibility check, and the gate power check (Section 10.3).

Exit: all tests pass, and the ledger records a hand-written graft on unconditional `base_k2` that went through the full path. The gate result of that graft can be accept or reject.

### Phase 3: Pilot (Wave 2a)

Cells: conditional `k4` and `alt`, npm 4 and 16, seeds 0, 1, 2. This gives 12 cells.

Before the pilot, run two checks:

1. Integration smoke: all 5 pilot arms on one cell (`k4`, seed 0, npm 4), with one LLM session each for A1 and A2. Record its cost.
2. Gate calibration: run B4's first repair on the 12 pilot cells through the gate. If fewer than half pass, loosen tests 3–5 one step (3: 0.10 → 0.15; 4 and 5: 0.01 → 0.02), record the change in `DEV_LOG.md`, and check again once.

Each run repairs one cluster: the open cluster with the largest residual basin mass in the first atlas. It must be class R. If it is class U, take the next class-R cluster. Record the purity of the chosen cluster.

Run arms B4, R0, D1, A1, and A2 on each cell. Each run makes one repair, with no recursion: one session for A1 and A2, up to 3 gate checks for every arm.

Evaluate each repair on the final-evaluation pool: the removed share of the target cluster's oracle modes, Collateral against π_ref, and D.

Measure the cost of one session and one run: tokens, cost, decodes, and wall time.

Decision rule: continue to Phase 4 if A1 or A2 removes at least 50% of the target modes' mass, with D below π_ref's D and Collateral at or below 0.10, in at least 8 of 12 cells. Otherwise stop and report. Report all five arms in both cases.

### Phase 4: Full tools and recursion (Wave 2b)

Build `contrast` and `unit_card` from the pilot's sandbox analyses. Give them to A1, A2, and A3 alike. Finish `rank1`, `ablate_direction`, and `set_units` if Phase 2 did not.

Make R0, D1, and B4 iterative: each repeats mining and repair in the same loop schedule as A3.

Run full recursive A3 on conditional `k4`, npm 4 and 16, seeds 0–4.

Exit: each run ends at a stop rule. The ledger, the atlas of each iteration, and the version stacks are on disk.

### Phase 5: Protocol freeze (Wave 2b)

Freeze the gate margins, thresholds, pool seeds, session limits, subagent models, and the grid (Section 9.1).

Write `PREREG.md` (Section 10.5).

Exit: `PREREG.md` is committed. Do not wait for the user.

### Phase 6: Confirmatory grid (Wave 3)

Run the grid on seeds 5–9 and the final evaluation.

Exit: each grid cell has a final evaluation for each of its arms.

The exploratory arms (Section 9.3) run in this phase, in share 3.

### Phase 7: Report (Wave 4)

Run any exploratory arm that Phase 6 did not finish.

Write `REPORT.md` (Section 9.6).

Exit: `REPORT.md` labels T1–T4 supported, unsupported, or inconclusive.

## 9. Experiment grid

### 9.1 Grid size

B0, B1a, and B1b run on all 140 conditional cells (seeds 5–9). The other code arms run on every cell that has an LLM arm. Code arms cost CPU time only.

LLM arms run on grid cells in the priority order below. Phase 5 computes the mean cost of an A3 run from Phase 4, including main-agent cost. It adds cells in order until the projected cost of Phases 6 and 7 reaches $120. The user can change this cap.

Priority order (seeds 5–9):

1. `k4`, npm 4 and 16
2. `alt`, npm 4 and 16
3. `k1`, npm 4 and 16
4. `contig`, npm 4 and 16
5. `pairs` and `b3p1`, npm 4 and 16
6. `k4` and `alt`, npm 1 and 64
7. `k2`, npm 4 and 16

A1 and A2 run on items 1 and 2 only. A1 and A2 recurse in the same loop as A3.

Phase 5 splits the grid into 3 shares of equal projected cost. Share 1 holds items 1–2 with A1 and A2. Share 3 holds the exploratory arms. Each share lists its cells for every arm, code arms included.

### 9.2 Arms

All arms start from π_ref and use the same logged negatives. All arms get the same cap on gate checks per run.

| Arm | Method |
|---|---|
| B0 | π_ref, no change. |
| B1a | The frozen recipe on all negatives in one training run. |
| B1b | The method-map config with the lowest D on seeds 0–4 for the setting. Oracle-selected, so it is optimistic. |
| B4 | Iterative find-then-fine-tune: per cluster, the frozen recipe on that cluster's negatives only, with the locality penalty. |
| B5 | Gated recipe selection: per run, the gate checks the 3 method-map configs with the lowest D on seeds 0–4 for the setting, and the run keeps the best accepted one. It gets the same gate feedback as the agent, with no LLM. |
| R0 | Iterative sampling-time controller: per class-R cluster, fit a logistic detector on flow states at k_commit (decodes ending in the cluster's basins against the rest, no labels), and run the best primitive in code when it fires. The gate applies. |
| D1 | Branch-and-distill: the A3 loop with code in place of the LLM. Per cluster, code takes the best primitive from the branch table, makes rescue targets for the members and their basin supports, and runs `targeted_train` with BC to those targets and the locality penalty. It submits a fixed list of 3 variants (set in Phase 2). |
| A1 | Agent with `targeted_train` only as its edit call. |
| A2 | Agent with the weight-writing calls only as its edit calls. |
| A3 | Agent with all edit calls. This is the practical system. |

A1, A2, and A3 get the same diagnostic tools: branch, inspect, sandbox, quick_check, and (from Phase 4) contrast and unit_card. The arms differ only in their edit calls.

### 9.3 Exploratory arms

Each exploratory arm changes one part of A3. They run in Phase 6, in share 3, on `k4`, npm 16, seeds 5–9.

| Arm | Change |
|---|---|
| A3-nobranch | No branch tool. The dossier has no branch table. |
| A3-sonnet | The surgeon uses Sonnet. |
| A3-nogate | The gate accepts every edit. Code still measures each test. |

### 9.4 Seeds

Each arm runs once per grid cell. Seeds 0–4 are for development. Seeds 5–9 are for confirmation. No seed 5–9 result changes a method, a threshold, or a prompt.

### 9.5 Metrics and statistics

The primary metric is D on the final-evaluation pool, computed with the toypref code.

The secondary metrics are:

- Removed and Collateral
- the mass between modes
- the fix rate per cluster: the share of first-atlas clusters whose residual basin mass falls by half or more
- cost, tokens, and decodes per accepted edit
- edit size: fraction of weights changed and units added
- accepted edits split by failure class and by edit call
- the gap between gate gain and final-evaluation gain for each accepted edit.

Per setting, report paired differences in the method map's format: mean ± standard error over seeds and npm values, and "lower in n of N".

For the claims, the unit is one (setting, seed) pair, with D averaged over npm. Cells that share a seed share π_ref, so also report a seed-level test (D averaged over settings and npm, n = 5) as a robustness check.

Use a one-sided paired t-test for T1 and T2 and a two-sided paired t-test for T3 and T4. Apply Holm correction over all comparisons in T1–T4 together: A3 against B1a, A3 against B5, A3 against D1, A2 against A1, and A2 against R0.

Report B1b next to B1a and B5 in every table. B1b is not a claim comparator, because its oracle selection makes it optimistic.

Before Phase 6, `PREREG.md` states the minimum detectable D difference of each comparison at 80% power, from the Phase 4 variance.

The claim labels for T1 and T2:

- Supported: the corrected test rejects in the claimed direction, and the arm's mean Collateral is not above its comparator's by more than 0.02.
- Unsupported: the 95% interval excludes a D improvement larger than 0.02.
- Inconclusive: neither of the two.

The claim labels for T3 and T4:

- Supported: the corrected test rejects. Report the direction.
- Unsupported: the 95% interval of the difference lies inside ±0.02.
- Inconclusive: neither of the two.

### 9.6 Report contents

- Table 1: D, Removed, and Collateral per arm and setting, in the method-map layout, seeds 5–9.
- Table 2: pilot results for B4, R0, D1, A1, and A2.
- Table 3: exploratory arm results.
- Table 4: each accepted A3 edit with its cluster, failure class, edit call, size, gate gain, and final gain.
- Table 5: cost, tokens, decodes, and wall time per arm.
- Figure 1: D per setting for all main arms, with standard-error bars.
- Figure 2: k_commit and the rescue curves of the pilot clusters.
- `results/method_map_rows.json`: the agent arms in the `DATA` format of the Toypref Method Map page, so the page can show them.

## 10. Gate and pre-registration

The gate is code. No LLM role can change it.

### 10.1 Gate batch

Each gate check draws a fresh batch of N_G = 16,384 pairs (c, ξ) from the gate pool. It decodes each pair with π_ref, the current version, and the new version. The oracle labels each action.

The target modes of a cluster are the majority oracle modes of its members. Only the gate knows them.

The gate has two scopes. Cluster scope is for per-cluster edits (agent arms, B4, R0, D1): the target is the cluster's modes. Global scope is for edits that use all negatives at once (B1a, B1b, B5): the target is the union of all first-atlas clusters' modes, and test 4 does not apply.

All masses are over the whole gate batch, so over all c, not only over the members' c range. A repair must hold at every condition.

### 10.2 Gate tests

All intervals come from a paired bootstrap over the batch with 2,000 resamples.

The gate accepts an edit only if all six tests pass:

1. Target: the mass on the target modes falls by at least 50% of its current value, and the 95% lower bound of the paired fall is above 0.
2. Progress: the one-sided 95% upper bound of D(new) − D(current) is below 0.
3. Drift: the one-sided 95% upper bound of Collateral(new, π_ref) is at or below 0.10.
4. Other forgotten modes: the one-sided 95% upper bound of the rise in mass on non-target forgotten modes is at or below 0.01.
5. Between modes: the one-sided 95% upper bound of the rise in mass between modes is at or below 0.01.
6. Unit tests: correct shapes, no NaN values, network inputs and activations only, and the graft identity for each graft.

The surgeon sees what Section 7.2 states: pass or fail for each test and the number of test 1.

The thresholds of tests 3–5 are development values. Phase 3 calibrates them before the pilot (Section 8). Phase 5 freezes them.

The ledger records each gate check with all its numbers, for accepts and rejects.

### 10.3 Reproducibility and power

A no-op edit (a graft with zero output weights) must give identical decodes. The toy is deterministic, so any difference is a bug.

Power check: add Gaussian noise to the last layer at the scale that raises D by 0.01 on 65,536 separate samples. Test 2 must reject this perturbation in at least 80% of 20 fresh gate batches. If it does not, double N_G and repeat.

### 10.4 Adaptive reuse

Fresh gate batches stop overfitting to fixed samples. The surgeon still gets gate feedback on the same distribution, so the gate is not a generalization guarantee. B5 gets the same feedback with no LLM. The final-evaluation pool gives no feedback to any loop.

### 10.5 Pre-registration

Write these items in `PREREG.md` in Phase 5:

- claims T1–T4, the exploratory comparisons, the metrics, and the statistics
- the gate tests, N_G, and the bootstrap method
- the miner's clustering rule and cut, the open-cluster threshold, and the branch thresholds 0.2 and 50%, M = 32, and the primitive list
- the D1 and B5 variant lists
- the session limits, the cost caps, and the subagent models
- the grid cells in order.

## 11. Risks and responses

| Risk | Signal | Response |
|---|---|---|
| Reproduction fails | Phase 1 exit fails | Stop and report to the user. |
| The surgeon can reach the oracle | The hook does not block, or a transcript audit hits | Stop and report to the user. |
| The miner joins different modes | Purity below 70% in the ledger | Report purity per cluster. Keep the run. |
| The branch finds no rescue | Class U holds more than half of the clusters | Report it. Add primitives only before Phase 5. |
| The surgeon fits the gate | Gate gain exceeds final gain by more than 0.05 in D | Report the gap for each accepted edit. |
| Baselines win | T1 unsupported | Report the result. It answers the question. |
| Cost is too high | Pilot cost per session above $5 | Report to the user before Phase 4. |

The escalation triggers: reproduction fails, the oracle is reachable, the pilot fails its decision rule, and pilot cost per session above $5. At each one, stop work and report to the user.

## 12. Decisions (defaults apply when the user starts Wave 0)

| # | Decision | Default | Alternative |
|---|---|---|---|
| TD1 | LLM roles | Claude Code subagents inside the cloud session; the credit pays | Anthropic API with a key; the credit does not pay |
| TD2 | Main testbed | Conditional toy; unconditional `base_k2` as smoke test only | Full grids on both toys |
| TD3 | Surgeon knowledge | Policy interface only; no ring geometry, no mode count (it can discover both by sampling π_ref) | Dossier describes the ring |
| TD4 | Gate feedback | Pass or fail per test, plus the target-mass fall; B5 gets the same feedback | All gate numbers, or accept or reject only |
| TD5 | After the pilot passes | Continue without waiting for the user | Wait for the user |
| TD6 | Pilot cells | `k4` and `alt`, npm 4 and 16, seeds 0–2 | Other settings |
| TD7 | Seeds | Development 0–4, confirmation 5–9, as in the method map | Other split |

## 13. Sources

- Plan: Agentic Policy Repair on Isaac Lab Factory GearMesh (the parent plan)
- Toypref Method Map (method definitions and reference values)
- Weight-Space Programming (Takehi, Teoh, Zhang, Khattab), https://rikiyat.github.io/blog/weight-space-programming/
- Claude Code subagents, https://code.claude.com/docs/en/sub-agents
- Claude API pricing, https://platform.claude.com/docs/en/about-claude/pricing
