# SURVEY (W0, 2026-11-04)

**VM.** 4 CPUs, 15 GB RAM, Ubuntu 24.04, Python 3.13.18, torch 2.14.0+cu130 (runs on CPU), numpy 2.2.6. The proxy blocks `download.pytorch.org`, so `cloud_setup.sh` falls back to the PyPI wheel, which pulls about 3 GB of `nvidia-*` libraries. A fresh install takes about 10 min (torch 3.4 min plus deps 6.4 min). That misses the 5 min target. Every later step of the script takes under 10 s. Fix: preinstall torch in the cloud environment's setup script.

**Base.** `origin/cond-reroute` is stale at c4f2217 and holds no testbed. `agentic-repair-toy` branches from `origin/main` (d707fab), as the updated W0 section says.

## Entry points (PLAN 5.1): all verified

- `src/toypref/cond/task.py`: `CondMixture` (`centers`, `sample`, `assign` = oracle; OFF = −1 beyond 3·mode_std), `CondVelocityNet`, `CondFlow` (`sample`: 20 Euler steps, t_k = 1 − k/20, `x ← x − v/20`; `flow_loss`), `frozen`.
- `src/toypref/cond/experiment.py`: `train_base`, `make_pool`, `forget_modes`, `LAYOUTS`, `editset`, `ARMS` (40 arms), `train_arm`, `Bank`, `compute`, plus `gen`/`nprng` (seed·1,000,003 + purpose) and the purposes P_* .
- `src/toypref/cond/targets.py`: `basins_at` (128 decodes, 5-NN typicality q0.9, single linkage at eps = 3 × median 1-NN), `NegBank` (vote v(z), θ = 0.5 × median leave-one-out self rate), `row_gen(seed, uid, stream)`, `target_one` (hop, basin, …), `build`.
- `scripts/cond_base.py`, `scripts/cond_cell.py` (skip-if-exists; writes `$TOYPREF_RUNS/cond/cells[_layout]/s{seed}_k{k}_n{npm}/{arm}.json`).
- `toypref.config.RUNS` reads `TOYPREF_RUNS` at import time, so `import toyrepair` must come first. It sets `<repo>/runs`.

## Network (PLAN 5.1): verified, with 2 details the plan omits

`inp` Linear(2→128). 3 `ResidualBlock`s: `h + fc2(SiLU(fc1(cat(LayerNorm(h), cond))))`, where fc1 is 256→128 and fc2 is 128→128. Then `out_norm` LayerNorm, **SiLU**, and `out` Linear(128→2), initialised to zero. `cond` (128) = `time_mlp` (64: sinusoidal t·1000, 64 → Linear-SiLU-Linear) ‖ `c_mlp` (64: 8 Fourier frequencies sin/cos(π·k·c) → Linear-SiLU-Linear).

## Metrics: `compute(bank, pol)` keys

- **D** = `tv_to_ideal`.
- **Removed** = 1 − `neg_mass_ratio`.
- **Collateral** = `retain_tv`.

The reference's `collateral` matches `retain_tv` (k4 AMdist 0.1116 vs this cell's 0.1075). Note: `retain_tv` renormalises over the **retained modes only**; the between-modes mass (`mass_off`) is reported separately. PLAN Section 2 says "retained modes plus the region between modes". The gate (W1-A2) should implement the PLAN definition and also log `retain_tv` for map comparability.

Other keys: `mass_m0..7`, `mass_off`, `tv_to_ideal_ref`, `retain_noise_flip`, `neg_noise_*`, `neg_still_forget_{near,far}`. `Bank` uses fixed seeds (`gen(seed, P_EVAL)`, 20k pairs), so all arms see the same final samples (PLAN 5.3, verified).

## Timing (1 torch thread, CPU seconds)

| Item | Time |
|---|---|
| π_ref training | not retrained; `ckpt/` has seeds 0–9 (checksums OK). Estimate from 500 steps: 840 s for 40k steps. |
| one `compute` call | 2.4 s (20k decodes) |
| frozen-recipe arm, 5,000 steps (`train_s`) | 88 s wall |
| full `cond_cell.py` on s0_k4_n4 (pool 200k, editset, Bank, both arms) | 146 s wall (113 s user) |

## Reproduction, `cond_cell.py --seed 0 --n_forget 4 --n_neg 4` (forget = [2, 3, 4, 7])

| Arm | D | Removed | Collateral | mass_off | Reference mean, k4 seeds 0–4 |
|---|---|---|---|---|---|
| B0_frozen | 0.516 | 0 | 0 | 0.026 | 0.5156 |
| R_hopB_bsup_neg_AMdist | 0.259 | 0.638 | 0.108 | 0.047 | 0.3176 (removed 0.486, collateral 0.112) |

B0 matches. The frozen recipe on this one cell beats the 5-seed × 4-npm mean, which is plausible. No per-cell reference exists (README).

## Subagent transcripts and isolation (step 7)

- The agent-frontmatter `hooks:` in `.claude/agents/*.md` **do not fire** in the cloud session: surgeon, surgeon-sonnet and triage all ran `cat toyrepair/cell.py`. The models were right (opus/sonnet/sonnet) and the tools were Bash only.
- **Workaround:** `.claude/settings.json` registers one project-level PreToolUse hook (matcher `*`) → `tools/role_hook_dispatch.py`. It reads `agent_type` from the hook input. For `surgeon`/`surgeon-sonnet`/`triage` it allows `SubagentHandback`, blocks non-Bash tools, and runs the unchanged `tools/surgeon_hook.py`. It passes every other agent and the main agent (`agent_type` absent).
- **Re-test:** `cat` was blocked and `toyrepair-tool --help` ran, for all three roles. The hook input keys are `agent_id, agent_type, cwd, effort, hook_event_name, permission_mode, prompt_id, scratchpad_dir, session_id, tool_input, tool_name, tool_use_id, transcript_path`. A log of the keys only goes to `~/.claude/role_hook_log.jsonl`.
- **Caveat:** project settings load at session start (they reached new subagents here mid-session, but not the main agent's own calls).

- Main transcript: `~/.claude/projects/-home-user-preference-toy/<session-uuid>.jsonl`.
- Subagents: `~/.claude/projects/*/<session-uuid>/subagents/agent-<id>.jsonl`, with `agent-<id>.meta.json` (`agentType`, `description`, `model`).
- Usage sits on `type:"assistant"` events at `message.usage.{input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens}`, with `message.model`, `message.id`, and `requestId`. Streaming repeats entries, so dedupe by `message.id`.
- Fork transcripts start with a `fork-context-ref` line.
