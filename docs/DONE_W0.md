STATUS: PASS-WITH-ISSUES — (1) agent-frontmatter hooks did not fire; fixed with a project-level hook, verified for all 3 roles; (2) on a fresh VM, cloud_setup.sh takes ~10 min, not 5, because of the torch download

# W0 summary (branch agentic-repair-toy, based on origin/main d707fab; origin/cond-reroute is stale)

1. **What works:**
   - `bash scripts/cloud_setup.sh`: it installs, checksums `ckpt`, copies it to `runs/`, creates user `sandbox`, sets the repo to mode 700, and checks isolation. It is idempotent and takes 10 s once torch is installed.
   - `toyrepair/` scaffold, every module of 6.11. `docs/INTERFACES.md` maps each module to its owner session (W1-A1, A2, B, C, D).
   - `cell.load_cell`: cond and uncond. Its negatives and vote are bit-identical to toypref.
   - `pools` seeds; `toyrepair-tool --help`.
   - `scripts/jobs.py` (run/ensure/status) and `python -m toyrepair.transcripts spend`.
   - Tests: 17 passed (`pytest tests/toyrepair tests/infra`, about 50 s).
2. **Reproduction, s0_k4_n4, forget [2,3,4,7]:**
   - B0: D 0.516 (map mean 0.5156).
   - Frozen recipe `R_hopB_bsup_neg_AMdist`: D 0.259, Removed 0.638, Collateral 0.108, mass_off 0.047. Map mean (seeds 0–4) is 0.318.
3. **Timing, 1 thread:**
   - `compute`: 2.4 s.
   - Frozen-recipe arm: 88 s.
   - Full `cond_cell` (2 arms): 146 s.
   - π_ref training was not rerun; the estimate is about 840 s for 40k steps.
4. **Metric keys:** D = `tv_to_ideal`, Removed = 1 − `neg_mass_ratio`, Collateral = `retain_tv`. **Note for W1-A2:** `retain_tv` renormalises over the retained modes only. PLAN §2 adds the region between modes. Implement the PLAN version and log `retain_tv` too.
5. **Isolation:**
   - The `hooks:` in `.claude/agents/*.md` never ran; all 3 roles could `cat` the repo.
   - Fix: `.claude/settings.json` → `tools/role_hook_dispatch.py`. It dispatches on the hook input's `agent_type` to the unchanged `surgeon_hook.py`, and passes the main agent and other agents. Re-tested: `cat` blocked and `toyrepair-tool --help` ran for surgeon, surgeon-sonnet, and triage.
   - New sessions load it at start. Keep the transcript audit (§7.5) on as the second layer.
6. **Transcripts:**
   - Main: `~/.claude/projects/<proj>/<session>.jsonl`.
   - Subagents: `~/.claude/projects/<proj>/<session>/subagents/agent-<id>.jsonl` plus `.meta.json` (`agentType`, `model`).
   - Usage: `message.usage.{input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens}`. Dedupe by `message.id`.
7. **For W1-A1 and W1-B:** CPU decodes differ by up to 4e-7 with batch size. Bit-identity tests must use the same batch size and contents.
8. **Setup:** `download.pytorch.org` is blocked. torch 2.14.0 comes from PyPI with the CUDA libs, about 3 GB. Preinstalling it in the environment's setup script would bring `cloud_setup.sh` under 5 min.
9. **Deviations:**
   - Run dirs live at `$TOYPREF_RUNS/repair/<run id>/`.
   - `cell.neg` keeps toypref's order; the surgeon and triage see `cell.public_order`.
   - `miner.log_purity` may import the oracle, besides gate and eval_final.
10. **Measured cost:** main agent $2.81 and subagents $3.93 (about $6.7 total). Records are in `results/spend.jsonl`.
11. **Open interface requests:** none.
