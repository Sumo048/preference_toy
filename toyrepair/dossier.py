"""Dossier builder (docs/PLAN.md 7.3).

Static part (<= 4000 tokens): the policy interface (2-D action, scalar condition, 20-step
decode, layer names and sizes), the CLI and edit API reference. No ring geometry, no mode count.
Text: ``toyrepair/prompts/surgeon.md`` with the arm's command list and ``docs/EDIT_REFERENCE.md``
(W1-B; a placeholder until W2a merges it) filled in.
Variable part (<= 6000 tokens): the target cluster's failure card, its branch table, the ledger
summary, and the remaining gate checks and cost. Pieces are cut in reverse priority (ledger
summary first, then branch table rows); every cut states how many rows it removed.

Token counts are estimates: characters / CHARS_PER_TOKEN, a ratio measured on real
surgeon transcripts (``python -m toyrepair.transcripts ratio <transcript>``; see
docs/DONE_W1-D.md).

Owner: W1-D.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

STATIC_TOKEN_CAP = 4000
VARIABLE_TOKEN_CAP = 6000
CARD_TOKEN_CAP = 150
ATLAS_TOKEN_CAP = 1500
LEDGER_SUMMARY_TOKEN_CAP = 2000
BRANCH_ROW_CAP = 40
#: characters per token, measured on two real surgeon transcripts (W1-D, 2026-11-04): tool-result
#: characters / (context growth - previous output tokens) gave 1.95 and 1.84 (the 6.7k-char prompt
#: alone 1.81); 1.8 is used so the caps hold. See docs/DONE_W1-D.md.
CHARS_PER_TOKEN = 1.8

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT_REFERENCE = os.path.join(REPO, "docs", "EDIT_REFERENCE.md")

EDIT_REFERENCE_PLACEHOLDER = (
    "(Placeholder: W2a inserts docs/EDIT_REFERENCE.md here.) Edit calls create a candidate version from\n"
    "`--parent` (default: the current version) and print its id. Array options name `.npy` files in scratch.\n"
    "- graft_gated_path: adds n gated hidden units beside a layer; their output is added to the layer's\n"
    "  output only where the detector `w . features(det_source) + det_b > det_thr` fires (and k in det_k).\n"
    "- rank1: W <- W + s u v^T.  ablate_direction: project unit v out of a layer (side pre) or out of the\n"
    "  next layer's input (side post).  set_units: write selected units' input rows, bias, outgoing columns.\n"
    "- targeted_train: gradient steps on a parameter subset with a locality anchor; objective bc (to\n"
    "  rescue targets), npo or kto on the cluster's negatives.")


@dataclass(frozen=True)
class Dossier:
    """static: text of the static part; variable: text of the variable part; arm: arm name
    (decides which commands are listed); tokens_static / tokens_variable: estimates."""
    static: str
    variable: str
    arm: str
    tokens_static: int = 0
    tokens_variable: int = 0

    def text(self) -> str:
        return self.static + "\n\n" + self.variable


def estimate_tokens(text: str) -> int:
    """Token estimate: ceil(chars / CHARS_PER_TOKEN)."""
    return int(-(-len(text) // CHARS_PER_TOKEN))


def cap_lines(lines: Sequence[str], max_rows: Optional[int] = None, max_tokens: Optional[int] = None,
              what: str = "rows") -> List[str]:
    """Keep leading lines within a row cap and a token cap; append a line that states how many
    were cut."""
    lines = list(lines)
    keep = lines if max_rows is None else lines[:max_rows]
    if max_tokens is not None:
        out, used = [], 0
        for ln in keep:
            t = estimate_tokens(ln + "\n")
            if used + t > max_tokens - 12:
                break
            out.append(ln)
            used += t
        keep = out
    cut = len(lines) - len(keep)
    if cut > 0:
        keep = keep + [f"[{cut} {what} cut]"]
    return keep


def cap_text(text: str, max_tokens: int, what: str = "lines") -> str:
    return "\n".join(cap_lines(text.split("\n"), max_tokens=max_tokens, what=what))


# ----------------------------------------------------------------------------- renderers
# The owners' ``render`` methods are used when implemented; these are the fallbacks.

def render_card(card) -> str:
    try:
        txt = card.render()
    except (NotImplementedError, AttributeError):
        k = "?" if card.k_commit is None else str(card.k_commit)
        txt = (f"cluster {card.cluster}: n={card.n_members} r_ref={card.r_ref:.3f} r_cur={card.r_cur:.3f} "
               f"c=[{card.c_range[0]:.2f},{card.c_range[1]:.2f}] coh={card.cohesion:.2f} "
               f"k_commit={k} class={card.failure_class} exemplars={','.join(str(e) for e in card.exemplars)}")
    return cap_text(txt, CARD_TOKEN_CAP)


def render_atlas(atlas, exclude: Sequence[int] = ()) -> str:
    cards = [c for c in atlas.cards if c.cluster not in set(exclude)]
    head = f"atlas of version {atlas.version_id}: {len(cards)} open clusters"
    if len(cards) != len(atlas.cards):
        head += f" ({len(atlas.cards) - len(cards)} more on cool-down, not listed)"
    return "\n".join(cap_lines([head] + [render_card(c) for c in cards], max_tokens=ATLAS_TOKEN_CAP,
                               what="cards"))


def render_branch(table) -> str:
    try:
        txt = table.render()
    except (NotImplementedError, AttributeError):
        lines = [f"branch table, cluster {table.cluster}: k_commit={table.k_commit} "
                 f"class={table.failure_class} decodes={table.n_decodes}",
                 "step  t     p(k)   95% CI         n    best primitives (rescue rate)"]
        for r in table.rows:
            best = "; ".join(f"{n} {v:.2f}" for n, v in r.best) or "-"
            lines.append(f"{r.k:>4}  {r.t:.2f}  {r.p:.3f}  [{r.ci[0]:.3f},{r.ci[1]:.3f}]  {r.n:>3}  {best}")
        if table.best_primitive is not None:
            lines.append(f"best primitive overall: {table.best_primitive.name}")
        txt = "\n".join(lines)
    return "\n".join(cap_lines(txt.split("\n"), max_rows=BRANCH_ROW_CAP + 3, what="table rows"))


def ledger_summary(run, max_tokens: int = LEDGER_SUMMARY_TOKEN_CAP) -> str:
    """W1-A2's ``Ledger.summary`` when implemented; else built here from the ledger records:
    recent sessions (cluster, submissions with pass/fail and test 1's number, result line) and
    all accepted edits. No oracle numbers beyond gate pass/fail and test 1."""
    from .ledger import Ledger
    try:
        return Ledger(os.path.join(run.dir, "ledger.jsonl")).summary(max_tokens)
    except NotImplementedError:
        pass
    from .orchestrate import ledger_records
    recs = ledger_records(run)
    accepted, sessions = [], {}
    for r in recs:
        k, d, sid = r.get("kind"), r.get("data") or {}, r.get("session")
        if k == "gate":
            sv = d.get("surgeon_view") or {}
            from .cli import GATE_TEST_LABELS
            fails = [GATE_TEST_LABELS.get(t, t) for t, ok in (sv.get("passed") or {}).items() if not ok]
            ci = sv.get("target_fall_ci") or [float("nan"), float("nan")]
            line = (f"  submit {r.get('version')}: {'ACCEPT' if sv.get('accepted') else 'reject'}"
                    f" fails={','.join(fails) or 'none'} target_fall={sv.get('target_fall', float('nan')):.4f}"
                    f" [{ci[0]:.4f},{ci[1]:.4f}]")
            sessions.setdefault(sid, {"lines": []})["lines"].append(line)
            if sv.get("accepted"):
                accepted.append(f"{r.get('version')} (session {sid}, {d.get('call', '?')}, cluster {d.get('cluster')})")
        elif k == "session" and d.get("event") == "start":
            sessions.setdefault(sid, {"lines": []})["cluster"] = d.get("cluster")
        elif k == "session" and d.get("event") == "end":
            s = sessions.setdefault(sid, {"lines": []})
            s["end"] = (d.get("summary") or "").strip().replace("\n", " ")[:160]
        elif k == "edit" and d.get("ok", True):
            sessions.setdefault(sid, {"lines": []})["lines"].append(
                f"  edit {r.get('version')}: {d.get('call')} frac={d.get('frac_weights_changed', 0):.4f} "
                f"units={d.get('units_added', 0)}")
    out = ["accepted edits: " + ("; ".join(accepted) if accepted else "none")]
    sess_lines = []
    for sid in sorted(sessions, key=lambda x: str(x)):
        s = sessions[sid]
        if "cluster" not in s:
            continue
        sess_lines.append(f"session {sid} cluster {s.get('cluster')}: {s.get('end', '(open)')}")
        sess_lines.extend(s["lines"])
    if sess_lines:
        # keep the most recent attempts: cut from the oldest
        kept = cap_lines(list(reversed(sess_lines)), max_tokens=max_tokens - estimate_tokens(out[0]) - 20,
                         what="older lines")
        note = kept[-1] if kept and kept[-1].endswith("cut]") else None
        body = list(reversed(kept[:-1] if note else kept))
        out.append("recent attempts:" + (f" {note}" if note else ""))
        out.extend(body)
    else:
        out.append("recent attempts: none")
    return "\n".join(out)


# ----------------------------------------------------------------------------- static / variable

def _command_reference(arm: str, phase4: bool = False) -> str:
    from .cli import USAGE, allowed_commands
    allowed = allowed_commands(arm, "surgeon", phase4)
    return "\n".join(f"- `{c}` {USAGE[c]}" for c in USAGE if c in allowed)


def edit_reference_for(ref: str, arm: str, phase4: bool = False) -> str:
    """Keep only the EDIT_REFERENCE paragraphs of edit calls this arm has (W2a pilot round 2:
    A1 surgeons spent calls building grafts their arm cannot submit). A paragraph belongs to
    call X when it starts with ``**X**``; a ``- `` list line continues the paragraph above it."""
    from .cli import EDIT_COMMANDS, allowed_commands
    allowed = set(allowed_commands(arm, "surgeon", phase4))
    out, keep = [], True
    for para in ref.split("\n\n"):
        head = para.lstrip()
        if head.startswith("**"):
            name = head[2:].split("**", 1)[0].strip()
            if name in EDIT_COMMANDS:
                keep = name in allowed
                if not keep:
                    continue
            else:
                keep = True
        out.append(para)
    return "\n\n".join(out)


def static_part(arm: str, phase4: bool = False) -> str:
    from .cli import SESSION_CALLS, SESSION_SUBMISSIONS
    with open(os.path.join(PROMPT_DIR, "surgeon.md")) as fh:
        tpl = fh.read()
    if os.path.exists(EDIT_REFERENCE):
        with open(EDIT_REFERENCE) as fh:
            ref = fh.read().strip()
    else:
        ref = EDIT_REFERENCE_PLACEHOLDER
    ref = edit_reference_for(ref, arm, phase4)
    from .cli import EDIT_COMMANDS, allowed_commands
    allowed = allowed_commands(arm, "surgeon", phase4)
    calls = ", ".join(f"`{c}`" for c in EDIT_COMMANDS if c in allowed) or "none"
    return (tpl.replace("{{COMMANDS}}", _command_reference(arm, phase4))
               .replace("{{EDIT_CALLS}}", calls)
               .replace("{{EDIT_REFERENCE}}", ref)
               .replace("{{CALLS}}", str(SESSION_CALLS))
               .replace("{{SUBMISSIONS}}", str(SESSION_SUBMISSIONS))).strip()


def triage_static() -> str:
    with open(os.path.join(PROMPT_DIR, "triage.md")) as fh:
        return fh.read().strip()


def variable_part(run, session) -> str:
    """Failure card, branch table, ledger summary, remaining gate checks and cost (in that order)."""
    from .orchestrate import branch_path, get_atlas, load_budget, load_pickle
    atlas = get_atlas(run, session.version_at_start)
    card = next((c for c in atlas.cards if c.cluster == session.cluster), None)
    head = (f"# Dossier: run {run.id}, session {session.session}, arm {run.arm}\n"
            f"Current version: {run.current}. Target cluster: {session.cluster}. Scratch directory: {run.scratch}")
    card_txt = "## Failure card\n" + (render_card(card) if card else f"cluster {session.cluster}: no card (closed)")
    bp = branch_path(run, session.session)
    if run.arm == "A3-nobranch":
        branch_txt = ""
    elif os.path.exists(bp):
        branch_txt = "## Branch table\n" + render_branch(load_pickle(bp))
    else:
        branch_txt = "## Branch table\n(not computed)"
    b = load_budget(run)
    cap = b.get("gate_cap") or 0
    budget_txt = (f"## Budget\nRun gate checks left: {max(cap - b['gate_checks'], 0) if cap else '?'} of {cap or '?'}. "
                  f"Run cost so far: ${b['cost_usd']:.2f} of ${run.cost_cap_usd:.2f}. "
                  f"This session: {session.calls} of {_calls()} calls used, "
                  f"{session.submissions} of {_subs()} submissions used.")
    fixed = estimate_tokens("\n\n".join([head, card_txt, branch_txt, budget_txt])) + 20
    room = VARIABLE_TOKEN_CAP - fixed
    if room < 200:      # cut the branch table first if it is huge (it is <= 40 rows, so rare)
        branch_txt = cap_text(branch_txt, max(VARIABLE_TOKEN_CAP // 2, 200), "table rows")
        fixed = estimate_tokens("\n\n".join([head, card_txt, branch_txt, budget_txt])) + 20
        room = VARIABLE_TOKEN_CAP - fixed
    led = "## Ledger summary\n" + ledger_summary(run, min(LEDGER_SUMMARY_TOKEN_CAP, max(room, 50)))
    parts = [head, card_txt] + ([branch_txt] if branch_txt else []) + [led, budget_txt]
    return "\n\n".join(parts)


def _calls():
    from .cli import SESSION_CALLS
    return SESSION_CALLS


def _subs():
    from .cli import SESSION_SUBMISSIONS
    return SESSION_SUBMISSIONS


def build(run, session) -> Dossier:
    st = static_part(run.arm, run.phase4_tools)
    var = variable_part(run, session)
    return Dossier(static=st, variable=var, arm=run.arm, tokens_static=estimate_tokens(st),
                   tokens_variable=estimate_tokens(var))


def triage_input(run) -> str:
    """The triage's input: static instructions, then the atlas of eligible clusters and the
    ledger summary."""
    from .orchestrate import get_atlas
    atlas = get_atlas(run)
    cool = [c for c, n in run.cooldown.items() if n > 0]
    return "\n\n".join([triage_static(), f"# Run {run.id}", "## Atlas\n" + render_atlas(atlas, exclude=cool),
                        "## Ledger summary\n" + ledger_summary(run)])
