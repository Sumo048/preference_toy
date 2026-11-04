"""W1-D: toyrepair-tool, dossier, orchestrator, transcripts (mock backend; no torch decodes)."""
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from toyrepair import cli, dossier, orchestrate as orc, transcripts as tr


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("TOYPREF_RUNS", str(tmp_path))
    return tmp_path


def _surgeon(run_id="r1", arm="A3", **kw):
    orc.init_run(run_id, arm, "cond", "k4", 0, 4, backend="mock", **kw)
    act = _next(run_id)
    assert act.startswith("TRIAGE triage ")
    orc.record(act.split()[2], summary='{"cluster": 1, "expected_class": "R", "hypothesis": "h"}',
               transcript=os.devnull)
    act = _next(run_id)
    assert act.split()[0] == "SESSION"
    return act.split()[2]


def _next(run_id):
    r = orc.load_run(run_id)
    with orc.run_lock(r.dir):
        return orc.next_action(orc.load_run(run_id))


def _call(capsys, *argv, stdin=None):
    rc = cli.main(list(argv), stdin=io.StringIO(stdin) if stdin is not None else None)
    out = capsys.readouterr().out
    return rc, out


def _npy(run_id, name, arr):
    import numpy as np
    np.save(os.path.join(orc.load_run(run_id).scratch, name), np.asarray(arr, dtype="float32"))


# ----------------------------------------------------------------------------- CLI

def test_call_and_submission_limits(runs, capsys):
    tok = _surgeon()
    _npy("r1", "u.npy", [1, 0])
    _npy("r1", "v.npy", [0] * 128)
    for i in range(4):
        rc, out = _call(capsys, "rank1", "--token", tok, "--layer", "out", "--u", "u.npy", "--v", "v.npy", "--s", "1")
        assert rc == 0 and f"created candidate v{i + 1}" in out
    # mock gate rejects without a good quick_check: 3 submissions, then refused
    for i in range(3):
        rc, out = _call(capsys, "submit_edit", "--token", tok, "--version", f"v{i + 1}")
        assert rc == 0 and "REJECT" in out or "ACCEPT" in out
    rc, out = _call(capsys, "submit_edit", "--token", tok, "--version", "v4")
    assert rc == 2 and "submission limit" in out
    calls = 8
    while calls < cli.SESSION_CALLS:
        rc, _ = _call(capsys, "atlas", "--token", tok)
        assert rc == 0
        calls += 1
    rc, out = _call(capsys, "atlas", "--token", tok)
    assert rc == 2 and "call limit" in out
    tools = orc.ledger_records(orc.load_run("r1"), "tool")
    assert len(tools) == cli.SESSION_CALLS + 1          # every call has a ledger record, refused ones too
    assert all("<token>" in r["data"]["argv"] for r in tools)
    assert not any(tok in json.dumps(r) for r in tools)


def test_accept_closes_session_and_moves_current(runs, capsys):
    tok = _surgeon()
    _npy("r1", "u.npy", [1, 0])
    _npy("r1", "v.npy", [0] * 128)
    for _ in range(2):
        _call(capsys, "rank1", "--token", tok, "--layer", "out", "--u", "u.npy", "--v", "v.npy", "--s", "1")
    # find a candidate with mock still_bad < 0.5
    good = None
    for v in ("v1", "v2"):
        rc, out = _call(capsys, "quick_check", "--token", tok, "--version", v)
        if float(re.search(r"still_bad=([\d.]+)", out).group(1)) < 0.5:
            good = v
    assert good
    _call(capsys, "submit_edit", "--token", tok, "--version", good)
    rc, out = _call(capsys, "submit_edit", "--token", tok, "--version", good)
    assert "ACCEPT" in out
    assert orc.load_run("r1").current == good
    rc, out = _call(capsys, "atlas", "--token", tok)
    assert rc == 2 and "accepted" in out
    res = orc.record(tok, transcript=os.devnull)
    assert res["accepted"] == good
    b = orc.load_budget(orc.load_run("r1"))
    assert b["no_accept_streak"] == 0 and b["gate_checks"] == 2


def test_arm_command_sets(runs, capsys):
    tok = _surgeon("a1", arm="A1")
    rc, out = _call(capsys, "rank1", "--token", tok, "--layer", "out", "--u", "u.npy", "--v", "v.npy", "--s", "1")
    assert rc == 2 and "not available" in out
    rc, out = _call(capsys, "triage-input", "--token", tok)
    assert rc == 2
    tok = _surgeon("a2", arm="A2")
    rc, out = _call(capsys, "targeted_train", "--token", tok, "--params", "out", "--objective", "npo", "--steps", "5")
    assert rc == 2 and "not available" in out
    tok = _surgeon("nb", arm="A3-nobranch")
    rc, out = _call(capsys, "branch", "--token", tok)
    assert rc == 2
    rc, out = _call(capsys, "prompt", "--token", tok)
    assert "## Branch table" not in out and "`branch`" not in out
    rc, out = _call(capsys, "contrast", "--token", tok, "--layer", "blocks.1.act", "--k", "4")
    assert rc == 2            # Phase 4 tools off by default
    assert cli.allowed_commands("A3", "triage") == ("triage-input",)
    for arm in orc.LLM_ARMS:
        assert "submit_edit" in cli.allowed_commands(arm, "surgeon")


def test_triage_token_only_triage_input(runs, capsys):
    orc.init_run("t1", "A3", "cond", "k4", 0, 4, backend="mock")
    tok = _next("t1").split()[2]
    rc, out = _call(capsys, "atlas", "--token", tok)
    assert rc == 2
    rc, out = _call(capsys, "triage-input", "--token", tok)
    assert rc == 0 and '"cluster"' in out and "cluster 0" in out


def test_path_checks(runs, capsys, tmp_path):
    tok = _surgeon()
    scratch = orc.load_run("r1").scratch
    outside = tmp_path / "secret.npy"
    import numpy as np
    np.save(outside, np.ones(2, "f4"))
    os.symlink(outside, os.path.join(scratch, "link.npy"))
    for bad in ("../run.json", "/etc/passwd", str(outside), "link.npy", "a/../../ledger.jsonl"):
        rc, out = _call(capsys, "rank1", "--token", tok, "--layer", "out", "--u", bad, "--v", "v.npy", "--s", "1")
        assert rc == 2 and "outside your scratch" in out, bad
    rc, out = _call(capsys, "targeted_train", "--token", tok, "--params", "out", "--objective", "bc", "--steps", "5",
                    "--data", "rescue:../x.npy:bsup")
    assert rc == 2 and "outside your scratch" in out
    assert cli.check_path("sub/x.npy", scratch).startswith(os.path.realpath(scratch))


def test_tokens_never_look_like_options():
    for _ in range(2000):
        t = orc.new_token()
        assert t[0].isalnum() and re.fullmatch(r"[A-Za-z0-9]{16}", t)


def test_unknown_token_and_help(runs, capsys):
    assert cli.main(["atlas", "--token", "nope-nope-nope"]) == 2
    assert cli.main([]) == 0
    assert "submit_edit" in capsys.readouterr().out


def test_versions_visible_only_in_lineage(runs, capsys):
    tok = _surgeon()
    rc, out = _call(capsys, "quick_check", "--token", tok, "--version", "v7")
    assert rc == 2 and "unknown version" in out


def test_row_caps_state_cut_count():
    lines = [f"row {i}" for i in range(50)]
    out = dossier.cap_lines(lines, max_rows=40)
    assert len(out) == 41 and out[-1] == "[10 rows cut]"
    out = dossier.cap_lines(["x" * 350] * 100, max_tokens=1000, what="cards")
    assert out[-1].endswith("cards cut]")
    assert sum(dossier.estimate_tokens(l) for l in out) <= 1000
    long = "\n".join("y" * 100 for _ in range(500))
    capped = cli.cap_output(long, 2000)
    assert "output lines cut]" in capped and dossier.estimate_tokens(capped) <= 2000 + 20


# ----------------------------------------------------------------------------- dossier

WORDS_ONLY = re.compile(r"\bring\b|\bmodes?\b|\beight\b|forgotten|retained|oracle", re.I)
FORBIDDEN_WORDS = re.compile(r"\bring\b|\bmodes?\b|\b8\b|\beight\b|forgotten|retained|oracle", re.I)


def test_static_prompt_caps_and_content():
    for arm in orc.LLM_ARMS:
        for p4 in (False, True):
            st = dossier.static_part(arm, p4)
            assert dossier.estimate_tokens(st) <= dossier.STATIC_TOKEN_CAP, (arm, dossier.estimate_tokens(st))
            assert "{{" not in st
            assert not FORBIDDEN_WORDS.search(st), FORBIDDEN_WORDS.search(st).group(0)
    assert "targeted_train" not in dossier.static_part("A2").split("## Edit reference")[0]
    assert "graft_gated_path" not in dossier.static_part("A1").split("## Edit reference")[0]
    tri = dossier.triage_static()
    assert not FORBIDDEN_WORDS.search(tri)
    assert dossier.estimate_tokens(tri) < 1000


def test_variable_part_cap_with_long_ledger(runs, capsys):
    tok = _surgeon()
    run = orc.load_run("r1")
    for i in range(400):
        orc.ledger_append(run, "edit", {"call": "rank1", "frac_weights_changed": 0.1, "units_added": 0},
                          session="s002", version=f"v{i}")
        sv = {"accepted": i % 2 == 0, "passed": {"target": True, "other_forgotten": False},
              "target_fall": 0.1, "target_fall_ci": [0.05, 0.15]}
        orc.ledger_append(run, "gate", {"cluster": 1, "surgeon_view": sv}, session="s002", version=f"v{i}")
    rc, out = _call(capsys, "dossier", "--token", tok)
    assert rc == 0
    var = out.rsplit("\n[calls left", 1)[0]
    assert dossier.estimate_tokens(var) <= dossier.VARIABLE_TOKEN_CAP
    # W1-A2's Ledger.summary ("rows cut") or dossier's fallback ("older lines cut")
    assert "rows cut" in var or "older lines cut]" in var
    assert "forgotten" not in var and "4_other" in var
    order = [var.index(h) for h in ("## Failure card", "## Branch table", "## Ledger summary", "## Budget")]
    assert order == sorted(order)
    assert not WORDS_ONLY.search(var)


# ----------------------------------------------------------------------------- orchestrator

def test_stop_rules_streak_cooldown_and_fallback(runs, capsys):
    orc.init_run("s1", "A3", "cond", "k4", 0, 4, backend="mock")
    clusters = []
    for i in range(3):
        act = _next("s1")
        assert act.startswith("TRIAGE")
        res = orc.record(act.split()[2], summary="not json", transcript=os.devnull)
        assert res["fallback"]
        act = _next("s1")
        assert act.startswith("SESSION surgeon")
        assert _next("s1").startswith("PENDING")
        clusters.append(res["cluster"])
        orc.record(act.split()[2], transcript=os.devnull)
    # fallback = largest residual not on cool-down; failed clusters cool down for 2 iterations
    assert clusters == [0, 1, 2]
    assert _next("s1") == "DONE no_accept_streak"
    assert orc.status()[0].endswith("no_accept_streak")


def test_gate_cap_and_cost_cap(runs):
    orc.init_run("g1", "A3", "cond", "k4", 0, 4, backend="mock")
    _next("g1")
    run = orc.load_run("g1")
    b = orc.load_budget(run)
    assert b["gate_cap"] == orc.gate_check_cap(3) == 15
    b["gate_checks"] = 15
    orc.save_budget(run, b)
    run.pending = None
    orc.save_run(run)
    assert _next("g1") == "DONE gate_cap"
    orc.init_run("c1", "A3", "cond", "k4", 0, 4, backend="mock", cost_cap_usd=1.0)
    run = orc.load_run("c1")
    b = orc.load_budget(run)
    b["cost_usd"] = 1.5
    orc.save_budget(run, b)
    assert _next("c1") == "DONE cost_cap"


def test_no_open_cluster(runs, monkeypatch):
    monkeypatch.setattr(cli.MockBackend, "N_CLUSTERS", 0)
    orc.init_run("n1", "A3", "cond", "k4", 0, 4, backend="mock")
    assert _next("n1") == "DONE no_open_cluster"


def test_code_arm_and_sonnet_agent(runs):
    orc.init_run("b4", "B4", "cond", "k4", 0, 4, backend="mock")
    assert _next("b4").startswith("CODE python scripts/run_arm.py --arm B4")
    orc.init_run("son", "A3-sonnet", "cond", "k4", 0, 4, backend="mock")
    tok = _next("son").split()[2]
    orc.record(tok, summary='{"cluster": 2, "expected_class": "R", "hypothesis": "h"}', transcript=os.devnull)
    assert _next("son").startswith("SESSION surgeon-sonnet ")


def test_parse_triage():
    assert orc.parse_triage('{"cluster": 3, "expected_class": "R", "hypothesis": "a"}')["cluster"] == 3
    assert orc.parse_triage('Here: {"cluster": 2, "expected_class": "U", "hypothesis": "b"} done')["cluster"] == 2
    assert orc.parse_triage("no json") is None
    assert orc.parse_triage('{"cluster": "x"}') is None


def test_eight_concurrent_sessions(runs, capsys):
    toks = [_surgeon(f"c{i}") for i in range(8)]

    def work(tok):
        for _ in range(6):
            assert cli.main(["atlas", "--token", tok]) == 0
        return tok

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, toks))
    for i, tok in enumerate(toks):
        run, s = orc.lookup_token(tok)
        assert run.id == f"c{i}" and s.calls == 6
    assert len(orc.status()) == 8


def test_cli_subprocess_and_stdin_sandbox(runs):
    tok = _surgeon()
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-m", "toyrepair.cli", "sandbox", "--token", tok, "--stdin"],
                       input="print(6*7)\nimport subprocess\nsubprocess.run(['true'])\n", capture_output=True,
                       text=True, env=env, timeout=120)
    assert "42" in r.stdout and "blocked" in r.stdout


# ----------------------------------------------------------------------------- transcripts

def _write_transcript(path, token, model="claude-opus-5-5", extra_text=""):
    ev = [
        {"type": "user", "message": {"role": "user", "content": f"Run r1, token {token}. Start."}},
        {"type": "assistant", "requestId": "q1", "message": {"id": "m1", "model": model, "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "toyrepair-tool prompt"}}],
            "usage": {"input_tokens": 10, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
                      "output_tokens": 50}}},
        {"type": "assistant", "requestId": "q1", "message": {"id": "m1", "model": model, "content": [
            {"type": "text", "text": ""}],
            "usage": {"input_tokens": 10, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
                      "output_tokens": 50}}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "x" * 3000 + extra_text}]}},
        {"type": "assistant", "requestId": "q2", "message": {"id": "m2", "model": model, "content": [
            {"type": "text", "text": "RESULT no-accept - tried rank1"}],
            "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 1060,
                      "output_tokens": 20}}},
    ]
    with open(path, "w") as fh:
        for e in ev:
            fh.write(json.dumps(e) + "\n")


def test_transcript_report_dedupe_cost_audit(tmp_path):
    p = tmp_path / "proj" / "sess" / "subagents"
    p.mkdir(parents=True)
    t = p / "agent-a.jsonl"
    _write_transcript(t, "TOK123456789", extra_text=" see runs/repair/other_run/ledger.jsonl")
    (p / "agent-a.meta.json").write_text(json.dumps({"agentType": "surgeon", "model": "opus"}))
    rep = tr.session_report(t, run_id="r1", other_runs=["other_run", "r10"], expected_agent="surgeon")
    assert rep["n_requests"] == 2
    assert rep["tokens"]["output_tokens"] == 70
    exp = (15 * 4 + 2000 * 5 + 1060 * 0.2 + 70 * 20) / 1e6
    assert abs(rep["cost_usd"] - exp) < 1e-9
    assert len(rep["audit_hits"]) == 1 and "other_run" in rep["audit_hits"][0]
    assert rep["final_text"].startswith("RESULT no-accept")
    assert rep["agent_ok"] and not rep["context_exceeded"]
    assert tr.find_subagent_for_token("TOK123456789", root=tmp_path) == str(t)
    assert tr.find_subagent_for_token("absent-token", root=tmp_path) is None
    r = tr.chars_per_token(t)
    # tool result tokens = 2065 - 1010 - 50 = 1005; chars = 3000 + extra
    assert r["n"] == 1 and r["tokens"] == 1005
    # run id r1 must not match inside r10 or other words
    _write_transcript(t, "TOK", extra_text=" runs/repair/r10/x")
    assert tr.isolation_audit(t, run_id="r10", other_runs=["r1"]) == []
    _write_transcript(t, "TOK", extra_text=" open src/toypref/cond/task.py")
    assert tr.isolation_audit(t, run_id="r1")


def test_record_marks_run_invalid_on_audit_hit(runs, tmp_path):
    orc.init_run("v1run", "A3", "cond", "k4", 0, 4, backend="mock")
    orc.init_run("other", "A3", "cond", "k4", 0, 4, backend="mock")
    tok = _next("v1run").split()[2]
    t = tmp_path / "agent-x.jsonl"
    _write_transcript(t, tok, extra_text=" cat runs/repair/other/ledger.jsonl")
    res = orc.record(tok, transcript=str(t))
    assert res["audit_hits"] == 1
    assert _next("v1run") == "DONE invalid_audit"


def test_static_fits_with_full_edit_reference(tmp_path, monkeypatch):
    """EDIT_REFERENCE.md budget: 1,500 tokens (W1-B) until pilot round 2, then 1,200 tokens: the
    round-2 template text grew (W2a); the real reference is filtered per arm and every arm's
    static part is checked against the cap in test_static_prompt_caps_and_content."""
    ref = tmp_path / "EDIT_REFERENCE.md"
    ref.write_text("word " * int(1200 * dossier.CHARS_PER_TOKEN / 5))
    monkeypatch.setattr(dossier, "EDIT_REFERENCE", str(ref))
    for arm in orc.LLM_ARMS:
        st = dossier.static_part(arm, phase4=True)
        assert "word word" in st
        assert dossier.estimate_tokens(st) <= dossier.STATIC_TOKEN_CAP, (arm, dossier.estimate_tokens(st))


def test_every_documented_option_passes_the_hook():
    """Every option value in the command reference must pass tools/surgeon_hook.py's charset."""
    hook = re.compile(r"^toyrepair-tool(?: [A-Za-z0-9_\-\.=:/,\[\]]+)*$")
    for cmd, usage in cli.USAGE.items():
        for m in re.finditer(r"`([^`]*)`", usage):
            for alt in m.group(1).replace("[", " ").replace("]", " ").split():
                for v in alt.split("|"):
                    assert not v or hook.match(f"toyrepair-tool {cmd} {v}"), (cmd, v)
