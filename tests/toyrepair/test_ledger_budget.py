"""W1-A2: ledger and budget counters (docs/PLAN.md 6.9, 6.10, 7.2, 7.6)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from toyrepair.budget import Budget, BudgetExceeded, SessionCounter, usage_cost
from toyrepair.ledger import KINDS, Ledger, LedgerRecord, now_ts


def _gate_data(accepted, fall, fails=()):
    passed = {k: k not in fails for k in ("target", "progress", "drift", "other_forgotten", "between", "unit")}
    return {"accepted": accepted, "numbers": {"D_new": 0.123456, "mass_new": [0.1] * 9},
            "surgeon_view": {"accepted": accepted, "passed": passed, "target_fall": fall,
                             "target_fall_ci": [fall - 0.01, fall + 0.01]}}


def test_append_and_read(tmp_path):
    L = Ledger(str(tmp_path / "r1" / "ledger.jsonl"))
    L.write("run", "r1", {"cell": "cond_k4_s0_n4", "x": np.float32(1.5), "arr": np.arange(3), "nan": float("nan")})
    L.append(LedgerRecord(kind="note", run="r1", ts=now_ts(), data={"msg": "hi"}))
    with pytest.raises(ValueError):
        L.write("bogus", "r1")
    recs = list(L.records())
    assert [r.kind for r in recs] == ["run", "note"]
    assert recs[0].data["x"] == 1.5 and recs[0].data["arr"] == [0, 1, 2] and recs[0].data["nan"] is None
    assert [r.kind for r in L.records("note")] == ["note"]
    # append-only: lines are never rewritten; a torn last line is skipped
    with open(L.path, "a") as f:
        f.write('{"kind": "note", "run"')
    assert len(list(L.records())) == 2
    with open(L.path) as f:
        assert json.loads(f.readline())["kind"] == "run"


def test_summary_hides_oracle_numbers_and_caps(tmp_path):
    L = Ledger(str(tmp_path / "ledger.jsonl"))
    L.write("session", "r", {"summary": "tried a graft", "accepted": False}, session="s1")
    L.write("edit", "r", {"call": "graft_gated_path", "args": {"layer": "blocks.1.fc2", "n_units": 4},
                          "parent": "v0", "ok": True}, session="s1", version="v1")
    L.write("gate", "r", _gate_data(False, 0.01, ("target",)), session="s1", version="v1")
    L.write("edit", "r", {"call": "rank1", "args": {"layer": "out"}, "parent": "v0", "ok": True},
            session="s2", version="v2")
    L.write("gate", "r", _gate_data(True, 0.05), session="s2", version="v2")
    s = L.summary()
    assert "ACCEPT" in s and "rank1" in s and "graft_gated_path" in s and "failed: 1_target" in s
    assert "0.123456" not in s and "mass_new" not in s and "D_new" not in s
    assert s.index("rank1") < s.index("Recent rejected")
    # invalidate the accepted gate record
    L.invalidate("r", 4, "audit hit")
    assert "ACCEPT" not in L.summary() and L.invalid_lines() == {4: "audit hit"}
    for i in range(300):
        L.write("edit", "r", {"call": "set_units", "args": {"layer": "blocks.0.fc1"}, "parent": "v0"},
                version=f"w{i}")
        L.write("gate", "r", _gate_data(False, 0.0, ("target", "progress")), version=f"w{i}")
    s = L.summary(max_tokens=200, recent=300)
    assert len(s) <= 200 * 3.5 and "rows cut" in s


def test_ledger_has_no_heavy_imports():
    code = ("import sys, toyrepair.ledger, toyrepair.budget; "
            "bad=[m for m in sys.modules if m.startswith('toypref') or m.startswith('toyrepair.oracle') or m=='torch']; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_budget_roundtrip_and_rules(tmp_path):
    p = str(tmp_path / "budget.json")
    b = Budget(gate_cap=3)
    b.add_decodes(16384 * 3, "gate"); b.add_decodes(0.05, "inspect")
    assert [b.next_gate_check() for _ in range(3)] == [0, 1, 2]
    with pytest.raises(BudgetExceeded):
        b.next_gate_check()
    assert b.stop_reason() == "gate_cap" and b.gate_checks_left() == 0
    c = b.add_usage("claude-opus-5-5", input=1000, cache_write=10_000, cache_read=100_000, output=2000)
    assert abs(c - (0.004 + 0.05 + 0.02 + 0.04)) < 1e-12
    assert abs(usage_cost("opus", {"input_tokens": 1000, "cache_creation_input_tokens": 10_000,
                                   "cache_read_input_tokens": 100_000, "output_tokens": 2000}) - c) < 1e-12
    b.save(p)
    b2 = Budget.load(p)
    assert b2 == b and b2.decodes_by["gate"] == 16384 * 3
    b3 = Budget()
    for _ in range(3):
        b3.end_session(False)
    assert b3.stop_reason() == "no_accept_streak"
    b3.end_session(True)
    assert b3.no_accept_streak == 0 and b3.sessions == 4 and b3.stop_reason() is None
    assert b3.stop_reason(n_open_clusters=0) == "no_open_cluster"
    b3.cost_usd = 8.0
    assert b3.stop_reason() == "cost_cap"
    assert Budget.load_or_new(str(tmp_path / "none.json")).gate_checks == 0


def test_session_counter(tmp_path):
    s = SessionCounter(session="s1")
    for i in range(20):
        s.use_call()
    with pytest.raises(BudgetExceeded):
        s.use_call()
    for i in range(3):
        s.use_submission()
    with pytest.raises(BudgetExceeded):
        s.use_submission()
    p = str(tmp_path / "s.json")
    s.save(p)
    assert SessionCounter.load(p) == s and s.remaining() == {"cli_calls": 0, "gate_submissions": 0}


def test_kinds():
    assert "gate" in KINDS and "invalidate" in KINDS
