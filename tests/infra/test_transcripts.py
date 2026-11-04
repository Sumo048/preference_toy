"""Tests for toyrepair/transcripts.py on synthetic transcripts."""
import json

from toyrepair import transcripts as T


def _a(mid, model, inp, cw, cr, out):
    return {"type": "assistant", "requestId": "r" + mid,
            "message": {"id": mid, "model": model, "usage": {"input_tokens": inp, "cache_creation_input_tokens": cw,
                                                             "cache_read_input_tokens": cr, "output_tokens": out}}}


def test_usage_dedupe_cost_audit(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [_a("m1", "claude-opus-5-5", 10, 1000, 0, 5), _a("m1", "claude-opus-5-5", 10, 1000, 0, 50),
            _a("m2", "claude-sonnet-5-5", 0, 0, 1_000_000, 1000), {"type": "user", "message": {"content": "cat toyrepair/oracle/x.py"}}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    u = T.usage_from_transcript(p)
    assert u.models["claude-opus-5-5"].output_tokens == 50 and u.models["claude-opus-5-5"].n_messages == 1
    want = (10 * 4 + 1000 * 5 + 50 * 20) / 1e6 + (1_000_000 * 0.10 + 1000 * 10) / 1e6
    assert abs(T.cost(u) - want) < 1e-12
    hits = T.audit_transcript(p, T.DEFAULT_FORBIDDEN)
    assert len(hits) == 1 and "oracle" in hits[0]


def test_fork_excludes_parent_messages(tmp_path):
    parent = tmp_path / "sess.jsonl"
    parent.write_text(json.dumps(_a("p1", "claude-opus-5-5", 1, 1, 1, 1)) + "\n")
    sub = tmp_path / "sess" / "subagents"
    sub.mkdir(parents=True)
    f = sub / "agent-x.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [{"type": "fork-context-ref", "parentSessionId": "sess"},
                                                    _a("p1", "claude-opus-5-5", 1, 1, 1, 1),
                                                    _a("c1", "claude-opus-5-5", 2, 0, 0, 3)]) + "\n")
    u = T.usage_from_transcript(f)
    assert u.models["claude-opus-5-5"].n_messages == 1 and u.models["claude-opus-5-5"].output_tokens == 3
    assert T.find_subagent_transcripts(tmp_path.parent) == [f] or f in T.find_subagent_transcripts(tmp_path.parent)
