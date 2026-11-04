"""Main-agent token usage and cost of this session from its Claude Code transcript (PLAN Sec. 2 prices,
Opus 5.5: $4 in, $5 cache write, $0.20 cache read, $20 out per M tokens). Stand-in for
`toyrepair.transcripts spend`, which is not on main."""
import glob, json, os, sys
P = {"in": 4.0, "cw": 5.0, "cr": 0.20, "out": 20.0}
files = sorted(glob.glob(os.path.expanduser("~/.claude/projects/-home-user-preference-toy/*.jsonl")), key=os.path.getmtime)
tot = {"in": 0, "cw": 0, "cr": 0, "out": 0}; seen = set()
for f in files[-1:]:
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = r.get("message") or {}
        u = m.get("usage") if isinstance(m, dict) else None
        if not u or m.get("id") in seen:
            continue
        seen.add(m.get("id"))
        tot["in"] += u.get("input_tokens", 0); tot["out"] += u.get("output_tokens", 0)
        tot["cw"] += u.get("cache_creation_input_tokens", 0); tot["cr"] += u.get("cache_read_input_tokens", 0)
cost = sum(tot[k] * P[k] / 1e6 for k in P)
print(json.dumps({"session": "E1", "transcript": os.path.basename(files[-1]) if files else None, **tot, "cost_usd": round(cost, 2)}))
