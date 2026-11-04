#!/usr/bin/env python3
"""Write results/pilot/manifest_<round>_llm_eval.jsonl: one eval_llm unit per finished A1/A2 pilot run
(its surgeon session recorded). Rerun as sessions finish; done units are skipped by jobs.py."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import toyrepair  # noqa
from toyrepair import orchestrate as orc
rnd = sys.argv[1]
lines = []
for d in sorted((REPO / "runs" / "repair").glob(f"pilot_{rnd}_cond_*_A[12]")):
    run = orc.load_run(d.name)
    if orc.load_budget(run).get("sessions", 0) < 1 or run.pending:
        continue
    key = f"cond_{run.setting}_s{run.seed}_n{run.npm}"
    out = f"results/pilot/{rnd}/{run.setting}/s{run.seed}/n{run.npm}/{run.arm}_{rnd}.json"
    lines.append({"id": f"eval_{d.name}", "out": out,
                  "cmd": ["python", "scripts/pilot.py", "eval_llm", "--round", rnd, "--arm", run.arm,
                          "--cell", key, "--run", d.name]})
p = REPO / "results" / "pilot" / f"manifest_{rnd}_llm_eval.jsonl"
p.write_text("".join(json.dumps(x) + "\n" for x in lines))
print(p, len(lines))
