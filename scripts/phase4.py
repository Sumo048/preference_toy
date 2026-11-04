#!/usr/bin/env python3
"""Phase 4 driver (W2b; docs/PLAN.md Section 8, Phase 4).

    python scripts/phase4.py manifest                # iterative B4, R0, D1 -> results/phase4/manifest_code.jsonl
    python scripts/phase4.py init                    # create the A3 runs (phase-4 tools on)
    python scripts/phase4.py eval --run <run id>     # final evaluation of an A3 run -> results/phase4/...
    python scripts/phase4.py report                  # results/phase4/table.md + table.json

Cells: cond k4, npm 4 and 16, seeds 0-4 (10). Code arms run the full loop (no max_rounds); A3 runs
recurse through ``toyrepair.orchestrate`` (triage -> branch -> surgeon session -> gate, repeated until
a stop rule). Every arm uses the calibrated gate (drift 0.15, other 0.02, between 0.02).
B0 and B1a come from the Phase 1 reproduction (``results/phase1``, the repo's code on the same cells).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import toyrepair  # noqa: E402,F401

CELLS = [("k4", s, n) for n in (4, 16) for s in range(5)]
CODE_ARMS = ("B4", "R0", "D1")
TAG = "_p4"
OUT = REPO / "results" / "phase4"


def key_of(setting, seed, npm):
    return f"cond_{setting}_s{seed}_n{npm}"


def a3_run_id(setting, seed, npm, arm="A3"):
    return f"p4_{key_of(setting, seed, npm)}_{arm}"


def out_path(arm, setting, seed, npm):
    return OUT / setting / f"s{seed}" / f"n{npm}" / f"{arm}{TAG}.json"


def manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    # D1 is the slowest arm: start it first
    for arm in ("D1", "B4", "R0"):
        for setting, seed, npm in CELLS:
            lines.append({"id": f"p4_{key_of(setting, seed, npm)}_{arm}",
                          "cmd": ["python", "scripts/run_arm.py", "--arm", arm, "--setting", setting, "--seed",
                                  str(seed), "--npm", str(npm), "--out", "results/phase4", "--tag", TAG],
                          "out": str(out_path(arm, setting, seed, npm).relative_to(REPO))})
    p = OUT / "manifest_code.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(p, len(lines), "units")


def init(arm="A3"):
    for setting, seed, npm in CELLS:
        rid = a3_run_id(setting, seed, npm, arm)
        if (Path(os.environ["TOYPREF_RUNS"]) / "repair" / rid / "run.json").exists():
            print("exists", rid)
            continue
        subprocess.run([sys.executable, "-m", "toyrepair.orchestrate", "init", "--run", rid, "--arm", arm,
                        "--grid", "cond", "--setting", setting, "--seed", str(seed), "--npm", str(npm),
                        "--phase4"], check=True)


def eval_run(run_id):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from toyrepair import eval_final, orchestrate as orc
    from toyrepair.cli import RealBackend
    run = orc.load_run(run_id)
    be = RealBackend()
    cell = be.cell(run)
    ver = be.version(run, run.current)
    r = eval_final.evaluate(cell, ver, run.arm)
    b = orc.load_budget(run)
    recs = orc.ledger_records(run)
    sess = [d for d in recs if d.get("kind") == "session" and (d.get("data") or {}).get("event") == "end"]
    tri = [d for d in recs if d.get("kind") == "triage" and (d.get("data") or {}).get("event") == "record"]
    vs = orc.versions(run)
    chain, v = [], run.current
    while v and v != "v0":
        chain.append({"version": v, "call": vs[v].get("call"), "patches": vs[v].get("patches")})
        v = vs[v]["parent"]
    gates = [d for d in recs if d.get("kind") == "gate"]
    rec = {"arm": run.arm, "cell": cell.key, "setting": run.setting, "seed": run.seed, "npm": run.npm,
           "run_id": run_id, "run_dir": run.dir,
           "output": {"version": run.current, "stop": run.stop, "invalid": run.invalid,
                      "accepted_chain": list(reversed(chain)), "gate_checks": b.get("gate_checks"),
                      "gate_cap": b.get("gate_cap"), "decodes": b.get("decodes"), "sessions": b.get("sessions"),
                      "cost_usd": b.get("cost_usd"), "n_first_clusters": run.n_first_clusters},
           "sessions": [{"session": d.get("session"), "cluster": (d.get("data") or {}).get("cluster"),
                         "summary": (d.get("data") or {}).get("summary"),
                         "calls": (d.get("data") or {}).get("calls"),
                         "submissions": (d.get("data") or {}).get("submissions"),
                         "accepted": (d.get("data") or {}).get("accepted"),
                         "cost_usd": ((d.get("data") or {}).get("usage") or {}).get("cost_usd"),
                         "usage": {k: ((d.get("data") or {}).get("usage") or {}).get(k)
                                   for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                             "cache_read_input_tokens", "peak_context", "requests",
                                             "context_exceeded")}}
                        for d in sess],
           "triage": [{"cluster": (d.get("data") or {}).get("cluster"),
                       "fallback": (d.get("data") or {}).get("fallback"),
                       "cost_usd": ((d.get("data") or {}).get("usage") or {}).get("cost_usd")} for d in tri],
           "gates": [{"version": d.get("version"), "cluster": (d.get("data") or {}).get("cluster"),
                      "call": (d.get("data") or {}).get("call"),
                      "view": (d.get("data") or {}).get("surgeon_view")} for d in gates],
           "final": {"source": "eval_final", "D": r.D, "removed": r.removed, "collateral": r.collateral,
                     "mass_off": r.mass_off, "collateral_plan": r.collateral_plan, "toypref": r.toypref,
                     "final_pool": r.final_pool}}
    p = out_path(run.arm, run.setting, run.seed, run.npm)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=float))
    print(p, "D", round(r.D, 4), "removed", round(r.removed, 3), "collateral", round(r.collateral, 4),
          "stop", run.stop, "sessions", b.get("sessions"), "cost", round(b.get("cost_usd") or 0, 2))


def _phase1(arm_file, setting, seed, npm):
    p = REPO / "results" / "phase1" / setting / f"s{seed}_{setting}_n{npm}" / arm_file
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    m = d.get("metrics", d)
    return {"D": m.get("tv_to_ideal"), "removed": 1 - m["neg_mass_ratio"] if "neg_mass_ratio" in m else None,
            "collateral": m.get("retain_tv")}


def report(arms=("B4", "R0", "D1", "A3")):
    import numpy as np
    rows = []
    for setting, seed, npm in CELLS:
        row = {"cell": key_of(setting, seed, npm), "seed": seed, "npm": npm}
        for name, f in (("B0", "B0_frozen.json"), ("B1a", "R_hopB_bsup_neg_AMdist.json")):
            row[name] = _phase1(f, setting, seed, npm)
        for arm in arms:
            p = out_path(arm, setting, seed, npm)
            if p.exists():
                d = json.loads(p.read_text())
                fin = d["final"]
                o = d["output"]
                row[arm] = {"D": fin["D"], "removed": fin["removed"], "collateral": fin["collateral"],
                            "mass_off": fin.get("mass_off"), "stop": o.get("stop"),
                            "accepted": len(o.get("accepted") or o.get("accepted_chain") or []),
                            "gate_checks": o.get("gate_checks"), "sessions": o.get("sessions"),
                            "cost_usd": o.get("cost_usd"), "wall_s": d.get("wall_s_total")}
        rows.append(row)
    names = ["B0", "B1a"] + list(arms)
    L = ["# Phase 4 (W2b): iterative arms on cond k4, npm 4 and 16, seeds 0-4", "",
         "D, Removed, Collateral = toypref `compute` (`tv_to_ideal`, 1 - `neg_mass_ratio`, `retain_tv`) on the "
         "final-evaluation bank. B0/B1a: Phase 1 reproduction (results/phase1). Mean ± SE over cells.", "",
         "| arm | cells | D | Removed | Collateral | accepted edits / run | gate checks / run | D < B1a |",
         "|---|---|---|---|---|---|---|---|"]
    for a in names:
        vals = [r[a] for r in rows if r.get(a) and r[a].get("D") is not None]
        if not vals:
            continue
        D = np.array([v["D"] for v in vals])
        se = D.std(ddof=1) / np.sqrt(len(D)) if len(D) > 1 else 0.0
        rem = np.mean([v["removed"] for v in vals if v.get("removed") is not None] or [np.nan])
        col = np.mean([v["collateral"] for v in vals if v.get("collateral") is not None] or [np.nan])
        acc = np.mean([v["accepted"] for v in vals if "accepted" in v] or [np.nan])
        gc = np.mean([v["gate_checks"] or 0 for v in vals if "gate_checks" in v] or [np.nan])
        lower = sum(1 for r in rows if r.get(a) and r.get("B1a") and r[a]["D"] < r["B1a"]["D"])
        L.append(f"| {a} | {len(vals)} | {D.mean():.3f} ± {se:.3f} | {rem:.3f} | {col:.3f} | "
                 f"{acc if not np.isnan(acc) else float('nan'):.2f} | {gc:.1f} | {lower}/{len(vals) if a != 'B1a' else 0} |")
    L += ["", "## Per cell (D)", "", "| cell | " + " | ".join(names) + " | A3 stop | A3 sessions | A3 $ |",
          "|" + "---|" * (len(names) + 4)]
    for r in rows:
        a3 = r.get("A3") or {}
        L.append(f"| {r['cell']} | " + " | ".join(f"{r[a]['D']:.3f}" if r.get(a) else "-" for a in names) +
                 f" | {a3.get('stop', '-')} | {a3.get('sessions', '-')} | "
                 f"{(a3.get('cost_usd') or 0):.2f} |")
    a3 = [r["A3"] for r in rows if r.get("A3")]
    if a3:
        cost = np.array([v.get("cost_usd") or 0 for v in a3])
        L += ["", f"A3 cost per run (surgeon + triage subagents): mean ${cost.mean():.2f}, sd {cost.std(ddof=1) if len(cost) > 1 else 0:.2f}, "
              f"total ${cost.sum():.2f} over {len(a3)} runs."]
    (OUT / "table.md").write_text("\n".join(L) + "\n")
    (OUT / "table.json").write_text(json.dumps(rows, indent=1, default=float))
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    sub.add_parser("init").add_argument("--arm", default="A3")
    sub.add_parser("eval").add_argument("--run", required=True)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        manifest()
    elif a.cmd == "init":
        init(a.arm)
    elif a.cmd == "eval":
        eval_run(a.run)
    elif a.cmd == "report":
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
