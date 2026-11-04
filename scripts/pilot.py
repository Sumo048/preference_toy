#!/usr/bin/env python3
"""Phase 3 pilot driver (W2a; docs/PLAN.md Section 8, Phase 3).

    python scripts/pilot.py targets [--cell KEY]     # pilot target per cell -> results/pilot/targets/<key>.json
    python scripts/pilot.py manifest --round r1 [--arms B4,R0,D1] [--cells ...]   # jobs.py manifest
    python scripts/pilot.py report --round r1        # Table 2 -> results/pilot/<round>/table2.md + .json

Pilot cells: cond k4 and alt, npm 4 and 16, seeds 0-2 (12 cells). The target of a cell is the open
cluster with the largest residual basin mass in the v0 atlas whose failure class is R (else the
next class-R cluster); every arm repairs that cluster once (no recursion, up to 3 gate checks).
The target file holds the cluster's uids (for the final evaluation's target modes), its card,
its branch curve (Figure 2) and its oracle purity. Oracle numbers here go to results only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import toyrepair  # noqa: E402,F401

PILOT_CELLS = [("k4", s, n) for n in (4, 16) for s in (0, 1, 2)] + [("alt", s, n) for n in (4, 16) for s in (0, 1, 2)]
CODE_ARMS = ("B4", "R0", "D1")
LLM_ARMS = ("A1", "A2")
TARGET_DIR = REPO / "results" / "pilot" / "targets"


def key_of(setting, seed, npm):
    from toyrepair.cell import cell_key
    return cell_key("cond", setting, seed, npm)


def parse_key(key):
    # cond_k4_s0_n4
    _, setting, s, n = key.split("_")
    return setting, int(s[1:]), int(n[1:])


def load(setting, seed, npm):
    from toyrepair.cli import cached_cell
    return cached_cell(types.SimpleNamespace(cell_args=("cond", setting, seed, npm)))


def targets(cells):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from toyrepair import flowstate, miner, oracle, policy
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for setting, seed, npm in cells:
        key = key_of(setting, seed, npm)
        out = TARGET_DIR / f"{key}.json"
        if out.exists():
            print("exists", out)
            continue
        t0 = time.time()
        cell = load(setting, seed, npm)
        v0 = policy.base_version(cell)
        atlas = miner.mine(cell, v0)
        tried, chosen = [], None
        for card in atlas.cards:
            cl = atlas.cluster_by_id(card.cluster)
            tab = flowstate.branch(cell, v0, cl)
            tried.append({"cluster": card.cluster, "r_cur": card.r_cur, "class": tab.failure_class,
                          "k_commit": tab.k_commit})
            if tab.failure_class == "R":
                chosen = (card, cl, tab)
                break
        rec = {"cell": key, "setting": setting, "seed": seed, "npm": npm, "n_open": len(atlas.cards),
               "n_clusters": len(atlas.clusters), "cluster_rule": miner.CLUSTER_RULE, "tried": tried,
               "wall_s": None}
        if chosen is None:
            rec.update(cluster=None, reason="no class-R cluster")
        else:
            card, cl, tab = chosen
            rec.update(cluster=int(card.cluster), uids=[int(u) for u in cl.uids], n=int(card.n_members),
                       r_ref=float(card.r_ref), cohesion=float(card.cohesion), k_commit=int(tab.k_commit),
                       failure_class=tab.failure_class,
                       best_primitive=getattr(tab.best_primitive, "name", None),
                       curve=[{"k": r.k, "t": r.t, "p": r.p, "ci": list(r.ci), "n": r.n} for r in tab.rows],
                       purity=float(oracle.purity(cell, [int(u) for u in cl.uids])),
                       target_modes=[int(m) for m in oracle.target_modes(cell, [int(u) for u in cl.uids])],
                       rank_by_residual=[c.cluster for c in atlas.cards].index(card.cluster))
        rec["wall_s"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(rec, indent=1))
        print(key, rec.get("cluster"), rec.get("failure_class"), rec.get("purity"), rec["wall_s"], "s")


def manifest(rnd, arms, cells):
    path = REPO / "results" / "pilot" / f"manifest_{rnd}.jsonl"
    lines = []
    for setting, seed, npm in cells:
        key = key_of(setting, seed, npm)
        tf = TARGET_DIR / f"{key}.json"
        for arm in arms:
            out = f"results/pilot/{rnd}/{setting}/s{seed}/n{npm}/{arm}.json"
            lines.append({"id": f"{rnd}_{key}_{arm}",
                          "cmd": ["python", "scripts/run_arm.py", "--arm", arm, "--setting", setting, "--seed",
                                  str(seed), "--npm", str(npm), "--max_rounds", "1", "--targets", str(tf.relative_to(REPO)),
                                  "--out", f"results/pilot/{rnd}", "--tag", f"_{rnd}"],
                          "out": out.replace(f"{arm}.json", f"{arm}_{rnd}.json")})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(path, len(lines), "units")


# ---------------------------------------------------------------- LLM arm evaluation
def eval_llm(rnd, arm, setting, seed, npm, run_id):
    """Final evaluation of an A1/A2 pilot run (its current version) -> results/pilot/<rnd>/.../<arm>_<rnd>.json"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from toyrepair import eval_final, orchestrate as orc
    from toyrepair.cli import RealBackend
    run = orc.load_run(run_id)
    key = key_of(setting, seed, npm)
    tgt = json.loads((TARGET_DIR / f"{key}.json").read_text())
    be = RealBackend()
    cell = be.cell(run)
    ver = be.version(run, run.current)
    r = eval_final.evaluate(cell, ver, arm, target_clusters=[tgt["uids"]])
    b = orc.load_budget(run)
    sess = [d for d in orc.ledger_records(run, "session") if (d.get("data") or {}).get("event") == "end"]
    usage = [(d.get("data") or {}).get("usage") or {} for d in sess]
    rec = {"arm": arm, "cell": key, "setting": setting, "seed": seed, "npm": npm, "run_id": run_id,
           "run_dir": run.dir, "output": {"version": run.current, "stop": run.stop, "invalid": run.invalid,
                                          "accepted": [run.current] if run.current != "v0" else [],
                                          "gate_checks": b.get("gate_checks"), "decodes": b.get("decodes"),
                                          "sessions": b.get("sessions"), "cost_usd": b.get("cost_usd")},
           "sessions": [{"summary": (d.get("data") or {}).get("summary"), "calls": (d.get("data") or {}).get("calls"),
                         "submissions": (d.get("data") or {}).get("submissions"),
                         "accepted": (d.get("data") or {}).get("accepted")} for d in sess],
           "usage": usage,
           "final": {"source": "eval_final", "D": r.D, "removed": r.removed, "collateral": r.collateral,
                     "mass_off": r.mass_off, "target_removed": r.target_removed, "target_modes": r.target_modes,
                     "collateral_plan": r.collateral_plan, "toypref": r.toypref, "final_pool": r.final_pool}}
    out = REPO / "results" / "pilot" / rnd / setting / f"s{seed}" / f"n{npm}" / f"{arm}_{rnd}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float))
    print(out, "D", round(r.D, 4), "target_removed", r.target_removed, "collateral", round(r.collateral, 4))


# ---------------------------------------------------------------- report
def _b0(setting, seed, npm):
    """B0 D on the cell (phase-1 cond_cell JSON, same toypref evaluation)."""
    p = REPO / "results" / "phase1" / setting / f"s{seed}_{setting}_n{npm}" / "B0_frozen.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("tv_to_ideal", d.get("metrics", {}).get("tv_to_ideal"))
    return None


def report(rnd, arms, cells):
    import numpy as np
    rows, missing = [], []
    for setting, seed, npm in cells:
        key = key_of(setting, seed, npm)
        tf = TARGET_DIR / f"{key}.json"
        tgt = json.loads(tf.read_text()) if tf.exists() else {}
        for arm in arms:
            p = REPO / "results" / "pilot" / rnd / setting / f"s{seed}" / f"n{npm}" / f"{arm}_{rnd}.json"
            if not p.exists():
                missing.append(f"{key}/{arm}")
                continue
            d = json.loads(p.read_text())
            f, o = d["final"], d["output"]
            d0 = (f.get("toypref") or {}).get("tv_to_ideal_ref")
            rows.append({"cell": key, "arm": arm, "cluster": tgt.get("cluster"), "purity": tgt.get("purity"),
                         "k_commit": tgt.get("k_commit"), "D": f["D"], "D_ref": d0, "removed": f["removed"],
                         "collateral": f["collateral"], "target_removed": f.get("target_removed"),
                         "accepted": bool(o.get("accepted")), "gate_checks": o.get("gate_checks"),
                         "decodes": o.get("decodes"), "cost_usd": o.get("cost_usd", 0.0),
                         "wall_s": d.get("wall_s_total"), "invalid": o.get("invalid")})

    def ok(r):   # PLAN Phase 3 decision rule, per cell and arm
        return (r["target_removed"] is not None and r["target_removed"] >= 0.5 and r["D_ref"] is not None
                and r["D"] < r["D_ref"] and r["collateral"] <= 0.10 and not r["invalid"])

    lines = [f"# Pilot round {rnd}: Table 2", "",
             "Per cell and arm: final D (toypref eval bank), π_ref's D, Removed, Collateral (method-map "
             "`retain_tv`), removed share of the target cluster's oracle modes, gate accept, and whether the "
             "cell meets the decision rule (target removed ≥ 0.5, D < D(π_ref), Collateral ≤ 0.10).", "",
             "| cell | arm | cluster | purity | D | D π_ref | Removed | Collateral | target removed | accepted | rule |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['cell']} | {r['arm']} | {r['cluster']} | {fmt(r['purity'])} | {fmt(r['D'])} | "
                     f"{fmt(r['D_ref'])} | {fmt(r['removed'])} | {fmt(r['collateral'])} | "
                     f"{fmt(r['target_removed'])} | {'yes' if r['accepted'] else 'no'} | {'✓' if ok(r) else ''} |")
    lines += ["", "## Summary per arm (mean over cells)", "",
              "| arm | cells | D | Removed | Collateral | target removed | accepted | cells meeting rule |",
              "|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        m = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None  # noqa: E731
        summ[arm] = {"cells": len(rs), "D": m("D"), "removed": m("removed"), "collateral": m("collateral"),
                     "target_removed": m("target_removed"), "accepted": sum(r["accepted"] for r in rs),
                     "rule_cells": sum(ok(r) for r in rs)}
        s = summ[arm]
        lines.append(f"| {arm} | {s['cells']} | {fmt(s['D'])} | {fmt(s['removed'])} | {fmt(s['collateral'])} | "
                     f"{fmt(s['target_removed'])} | {s['accepted']} | {s['rule_cells']} of {s['cells']} |")
    llm = [summ[a]["rule_cells"] for a in ("A1", "A2") if a in summ]
    passed = bool(llm) and max(llm) >= 8
    lines += ["", f"**Decision rule** (A1 or A2 meets the rule in ≥ 8 of 12 cells): "
              f"{'PASS' if passed else 'FAIL'} (A1 {summ.get('A1', {}).get('rule_cells')}, "
              f"A2 {summ.get('A2', {}).get('rule_cells')})."]
    if missing:
        lines += ["", f"Missing: {len(missing)} ({', '.join(missing[:30])})"]
    od = REPO / "results" / "pilot" / rnd
    od.mkdir(parents=True, exist_ok=True)
    (od / "table2.md").write_text("\n".join(lines) + "\n")
    (od / "table2.json").write_text(json.dumps({"rows": rows, "summary": summ, "decision_pass": passed,
                                                 "missing": missing}, indent=1))
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("targets")
    p.add_argument("--cell", action="append", default=None, help="cell key, e.g. cond_k4_s0_n4")
    p = sub.add_parser("manifest")
    p.add_argument("--round", required=True)
    p.add_argument("--arms", default=",".join(CODE_ARMS))
    p.add_argument("--cell", action="append", default=None)
    p = sub.add_parser("eval_llm")
    p.add_argument("--round", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--run", required=True)
    p = sub.add_parser("report")
    p.add_argument("--round", required=True)
    p.add_argument("--arms", default="B4,R0,D1,A1,A2")
    p.add_argument("--cell", action="append", default=None)
    a = ap.parse_args(argv)
    cells = [parse_key(k) for k in a.cell] if getattr(a, "cell", None) and a.cmd != "eval_llm" else PILOT_CELLS
    if a.cmd == "targets":
        targets(cells)
    elif a.cmd == "manifest":
        manifest(a.round, a.arms.split(","), cells)
    elif a.cmd == "eval_llm":
        s, seed, npm = parse_key(a.cell)
        eval_llm(a.round, a.arm, s, seed, npm, a.run)
    else:
        report(a.round, a.arms.split(","), cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
